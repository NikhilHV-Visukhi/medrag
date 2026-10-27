"""Evaluate retrieval (per configuration) and answers (per LLM), with an LLM judge and 95% confidence intervals."""
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from medrag.providers.llms import get_judge, get_llm
from medrag.rag.answer import RAG
from medrag.retrieval.retriever import Retriever

ABSTAIN_RE = re.compile(r"(no patient matching|not (be )?found|do(es)? not contain|no record|not in the records|no information|unable to find)", re.I)
NEGATIVE_RE = re.compile(r"\b(no|not|never|none|did not|has not|hasn't|without)\b", re.I)

JUDGE_PROMPT = """You are grading an answer from a clinical records assistant.

QUESTION: {q}
REFERENCE ANSWER (ground truth from the database): {ref}
RETRIEVED SOURCES GIVEN TO THE ASSISTANT:
{ctx}

ASSISTANT ANSWER:
{ans}

Score each from 1 (worst) to 5 (best):
- correctness: does the answer agree with the reference answer? (5 = same facts, 1 = wrong or missing)
- faithfulness: is every claim in the answer supported by the sources or computed facts? (5 = nothing invented)
- answer_relevance: does the answer address what was asked, without padding?
- context_relevance: do the retrieved sources contain what is needed to answer?
Reply with JSON only: {{"correctness": n, "faithfulness": n, "answer_relevance": n, "context_relevance": n, "reason": "one sentence"}}"""
JUDGE_KEYS = ["correctness", "faithfulness", "answer_relevance", "context_relevance"]


@dataclass
class Config:
    label: str
    retriever: Retriever
    mode: str = "hybrid+rerank"
    use_filters: bool = True
    expand: bool = True


def retrieval_metrics(ranked_docs: list[str], gold: list[str], k: int) -> dict:
    ranked = list(dict.fromkeys(ranked_docs))[:k]
    rel = [1 if d in gold else 0 for d in ranked]
    first = next((i for i, r in enumerate(rel) if r), None)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return {"hit": float(first is not None), "mrr": 1 / (first + 1) if first is not None else 0.0,
            "recall": sum(rel) / len(gold), "ndcg": dcg / idcg if idcg else 0.0}


def ci(values, n_boot: int = 1000, seed: int = 0) -> str:
    """Mean with a bootstrap 95% confidence interval, e.g. '0.81 [0.72-0.89]'."""
    v = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(v) == 0:
        return "-"
    if len(v) == 1 or v.min() == v.max():
        return f"{v.mean():.3f}"
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return f"{v.mean():.3f} [{np.percentile(means, 2.5):.2f}-{np.percentile(means, 97.5):.2f}]"


def eval_retrieval(items: list[dict], configs: list[Config], k: int = 8) -> pd.DataFrame:
    rows = []
    gold_items = [it for it in items if it["gold_docs"]]
    for c in configs:
        for it in tqdm(gold_items, desc=c.label, leave=False):
            t0 = time.time()
            r = c.retriever.search(it["question"], k=k, mode=c.mode, use_filters=c.use_filters, expand=c.expand)
            ranked = r.hits
            if not any(g.startswith("patient:") for g in it["gold_docs"]):
                ranked = ranked[ranked["via"] != "profile"]  # the pinned profile always ranks first; score the search itself
            rows.append({"config": c.label, "type": it["type"], "difficulty": it.get("difficulty", "easy"), "id": it["id"],
                         "in_dictionary": it.get("in_dictionary"), "latency_s": time.time() - t0,
                         **retrieval_metrics(ranked["doc_id"].tolist(), it["gold_docs"], k)})
    return pd.DataFrame(rows)


def _fact_hit(fact: str, text: str) -> bool:
    return any(alt.strip().lower() in text for alt in fact.split("|"))


def _judge(judge, it, ans) -> dict:
    ctx = (f"COMPUTED FACTS:\n{ans.facts}\n\n" if ans.facts else "") + "\n---\n".join(ans.result.hits["text"].head(8).tolist())
    prompt = JUDGE_PROMPT.format(q=it["question"], ref=it["answer"], ctx=ctx[:14000], ans=ans.text)
    out = {f"judge_{k}": None for k in JUDGE_KEYS} | {"judge_reason": None}
    try:
        raw = judge.complete("You are a strict, fair evaluator. Output JSON only.", prompt, 600)
        j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        out |= {f"judge_{k}": float(j[k]) for k in JUDGE_KEYS} | {"judge_reason": j.get("reason")}
    except Exception as e:  # a judge failure should not lose the answer's other metrics
        out["judge_reason"] = f"judge error: {str(e)[:150]}"
    return out


def score_answer(it: dict, text: str) -> dict:
    low = text.lower()
    row: dict = {}
    if it.get("facts"):
        row["fact_recall"] = sum(_fact_hit(f, low) for f in it["facts"]) / len(it["facts"])
    if "number" in it:
        n = it["number"]
        row["number_correct"] = float(bool(re.search(rf"(?<![\d,.])({n:,}|{n})(?![\d,]|\.\d)", text)))
    if it.get("expect_no"):
        row["negative_correct"] = float(bool(NEGATIVE_RE.search(text)) and not low.lstrip().startswith("yes"))
    row["abstain_correct"] = float(bool(ABSTAIN_RE.search(text)) == bool(it.get("abstain")))
    # one pass/fail per question from whichever deterministic check applies
    if "number_correct" in row:
        row["auto_correct"] = row["number_correct"]
    elif "negative_correct" in row:
        row["auto_correct"] = row["negative_correct"]
    elif it.get("abstain"):
        row["auto_correct"] = row["abstain_correct"]
    elif "fact_recall" in row:
        row["auto_correct"] = float(row["fact_recall"] == 1.0)
    return row


def eval_generation(items: list[dict], retriever: Retriever, llm_specs: list[str], use_judge: bool = False) -> pd.DataFrame:
    judge = get_judge() if use_judge else None
    rows = []
    for spec in llm_specs:
        rag = RAG(retriever, get_llm(spec))
        for it in tqdm(items, desc=f"answers | {spec}"):
            base = {"llm": spec, "type": it["type"], "difficulty": it.get("difficulty", "easy"), "id": it["id"], "question": it["question"],
                    "expected": it["answer"]}
            try:
                ans = rag.ask(it["question"])
            except Exception as e:  # keep going: one provider failing should not lose the whole run
                rows.append(base | {"error": str(e)[:200]})
                continue
            row = base | {"answer": ans.text, "retrieval_s": ans.retrieval_s, "generation_s": ans.generation_s,
                          "cites_sources": float(bool(re.search(r"\[S\d+\]", ans.text)))} | score_answer(it, ans.text)
            if judge:
                row |= _judge(judge, it, ans)
            rows.append(row)
    return pd.DataFrame(rows)


def report(ret_df: pd.DataFrame | None, gen_df: pd.DataFrame | None, out_dir, k: int) -> str:
    lines = ["# MedRAG evaluation", "", f"Run: {datetime.now():%Y-%m-%d %H:%M}", "",
             "Numbers are means with bootstrap 95% confidence intervals in brackets.", ""]
    if ret_df is not None and len(ret_df):
        lines += [f"## Retrieval (k={k})", ""]
        for diff, d in ret_df.groupby("difficulty"):
            tab = d.groupby("config").agg(**{f"Hit@{k}": ("hit", ci), "MRR": ("mrr", ci), f"Recall@{k}": ("recall", ci),
                                             f"nDCG@{k}": ("ndcg", ci), "Latency (s)": ("latency_s", lambda x: f"{x.mean():.2f}")})
            tab = tab.loc[d.groupby("config")["mrr"].mean().sort_values(ascending=False).index].reset_index()
            lines += [f"### {diff} questions ({d['id'].nunique()})", "", tab.to_markdown(index=False), ""]
        by_type = ret_df.pivot_table(index="config", columns="type", values="mrr", aggfunc="mean").round(3).reset_index()
        lines += ["### MRR by question type", "", by_type.to_markdown(index=False), ""]
        dic = ret_df.dropna(subset=["in_dictionary"])
        if len(dic):
            t = dic.pivot_table(index="config", columns="in_dictionary", values="recall", aggfunc="mean").round(3)
            t.columns = [f"wording in dictionary={c}" for c in t.columns]
            lines += ["### Recall on paraphrased questions, by whether the wording is in the expansion dictionary", "",
                      t.reset_index().to_markdown(index=False), ""]
    if gen_df is not None and len(gen_df):
        ok = gen_df[gen_df.get("error").isna()] if "error" in gen_df else gen_df
        cols = [c for c in ["auto_correct", "fact_recall", "number_correct", "negative_correct", "abstain_correct", "cites_sources",
                            "judge_correctness", "judge_faithfulness", "judge_answer_relevance", "judge_context_relevance"] if c in ok]
        tab = ok.groupby(["llm", "difficulty"]).agg(**{c: (c, ci) for c in cols},
                                                    **{"Gen. time (s)": ("generation_s", lambda x: f"{x.mean():.1f}")}).reset_index()
        lines += ["## Answers", "", tab.to_markdown(index=False), "",
                  "- auto_correct: pass/fail from the deterministic check that fits the question (exact number, all key facts, correct 'no', correct 'not found')",
                  "- judge_*: 1-5 scores from the judge model, which is a different model from the one answering", ""]
        by_type = ok.pivot_table(index="llm", columns="type", values="auto_correct", aggfunc="mean").round(2).reset_index()
        lines += ["### auto_correct by question type", "", by_type.to_markdown(index=False), ""]
        if "judge_correctness" in ok and ok["judge_correctness"].notna().any():
            both = ok.dropna(subset=["judge_correctness", "auto_correct"])
            agree = ((both["judge_correctness"] >= 4) == (both["auto_correct"] == 1)).mean()
            lines += [f"Judge vs deterministic check agreement: {agree:.0%} of {len(both)} answers "
                      "(judge correctness >= 4 counted as pass). Low agreement means one of the two needs review.", ""]
        if "error" in gen_df and gen_df["error"].notna().any():
            lines.append(f"**{gen_df['error'].notna().sum()} answers failed with errors**; see generation.csv.")
    md = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    if ret_df is not None:
        ret_df.to_csv(out_dir / "retrieval.csv", index=False)
    if gen_df is not None:
        gen_df.to_csv(out_dir / "generation.csv", index=False)
    return md
