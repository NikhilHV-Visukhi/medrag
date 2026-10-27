"""Evaluate retrieval (per index x mode) and answers (per LLM), and write a comparison report."""
import json
import math
import re
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from medrag.providers.llms import get_judge, get_llm
from medrag.rag.answer import RAG
from medrag.retrieval.retriever import MODES, Retriever

ABSTAIN_RE = re.compile(r"(no patient matching|not (be )?found|do(es)? not contain|no record|not in the records|no information|unable to find)", re.I)

JUDGE_PROMPT = """You are grading an answer from a clinical records assistant.

QUESTION: {q}
REFERENCE ANSWER (ground truth from the database): {ref}
RETRIEVED SOURCES GIVEN TO THE ASSISTANT:
{ctx}

ASSISTANT ANSWER:
{ans}

Score each from 1 (worst) to 5 (best):
- correctness: does the answer agree with the reference answer?
- faithfulness: is every claim supported by the sources (no invented facts)?
Reply with JSON only, e.g. {{"correctness": 4, "faithfulness": 5, "reason": "..."}}"""


def retrieval_metrics(ranked_docs: list[str], gold: list[str], k: int) -> dict:
    ranked = list(dict.fromkeys(ranked_docs))[:k]
    rel = [1 if d in gold else 0 for d in ranked]
    first = next((i for i, r in enumerate(rel) if r), None)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return {"hit": float(first is not None), "mrr": 1 / (first + 1) if first is not None else 0.0,
            "recall": sum(rel) / len(gold), "ndcg": dcg / idcg if idcg else 0.0}


def eval_retrieval(items: list[dict], retrievers: dict[str, Retriever], k: int = 8, modes=MODES) -> pd.DataFrame:
    rows = []
    gold_items = [it for it in items if it["gold_docs"]]
    for name, ret in retrievers.items():
        configs = [(m, True) for m in modes] + [("hybrid+rerank", False)]
        for mode, filt in configs:
            label = mode + ("" if filt else " (no filters)")
            for it in tqdm(gold_items, desc=f"{name} | {label}", leave=False):
                t0 = time.time()
                r = ret.search(it["question"], k=k, mode=mode, use_filters=filt)
                ranked = r.hits
                if not any(g.startswith("patient:") for g in it["gold_docs"]):
                    ranked = ranked[ranked["via"] != "profile"]  # the pinned profile always ranks first; score the search itself
                rows.append({"index": name, "mode": label, "type": it["type"], "id": it["id"], "latency_s": time.time() - t0,
                             **retrieval_metrics(ranked["doc_id"].tolist(), it["gold_docs"], k)})
    return pd.DataFrame(rows)


def _judge(judge, it, ans) -> dict:
    ctx = "\n---\n".join(ans.result.hits["text"].head(8).tolist())[:12000]
    raw = judge.complete("You are a strict evaluator. Output JSON only.", JUDGE_PROMPT.format(q=it["question"], ref=it["answer"], ctx=ctx, ans=ans.text), 400)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
        return {"judge_correctness": float(j.get("correctness")), "judge_faithfulness": float(j.get("faithfulness"))}
    except (TypeError, ValueError):
        return {"judge_correctness": None, "judge_faithfulness": None}


def eval_generation(items: list[dict], retriever: Retriever, llm_specs: list[str], use_judge: bool = False) -> pd.DataFrame:
    judge = get_judge() if use_judge else None
    rows = []
    for spec in llm_specs:
        rag = RAG(retriever, get_llm(spec))
        for it in tqdm(items, desc=f"answers | {spec}"):
            try:
                ans = rag.ask(it["question"])
            except Exception as e:  # keep going: one provider failing should not lose the whole run
                rows.append({"llm": spec, "type": it["type"], "id": it["id"], "error": str(e)[:200]})
                continue
            text = ans.text.lower()
            row = {"llm": spec, "type": it["type"], "id": it["id"], "question": it["question"], "answer": ans.text,
                   "retrieval_s": ans.retrieval_s, "generation_s": ans.generation_s,
                   "cites_sources": float(bool(re.search(r"\[S\d+\]", ans.text)))}
            if it.get("facts"):
                row["fact_recall"] = sum(f.lower() in text for f in it["facts"]) / len(it["facts"])
            if "number" in it:
                row["number_correct"] = float(bool(re.search(rf"(?<![\d,.]){it['number']:,}(?![\d,]|\.\d)", ans.text) or
                                                   re.search(rf"(?<![\d,.]){it['number']}(?![\d,]|\.\d)", ans.text)))
            abstained = bool(ABSTAIN_RE.search(ans.text))
            row["abstain_correct"] = float(abstained == bool(it.get("abstain")))
            if judge and not it.get("abstain"):
                row |= _judge(judge, it, ans)
            rows.append(row)
    return pd.DataFrame(rows)


def report(ret_df: pd.DataFrame | None, gen_df: pd.DataFrame | None, out_dir, k: int) -> str:
    lines = [f"# MedRAG evaluation", "", f"Run: {datetime.now():%Y-%m-%d %H:%M}", ""]
    if ret_df is not None and len(ret_df):
        agg = ret_df.groupby(["index", "mode"])[["hit", "mrr", "recall", "ndcg", "latency_s"]].mean().round(3)
        agg = agg.sort_values("mrr", ascending=False).reset_index()
        lines += [f"## Retrieval (k={k}, {ret_df['id'].nunique()} questions with known gold documents)", "",
                  agg.rename(columns={"hit": f"Hit@{k}", "mrr": "MRR", "recall": f"Recall@{k}", "ndcg": f"nDCG@{k}", "latency_s": "Latency (s)"}).to_markdown(index=False), ""]
        by_type = ret_df.pivot_table(index=["index", "mode"], columns="type", values="mrr", aggfunc="mean").round(3).reset_index()
        lines += ["### MRR by question type", "", by_type.to_markdown(index=False), ""]
    if gen_df is not None and len(gen_df):
        cols = [c for c in ["fact_recall", "number_correct", "abstain_correct", "cites_sources", "judge_correctness",
                            "judge_faithfulness", "retrieval_s", "generation_s"] if c in gen_df]
        agg = gen_df.groupby("llm")[cols].mean().round(3)
        lines += ["## Answers", "", agg.to_markdown(), "",
                  "- fact_recall: share of key facts from the gold report present in the answer",
                  "- number_correct: count questions answered with the exact number",
                  "- abstain_correct: says 'not found' exactly when it should",
                  "- judge_*: 1-5 scores from the judge model (a different model from the one answering)", ""]
        errs = gen_df["error"].notna().sum() if "error" in gen_df else 0
        if errs:
            lines.append(f"**{errs} answers failed with errors**; see generation.csv.")
    md = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    if ret_df is not None:
        ret_df.to_csv(out_dir / "retrieval.csv", index=False)
    if gen_df is not None:
        gen_df.to_csv(out_dir / "generation.csv", index=False)
    return md
