"""Summarise D answers (deterministic checks) per model from the resume files; one row per (llm, question id)."""
import json
import sys
from pathlib import Path

import pandas as pd

# gen_large_local.jsonl also holds the qwen2.5 answers (copied in from gen_large_local_q25.jsonl before judging).
files = sys.argv[1:] or ["reports/gen_large.jsonl", "reports/gen_large_local.jsonl", "reports/gen_large_nvidia.jsonl"]
rows = []
for f in files:
    if Path(f).exists():
        rows += [json.loads(line) for line in Path(f).read_text(encoding="utf-8").splitlines() if line.strip()]
df = pd.DataFrame([r for r in rows if not r.get("error") and r.get("answer")])
df = df.drop_duplicates(["llm", "id"], keep="last")

out = []
for llm, g in df.groupby("llm"):
    ans, neg = g[~g.should_abstain.astype(bool)], g[g.should_abstain.astype(bool)]
    cited = g[g.cited_sentences > 0]
    row = {"model": llm, "answers": len(g)}
    for d in ["easy", "hard", "documents"]:
        s = g[g.difficulty == d]
        row[f"pass {d}"] = f"{s.auto_correct.mean():.0%} ({len(s)})" if len(s) else "-"
    row |= {
        "pass all": f"{g.auto_correct.mean():.0%}",
        "fact recall": round(ans.fact_recall.mean(), 2),
        "correct not-found": f"{neg.abstain_correct.mean():.0%} ({len(neg)})" if len(neg) else "-",
        "false not-found": f"{ans.abstained.mean():.0%}",
        "cites sources": f"{g.cites_sources.mean():.0%}",
        "citation precision": round(cited.citation_precision.mean(), 2) if len(cited) else None,
        "citation coverage": round(cited.citation_coverage.mean(), 2) if len(cited) else None,
        "invalid citations": round(g.invalid_citation_rate.mean(), 3),
        "gen s mean/median": f"{g.generation_s.mean():.1f} / {g.generation_s.median():.1f}",
        "judged": int(g.judge_correctness.notna().sum()) if "judge_correctness" in g else 0,
    }
    out.append(row)
print(pd.DataFrame(out).to_markdown(index=False))

j = df[df.judge_correctness.notna()] if "judge_correctness" in df else df.iloc[:0]
if len(j):
    print(f"\nLLM judge ({', '.join(sorted(j.judge.unique()))}; scores 1-5, 'correct' = correctness >= 4)")
    jt = j.groupby("llm").agg(
        judged=("id", "size"),
        correct=("judge_correctness", lambda s: f"{(s >= 4).mean():.0%}"),
        correctness=("judge_correctness", "mean"), faithfulness=("judge_faithfulness", "mean"),
        answer_relevance=("judge_answer_relevance", "mean"), context_relevance=("judge_context_relevance", "mean"),
        agrees_with_checks=("auto_correct", lambda s: None))
    jt["agrees_with_checks"] = j.groupby("llm").apply(lambda g: f"{((g.judge_correctness >= 4) == (g.auto_correct == 1)).mean():.0%}")
    print(jt.round(2).to_markdown())

print("\nPass rate by question type")
print((df.pivot_table(index="type", columns="llm", values="auto_correct", aggfunc="mean") * 100).round(0).to_markdown())
print("\nQuestions per type:", df.groupby("llm").type.value_counts().unstack(0).iloc[:, 0].to_dict())
