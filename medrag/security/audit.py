"""Audit trail and request metrics, both append-only JSONL.

audit.jsonl: who asked (user, role), when, which patients and records were read, which were refused,
which LLM received them (and whether identifiers were pseudonymised), and injection flags. The question
text is stored with identifiers redacted; record ids are kept, because they are what an audit needs.

metrics.jsonl: latency per stage, estimated tokens, estimated cost and errors, per request.
`python -m medrag metrics` summarises it (p50 / p95 / p99, error rate, tokens, cost).
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from medrag.config import get_settings

_lock = threading.Lock()


def _path(kind: str) -> Path:
    s = get_settings()
    p = getattr(s, f"{kind}_log") or s.reports_dir / f"{kind}.jsonl"
    return Path(p)


def append(kind: str, record: dict) -> None:
    p = _path(kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")} | record, default=str)
    with _lock, p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read(kind: str) -> pd.DataFrame:
    p = _path(kind)
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()])


def estimate_tokens(text: str) -> int:
    """About 4 characters per token for English; used when a provider does not report usage."""
    return max(1, len(text or "") // 4)


def metrics_summary(df: pd.DataFrame | None = None) -> str:
    df = read("metrics") if df is None else df
    if df.empty:
        return "No requests recorded yet."
    lines = [f"Requests: {len(df):,}  (from {df['ts'].min()} to {df['ts'].max()})",
             f"Errors: {int(df['error'].notna().sum()) if 'error' in df else 0} ({100 * df['error'].notna().mean() if 'error' in df else 0:.1f}%)", ""]
    rows = []
    for llm, g in df.groupby("llm"):
        row = {"llm": llm, "requests": len(g)}
        for col in ("total_s", "retrieval_s", "generation_s"):
            v = g[col].dropna() if col in g else pd.Series(dtype=float)
            if len(v):
                row |= {f"{col} p50": round(v.quantile(.5), 2), f"{col} p95": round(v.quantile(.95), 2), f"{col} p99": round(v.quantile(.99), 2)}
        row |= {"tokens in": int(g["tokens_in"].sum()), "tokens out": int(g["tokens_out"].sum()), "cost USD": round(g["cost_usd"].sum(), 4),
                "cache hits": int(g.get("cache_hit", pd.Series(dtype=float)).fillna(0).sum())}
        rows.append(row)
    return "\n".join(lines) + pd.DataFrame(rows).to_markdown(index=False)
