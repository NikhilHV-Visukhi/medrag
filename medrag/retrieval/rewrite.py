"""LLM query transformations for retrieval, cached on disk so repeated evaluations cost nothing.

- rewrite: one search query in radiology-report wording, searched next to the original question.
- multi:   three different phrasings, each searched; all rankings are fused (multi-query retrieval).
- hyde:    a short hypothetical report passage that would answer the question; its embedding replaces
           the question's in dense search (Hypothetical Document Embeddings). BM25 keeps the question.
"""
import hashlib
import json
import re
import threading

from medrag.config import get_settings
from medrag.providers.llms import LLM

RULES = ("Plain English keywords, not SQL or code. Do not add names, MRNs, dates or details that are not in the question; "
         "copy any that are.")
PROMPTS = {
    "rewrite": "Rewrite this question about hospital imaging records as one short search phrase using the words a "
               f"radiology report or clinical note would use. {RULES} Reply with the phrase only.\n\nQuestion: {{q}}",
    "multi": "Write three different short search phrases for this question about hospital imaging records, using the "
             f"words radiology reports or clinical notes would use. {RULES} One phrase per line, nothing else.\n\nQuestion: {{q}}",
    "hyde": "Write a 2-4 sentence excerpt of a radiology report or clinical note that would answer this question. "
            "Use typical report wording (findings, impression). Do not mention that it is hypothetical.\n\nQuestion: {q}",
}
_lock = threading.Lock()
_cache: dict | None = None


def _cache_path():
    return get_settings().reports_dir / "rewrite_cache.json"


def _load() -> dict:
    global _cache
    if _cache is None:
        p = _cache_path()
        _cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _cache


def transform(llm: LLM, kind: str, question: str) -> tuple[list[str], str | None]:
    """Returns (extra search queries, hypothetical document or None)."""
    if kind not in PROMPTS:
        raise ValueError(f"Unknown query_rewrite {kind!r}; choose from none, {', '.join(PROMPTS)}")
    key = hashlib.sha1(f"{kind}|{llm.label}|{question}".encode()).hexdigest()
    with _lock:
        cached = _load().get(key)
    if cached is None:
        cached = llm.complete("You help search medical records. Follow the instructions exactly.", PROMPTS[kind].format(q=question), 200).strip()
        with _lock:
            _load()[key] = cached
            _cache_path().parent.mkdir(parents=True, exist_ok=True)
            _cache_path().write_text(json.dumps(_cache, indent=0), encoding="utf-8")
    if kind == "hyde":
        return [], cached
    lines = [re.sub(r"^\s*(?:[-*\d.)]+\s*)", "", ln).strip().strip('"') for ln in cached.splitlines()]
    return [ln for ln in lines if ln][: 1 if kind == "rewrite" else 3], None
