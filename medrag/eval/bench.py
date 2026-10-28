"""Benchmarks on a fixed subset of the index: embedding models (A) and chunking strategies (B).

Embedding all 121k chunks with a base-size model takes 8-10 hours on a laptop CPU, so every candidate is
built on the same subset of records: every record of every patient a test question is about (which
includes all gold documents), plus random distractor documents, studies and profiles. Every candidate
gets the same questions, the same records and the same retrieval configuration; only the embedder (or
the chunking) changes. Records are chosen by doc_id, so a different chunking of the same records is
compared fairly. Absolute scores are higher than on the full index (fewer distractors); compare
candidates with each other, not with the full-index numbers.
"""
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from medrag.config import get_settings
from medrag.eval import run, testset
from medrag.index.store import Index, build
from medrag.providers.embeddings import get_embedder
from medrag.retrieval.retriever import Retriever


def rss_mb() -> float:
    """Resident memory of this process in MB (Windows working set; ru_maxrss elsewhere)."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                               "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def bench_dir() -> Path:
    return get_settings().index_dir / "bench"


def load_items(meta: pd.DataFrame, per_type: int = 20) -> list[dict]:
    """One test set for every benchmark, built from the full index and saved, so all candidates get the same questions."""
    path = bench_dir() / "testset.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    items = testset.build(meta, per_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    testset.save(items, path)
    return items


def subset_doc_ids(meta: pd.DataFrame, items: list[dict], n_documents: int = 4000, n_other: int = 5000, seed: int = 7) -> list[str]:
    path = bench_dir() / "subset_doc_ids.json"
    if path.exists():
        return json.loads(path.read_text())
    gold = {g for it in items for g in it["gold_docs"]}
    pids = set(meta.loc[meta["doc_id"].isin(gold), "patient_id"])
    keep = set(meta.loc[meta["patient_id"].isin(pids), "doc_id"]) | gold
    rng = np.random.default_rng(seed)
    rest = meta.loc[~meta["doc_id"].isin(keep)].drop_duplicates("doc_id")
    docs = np.sort(rest.loc[rest["doc_type"] == "document", "doc_id"].to_numpy())
    other = np.sort(rest.loc[rest["doc_type"] != "document", "doc_id"].to_numpy())
    keep |= set(rng.choice(docs, min(n_documents, len(docs)), replace=False))
    keep |= set(rng.choice(other, min(n_other, len(other)), replace=False))
    out = sorted(keep)
    path.write_text(json.dumps(out))
    return out


def _slug(*parts: str) -> str:
    return "__".join(parts).replace("/", "_").replace(":", "_")


def bench_index(chunks: pd.DataFrame, provider: str, model: str, label: str, reuse_from: Index | None = None) -> tuple[Index, dict]:
    """Build (or reuse) one benchmark index; returns it with build time and memory figures."""
    path = bench_dir() / _slug(label, provider, model)
    before = rss_mb()
    t0 = time.time()
    emb = get_embedder(provider, model)
    emb.embed_documents(["memory probe"])  # weights are memory-mapped: touch them so they count
    load_s, model_mb = time.time() - t0, rss_mb() - before
    reuse = None
    if reuse_from is not None and (reuse_from.manifest["provider"], reuse_from.manifest["model"]) == (provider, model):
        reuse = dict(zip(reuse_from.meta["chunk_id"], np.asarray(reuse_from.vectors)))
    if (path / "manifest.json").exists():
        m = json.loads((path / "manifest.json").read_text())
        stats = json.loads((path / "bench_stats.json").read_text()) if (path / "bench_stats.json").exists() else {}
    else:
        m = build(chunks, emb, path, reuse=reuse)
        n = m["embedded_now"]
        stats = {"embed_s_per_1k": round(m["seconds"] / n * 1000, 1) if n else None, "embedded_now": n,
                 "reused_vectors": m["reused"], "model_load_s": round(load_s, 1), "model_ram_mb": round(model_mb)}
        (path / "bench_stats.json").write_text(json.dumps(stats))
    stats["vectors_mb"] = round(m["chunks"] * m["dim"] * 4 / 2**20, 1)
    return Index(path), m | stats


def run_embedders(specs: list[tuple[str, str]], out: Path, k: int = 8) -> pd.DataFrame:
    import sentence_transformers, transformers  # noqa: F401  (import torch first, so model RAM excludes the libraries)

    s = get_settings()
    main = Index(s.index_path())
    items = load_items(main.meta)
    keep = subset_doc_ids(main.meta, items)
    chunks = main.meta[main.meta["doc_id"].isin(keep)].reset_index(drop=True)
    print(f"benchmark subset: {chunks['doc_id'].nunique():,} records, {len(chunks):,} chunks ({chunks['doc_type'].value_counts().to_dict()})")
    rows, info = [], []
    for provider, model in specs:
        idx, m = bench_index(chunks, provider, model, "embed", reuse_from=main)
        ret = Retriever(idx)
        name = f"{provider}:{model}"
        cfgs = [run.Config(f"{name} | dense", ret, mode="dense"), run.Config(f"{name} | hybrid+rerank", ret)]
        df = run.eval_retrieval(items, cfgs, k)
        ret.embedder.__dict__.pop("_qcache", None)  # time the model, not the query cache
        t0 = time.time()
        for it in items[:50]:
            ret.embedder.embed_query(it["question"])
        info.append({"model": name, "dim": m["dim"], "chunks": m["chunks"], "build_s": m["seconds"], "embed_s_per_1k_chunks": m.get("embed_s_per_1k"),
                     "model_ram_mb": m.get("model_ram_mb"), "vectors_mb": m["vectors_mb"], "query_embed_ms": round((time.time() - t0) / 50 * 1000, 1)})
        rows.append(df)
        pd.concat(rows).to_csv(out / "retrieval.csv", index=False)
        pd.DataFrame(info).to_csv(out / "models.csv", index=False)
        print(f"done {name}: dense MRR {df[df.config.str.endswith('dense')]['mrr'].mean():.3f}", flush=True)
    return pd.concat(rows)


CHUNKINGS = {  # name -> build_chunks arguments
    "section-220": dict(max_words=220, strategy="section"),
    "section-120": dict(max_words=120, strategy="section"),
    "section-350": dict(max_words=350, strategy="section"),
    "recursive-220-o40": dict(max_words=220, strategy="recursive", overlap_words=40),
    "recursive-220-o0": dict(max_words=220, strategy="recursive", overlap_words=0),
    "recursive-120-o20": dict(max_words=120, strategy="recursive", overlap_words=20),
}


def run_chunkings(names: list[str], out: Path, k: int = 8) -> pd.DataFrame:
    from medrag.data import store
    from medrag.ingest.documents import build_chunks

    s = get_settings()
    main = Index(s.index_path())
    items = load_items(main.meta)
    keep = set(subset_doc_ids(main.meta, items))
    tables = store.read_all("clean")
    rows, info = [], []
    for name in names:
        chunks = build_chunks(tables, **CHUNKINGS[name])
        chunks = chunks[chunks["doc_id"].isin(keep)].reset_index(drop=True)
        idx, m = bench_index(chunks, s.embed_provider, s.embed_model, f"chunk-{name}", reuse_from=main)
        ret = Retriever(idx)
        df = run.eval_retrieval(items, [run.Config(f"{name} | dense", ret, mode="dense"), run.Config(f"{name} | hybrid+rerank", ret)], k)
        words = chunks["text"].str.split().str.len()
        info.append({"chunking": name, "chunks": len(chunks), "mean_words": round(words.mean()), "max_words": int(words.max()),
                     "doc_chunks_per_document": round((chunks["doc_type"] == "document").sum() / chunks.loc[chunks["doc_type"] == "document", "doc_id"].nunique(), 2),
                     "build_s": m["seconds"]})
        rows.append(df)
        pd.concat(rows).to_csv(out / "retrieval.csv", index=False)
        pd.DataFrame(info).to_csv(out / "chunkings.csv", index=False)
        print(f"done {name}: hybrid+rerank MRR {df[df.config.str.endswith('rerank')]['mrr'].mean():.3f}", flush=True)
    return pd.concat(rows)


# Retrieval ablations on the full index (C). One change per variant against the default configuration.
SEMANTIC_TYPES = ["semantic", "hard_semantic_cohort", "hard_paraphrase_patient", "doc_semantic", "doc_transcription_semantic"]
TUNING = {  # name -> (Retriever keyword arguments, question types or None for all)
    "default": ({}, None),
    "rrf_k=20": ({"rrf_k": 20}, None),
    "rrf_k=100": ({"rrf_k": 100}, None),
    "dense_weight=0.5": ({"dense_weight": 0.5}, None),
    "dense_weight=2": ({"dense_weight": 2.0}, None),
    "candidates=40": ({"candidates": 40}, None),
    "candidates=100": ({"candidates": 100}, None),
    "rerank_pool=20": ({"rerank_pool": 20}, None),
    "rerank_pool=50": ({"rerank_pool": 50}, None),
    "reranker=bge-reranker-v2-m3": ({"rerank_model": "BAAI/bge-reranker-v2-m3"}, None),
    "reranker=MiniLM": ({"rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}, None),
    "no reranker (hybrid)": ({"rerank_model": "none"}, None),
    "default (semantic types)": ({}, SEMANTIC_TYPES),
    "query rewrite": ({"query_rewrite": "rewrite"}, SEMANTIC_TYPES),
    "multi-query": ({"query_rewrite": "multi"}, SEMANTIC_TYPES),
    "HyDE": ({"query_rewrite": "hyde"}, SEMANTIC_TYPES),
}


def run_tuning(names: list[str], out: Path, k: int = 8, per_type: int = 20) -> pd.DataFrame:
    s = get_settings()
    main = Index(s.index_path())
    items = testset.build(main.meta, per_type)
    testset.save(items, out / "testset.json")
    rows = []
    for name in names:
        kw, types = TUNING[name]
        ret = Retriever(main, **kw)
        its = [it for it in items if types is None or it["type"] in types]
        df = run.eval_retrieval(its, [run.Config(name, ret)], k)
        rows.append(df)
        pd.concat(rows).to_csv(out / "retrieval.csv", index=False)
        print(f"done {name}: MRR {df['mrr'].mean():.3f} over {df['id'].nunique()} questions", flush=True)
    return pd.concat(rows)


def summary(ret_df: pd.DataFrame, extra: pd.DataFrame, out: Path, title: str, k: int = 8) -> str:
    """Per-difficulty MRR / Recall / nDCG with CIs, plus the cost table."""
    lines = [f"# {title}", "", "Compare rows with each other only (see medrag/eval/bench.py).", ""]
    for diff, d in ret_df.groupby("difficulty"):
        tab = d.groupby("config", sort=False).agg(**{"MRR": ("mrr", run.ci), f"Recall@{k}": ("recall", run.ci), f"nDCG@{k}": ("ndcg", run.ci),
                                                     f"Hit@{k}": ("hit", run.ci), "Latency (s)": ("latency_s", lambda x: f"{x.mean():.2f}")})
        lines += [f"## {diff} ({d['id'].nunique()} questions)", "", tab.reset_index().to_markdown(index=False), ""]
    allq = ret_df.groupby("config", sort=False).agg(MRR=("mrr", "mean"), Recall=("recall", "mean"), nDCG=("ndcg", "mean")).round(3)
    lines += ["## All questions", "", allq.reset_index().to_markdown(index=False), ""]
    lines += ["## MRR by question type", "", ret_df.pivot_table(index="config", columns="type", values="mrr", sort=False).round(3).reset_index().to_markdown(index=False), ""]
    if len(extra):
        lines += ["## Cost", "", extra.to_markdown(index=False), ""]
    md = "\n".join(lines)
    (out / "summary.md").write_text(md, encoding="utf-8")
    return md
