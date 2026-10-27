"""Hybrid index on disk: chunk metadata (parquet) + dense vectors (npy + FAISS) + BM25 (bm25s).

One index directory per embedding model, so different models can be built and compared side by side.
Re-building is incremental: vectors for chunks whose text has not changed are reused, only new or
changed chunks are embedded.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import bm25s
import faiss
import numpy as np
import pandas as pd
import Stemmer
from tqdm import tqdm

from medrag.providers.embeddings import Embedder

STEMMER = Stemmer.Stemmer("english")
HNSW_THRESHOLD = 150_000  # above this many chunks use an approximate HNSW graph instead of exact search


def tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=False)


def build(chunks: pd.DataFrame, embedder: Embedder, path: Path, batch: int = 256) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    started = time.time()

    old: dict[str, np.ndarray] = {}
    if (path / "chunks.parquet").exists() and (path / "vectors.npy").exists():
        old_ids = pd.read_parquet(path / "chunks.parquet", columns=["chunk_id"])["chunk_id"]
        old_vecs = np.load(path / "vectors.npy")
        old = dict(zip(old_ids, old_vecs))

    chunks = chunks.reset_index(drop=True)
    todo = [i for i, cid in enumerate(chunks["chunk_id"]) if cid not in old]
    dim = len(next(iter(old.values()))) if old else len(embedder.embed_documents(["dimension probe"])[0])
    vecs = np.zeros((len(chunks), dim), np.float32)
    for i, cid in enumerate(chunks["chunk_id"]):
        if cid in old:
            vecs[i] = old[cid]

    texts = chunks["text"].tolist()
    for start in tqdm(range(0, len(todo), batch), desc=f"embedding {len(todo):,} new chunks", unit="batch"):
        idx = todo[start:start + batch]
        vecs[idx] = embedder.embed_documents([texts[i] for i in idx])

    np.save(path / "vectors.npy", vecs)
    chunks.to_parquet(path / "chunks.parquet", index=False)

    if len(chunks) > HNSW_THRESHOLD:
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(path / "dense.faiss"))

    bm25 = bm25s.BM25(k1=1.2, b=0.75)
    bm25.index(tokenize(texts), show_progress=False)
    bm25.save(str(path / "bm25"))

    manifest = {
        "provider": embedder.name, "model": embedder.model, "dim": dim, "chunks": len(chunks),
        "embedded_now": len(todo), "reused": len(chunks) - len(todo), "dense_index": type(index).__name__,
        "built_at": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - started, 1),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class Index:
    def __init__(self, path: Path):
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"No index at {path}. Run: python -m medrag index")
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.meta = pd.read_parquet(path / "chunks.parquet")
        self.vectors = np.load(path / "vectors.npy", mmap_mode="r")
        self.dense = faiss.read_index(str(path / "dense.faiss"))
        if hasattr(self.dense, "hnsw"):
            self.dense.hnsw.efSearch = 128
        self.bm25 = bm25s.BM25.load(str(path / "bm25"))
        self._vocab = self.bm25.vocab_dict

    def dense_search(self, qvec: np.ndarray, k: int, rows: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-k (row, score). With `rows` (a filter), search exactly inside that subset."""
        if rows is not None:
            if len(rows) == 0:
                return []
            scores = np.asarray(self.vectors[rows]) @ qvec
            top = np.argsort(-scores)[:k]
            return [(int(rows[i]), float(scores[i])) for i in top]
        scores, ids = self.dense.search(qvec[None, :].astype(np.float32), k)
        return [(int(i), float(sc)) for i, sc in zip(ids[0], scores[0]) if i >= 0]

    def sparse_search(self, query: str, k: int, rows: np.ndarray | None = None) -> list[tuple[int, float]]:
        toks = tokenize([query])
        inv = {v: key for key, v in toks.vocab.items()}
        words = [inv[t] for t in toks.ids[0] if inv[t] in self._vocab]
        if not words:
            return []
        scores = self.bm25.get_scores(words)
        if rows is not None:
            sub = scores[rows]
            top = np.argsort(-sub)[:k]
            return [(int(rows[i]), float(sub[i])) for i in top if sub[i] > 0]
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
