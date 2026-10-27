"""Retrieval pipeline:

  1. parse the question into filters (patient, modality, body part, priority, dates)
  2. restrict the search space to rows matching the filters (relaxing them if nothing matches)
  3. dense (embedding) search  +  sparse (BM25) search  over that space
  4. fuse the two rankings with Reciprocal Rank Fusion
  5. re-score the fused candidates with a cross-encoder reranker
  6. pin the patient's profile chunk and, for "latest" questions, their most recent studies
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from medrag.config import get_settings
from medrag.index.store import Index
from medrag.providers.embeddings import get_embedder
from medrag.retrieval.expansion import expand as expand_query, impression_pattern
from medrag.retrieval.query_parser import Filters, QueryParser

MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


@dataclass
class Hit:
    row: int
    score: float
    sources: list[str] = field(default_factory=list)


@dataclass
class Result:
    hits: pd.DataFrame          # retrieved chunks with metadata + score
    filters: Filters
    subset: np.ndarray | None   # rows matching the filters (None = whole index)
    notes: list[str]


def rrf(rankings: dict[str, list[tuple[int, float]]], k: int = 60) -> list[Hit]:
    fused: dict[int, Hit] = {}
    for name, ranking in rankings.items():
        for rank, (row, _) in enumerate(ranking):
            h = fused.setdefault(row, Hit(row, 0.0))
            h.score += 1.0 / (k + rank + 1)
            h.sources.append(name)
    return sorted(fused.values(), key=lambda h: -h.score)


class Reranker:
    _models: dict = {}

    def __init__(self, model: str):
        if model not in self._models:
            from sentence_transformers import CrossEncoder

            self._models[model] = CrossEncoder(model, max_length=512)
        self.ce = self._models[model]

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.asarray(self.ce.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False))


class Retriever:
    def __init__(self, index: Index | None = None, embed_provider: str | None = None, embed_model: str | None = None,
                 rerank_model: str | None = None, rerank_blend: str | None = None):
        s = get_settings()
        self.index = index or Index(s.index_path(embed_provider, embed_model))
        self.meta = self.index.meta
        self.embedder = get_embedder(self.index.manifest["provider"], self.index.manifest["model"])
        self.parser = QueryParser(self.meta)
        self.rerank_model = rerank_model if rerank_model is not None else s.rerank_model
        self._reranker = None
        self.rerank_blend = rerank_blend or s.rerank_blend
        is_study = self.meta["doc_type"] == "study"
        imp = self.meta["impression"].fillna("").str.lower()
        self._abnormal = is_study & ~imp.str.contains("no acute abnormality") & (imp != "")

    @property
    def reranker(self) -> Reranker | None:
        if self._reranker is None and self.rerank_model and self.rerank_model != "none":
            self._reranker = Reranker(self.rerank_model)
        return self._reranker

    # ---------------- filtering ----------------
    def _mask(self, f: Filters, use: set[str]) -> pd.Series:
        m = pd.Series(True, index=self.meta.index)
        if "patient" in use and f.patient_ids:
            m &= self.meta["patient_id"].isin(f.patient_ids)
        if "modality" in use and f.modality:
            m &= self.meta["modality"] == f.modality
        if "body_part" in use and f.body_part:
            m &= self.meta["body_part"] == f.body_part
        if "priority" in use and f.priority:
            m &= self.meta["priority"] == f.priority
        if "dates" in use and f.date_from is not None:
            m &= self.meta["study_date"] >= f.date_from
        if "dates" in use and f.date_to is not None:
            m &= self.meta["study_date"] <= f.date_to
        if "abnormal" in use and f.abnormal_only:
            m &= self._abnormal
        if "documents" in use and f.documents:
            m &= self.meta["doc_type"].isin(["document", "patient"])
        if "documents" in use and f.specialty:
            m &= (self.meta["specialty"] == f.specialty) | (self.meta["doc_type"] == "patient")
        return m

    def filter_rows(self, f: Filters, notes: list[str]) -> np.ndarray | None:
        """Rows matching the filters; drops the least important filter until something matches."""
        order = ["abnormal", "documents", "priority", "dates", "body_part", "modality", "patient"]
        use = set(order)
        while use:
            m = self._mask(f, use)
            if m.any():
                if m.all():
                    return None
                return np.flatnonzero(m.to_numpy())
            dropped = order[len(order) - len(use)]
            use.discard(dropped)
            notes.append(f"No records match every filter; relaxed the '{dropped}' filter.")
        return None

    # ---------------- search ----------------
    def finding_mask(self, labels: list[str]) -> pd.Series:
        """Studies whose report impression shows any of the given findings (radiologist wording)."""
        imp = self.meta["impression"].fillna("")
        pattern = "|".join(f"(?:{impression_pattern(lbl)})" for lbl in labels)
        return imp.str.contains(pattern, case=False, regex=True)

    def search(self, query: str, k: int | None = None, mode: str = "hybrid+rerank", candidates: int | None = None,
               use_filters: bool = True, expand: bool = True) -> Result:
        s = get_settings()
        k = k or s.top_k
        n_cand = candidates or s.candidates
        notes: list[str] = []
        f = self.parser.parse(query)
        rows = self.filter_rows(f, notes) if use_filters else None
        q = query
        if expand:
            q, exp_notes = expand_query(query)
            notes += exp_notes

        rankings = {}
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            rankings["dense"] = self.index.dense_search(self.embedder.embed_query(q), n_cand, rows)
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            rankings["bm25"] = self.index.sparse_search(q, n_cand, rows)
        hits = rrf(rankings) if len(rankings) > 1 else [Hit(r, sc, list(rankings)) for r, sc in next(iter(rankings.values()))]

        if mode == "hybrid+rerank" and self.reranker and hits:
            pool = hits[: max(s.rerank_pool, k)]
            scores = self.reranker.score(q, self.meta["text"].iloc[[h.row for h in pool]].tolist())
            if self.rerank_blend == "fuse":
                # Merge the reranker's order with the hybrid order, so a reranker trained on another domain
                # cannot throw away what the first-stage search got right.
                by_rerank = [(pool[i].row, float(scores[i])) for i in np.argsort(-scores)]
                fused = rrf({"hybrid": [(h.row, h.score) for h in pool], "rerank": by_rerank})
                src = {h.row: h.sources for h in pool}
                hits = [Hit(h.row, h.score, src[h.row]) for h in fused]
            else:
                for h, sc in zip(pool, scores):
                    h.score = float(sc)
                hits = sorted(pool, key=lambda h: -h.score)

        hits = hits[:k]

        # Pin what a clinician would always want in front of them.
        pinned: list[Hit] = []
        if f.patient_ids and use_filters:
            prof = self.meta.index[(self.meta["doc_type"] == "patient") & self.meta["patient_id"].isin(f.patient_ids) & (self.meta["part"] == 0)]
            pinned += [Hit(int(r), float("inf"), ["profile"]) for r in prof]
            if f.latest and rows is not None:
                studies = self.meta.iloc[rows]
                studies = studies[studies["doc_type"] == "study"].sort_values("study_date", ascending=False)
                pinned += [Hit(int(r), float("inf"), ["latest"]) for r in studies.index[:3]]
        seen = {h.row for h in pinned}
        hits = pinned + [h for h in hits if h.row not in seen][: max(k - len(pinned), k // 2)]

        df = self.meta.iloc[[h.row for h in hits]].copy()
        df["score"] = [h.score for h in hits]
        df["via"] = ["+".join(dict.fromkeys(h.sources)) for h in hits]
        return Result(hits=df, filters=f, subset=rows, notes=notes)
