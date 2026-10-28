"""Retrieval pipeline:

  1. parse the question into filters (patient, modality, body part, priority, dates)
  2. restrict the search space to rows matching the filters (relaxing them if nothing matches)
  3. dense (embedding) search  +  sparse (BM25) search  over that space
  4. fuse the two rankings with Reciprocal Rank Fusion
  5. re-score the fused candidates with a cross-encoder reranker
  6. pin the patient's profile chunk and, for "latest" questions, their most recent studies
"""
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from medrag.config import get_settings
from medrag.index.store import Index
from medrag.providers.embeddings import get_embedder
from medrag.retrieval.expansion import expand as expand_query, impression_pattern
from medrag.retrieval.query_parser import Filters, QueryParser

MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")
# First is relaxed first. Documents have no body part and a lay word often sets one ("back" -> spine), so
# body_part goes before "documents": "orthopedic document about back pain" must not fall back to spine studies.
# Dates and modality stay after it: "X-ray report for MRN... dated ..." means the study, not a document.
FILTER_ORDER = ["abnormal", "body_part", "documents", "priority", "dates", "modality", "patient"]


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
    scope: pd.Series | None = None  # rows the user may see in computed facts (None = all); see security/access.py
    timings: dict[str, float] = field(default_factory=dict)  # seconds per retrieval stage (tracing)
    denied_patients: list[int] = field(default_factory=list)  # patients named in the question the user may not see


def rrf(rankings: dict[str, list[tuple[int, float]]], k: int = 60, weights: dict[str, float] | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion; a ranking's weight multiplies its 1/(k + rank) contribution (default 1).
    A name like 'dense#2' (one ranking per rewritten query) takes the weight of 'dense'."""
    fused: dict[int, Hit] = {}
    for name, ranking in rankings.items():
        w = (weights or {}).get(name.split("#")[0], 1.0)
        for rank, (row, _) in enumerate(ranking):
            h = fused.setdefault(row, Hit(row, 0.0))
            h.score += w / (k + rank + 1)
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
                 rerank_model: str | None = None, rerank_blend: str | None = None, rrf_k: int | None = None,
                 dense_weight: float | None = None, rerank_pool: int | None = None, candidates: int | None = None,
                 query_rewrite: str | None = None):
        s = get_settings()
        self.rrf_k = rrf_k or s.rrf_k
        self.dense_weight = s.dense_weight if dense_weight is None else dense_weight
        self.rerank_pool = rerank_pool or s.rerank_pool
        self.candidates = candidates or s.candidates
        self.query_rewrite = query_rewrite or s.query_rewrite
        self._rewriter = None
        self._cache_size = s.retrieval_cache_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
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

    def filter_rows(self, f: Filters, notes: list[str], allowed: pd.Series | None = None) -> np.ndarray | None:
        """Rows matching the filters; drops the least important filter until something matches.
        `allowed` (the user's access) is applied on top and is never relaxed."""
        order = FILTER_ORDER
        use = set(order)
        while use:
            m = self._mask(f, use)
            if allowed is not None:
                m &= allowed
            if m.any():
                if m.all():
                    return None
                return np.flatnonzero(m.to_numpy())
            dropped = order[len(order) - len(use)]
            use.discard(dropped)
            if not self._mask(f, {dropped}).all():  # only report filters the question actually set
                notes.append(f"No records match every filter; relaxed the '{dropped}' filter.")
        return None if allowed is None else np.flatnonzero(allowed.to_numpy())  # possibly empty: nothing the user may see

    # ---------------- search ----------------
    def finding_mask(self, labels: list[str]) -> pd.Series:
        """Studies whose report impression shows any of the given findings (radiologist wording)."""
        imp = self.meta["impression"].fillna("")
        pattern = "|".join(f"(?:{impression_pattern(lbl)})" for lbl in labels)
        return imp.str.contains(pattern, case=False, regex=True)

    def search(self, query: str, k: int | None = None, mode: str = "hybrid+rerank", candidates: int | None = None,
               use_filters: bool = True, expand: bool = True, user=None) -> Result:
        """`user` (security.access.User) restricts the search to the records that user may see; None = no restriction."""
        key = (query, k, mode, candidates, use_filters, expand, user)
        with self._lock:
            r = self._cache.get(key)
            if r is not None:  # identical question from the same access scope: reuse, as a copy
                self._cache.move_to_end(key)
                return replace(r, hits=r.hits.copy(), notes=list(r.notes), timings={"total": 0.0, "cache_hit": 1.0})
        r = self._search(query, k, mode, candidates, use_filters, expand, user)
        if self._cache_size:
            with self._lock:
                self._cache[key] = replace(r, hits=r.hits.copy(), notes=list(r.notes))
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return r

    def _search(self, query, k, mode, candidates, use_filters, expand, user) -> Result:
        s = get_settings()
        k = k or s.top_k
        n_cand = candidates or self.candidates
        notes: list[str] = []
        t = {"start": time.perf_counter()}
        f = self.parser.parse(query)
        allowed = user.allowed_mask(self.meta) if user is not None else None
        denied: list[int] = []
        if allowed is not None and f.patient_ids:
            # Patient-level rule: a named patient the user may not see is treated exactly like an unknown
            # one, so the answer does not reveal that the patient exists.
            ok = [p for p in f.patient_ids if p in user.patients]
            denied = [p for p in f.patient_ids if p not in user.patients]
            if not ok:
                f.unresolved_patient = ", ".join(f.patient_names) or "the requested patient"
            f.patient_ids = ok
            f.patient_names = [self.parser.name_of[p] for p in ok]
        if use_filters:
            rows = self.filter_rows(f, notes, allowed)
        else:
            rows = None if allowed is None else np.flatnonzero(allowed.to_numpy())
        t["filter"] = time.perf_counter()
        q = query
        if expand:
            q, exp_notes = expand_query(query)
            notes += exp_notes

        # Optional LLM query transformation: extra search queries (rewrite / multi) or a hypothetical
        # report to embed instead of the question (HyDE). Filters always come from the original question.
        extra_queries, hyde_doc = [], None
        if self.query_rewrite != "none":
            from medrag.retrieval.rewrite import transform

            if self._rewriter is None:
                from medrag.providers.llms import get_llm

                self._rewriter = get_llm(s.rewrite_llm or None)
            extra_queries, hyde_doc = transform(self._rewriter, self.query_rewrite, query)
        t["rewrite"] = time.perf_counter()

        rankings = {}
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            qvec = self.embedder.embed_documents([hyde_doc])[0] if hyde_doc else self.embedder.embed_query(q)
            rankings["dense"] = self.index.dense_search(qvec, n_cand, rows)
            for i, eq in enumerate(extra_queries, 2):
                rankings[f"dense#{i}"] = self.index.dense_search(self.embedder.embed_query(eq), n_cand, rows)
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            rankings["bm25"] = self.index.sparse_search(q, n_cand, rows)
            for i, eq in enumerate(extra_queries, 2):
                rankings[f"bm25#{i}"] = self.index.sparse_search(eq, n_cand, rows)
        t["search"] = time.perf_counter()
        if len(rankings) > 1:
            hits = rrf(rankings, self.rrf_k, {"dense": self.dense_weight})
        else:
            hits = [Hit(r, sc, list(rankings)) for r, sc in next(iter(rankings.values()))]

        if mode == "hybrid+rerank" and self.reranker and hits:
            pool = hits[: max(self.rerank_pool, k)]
            scores = self.reranker.score(q, self.meta["text"].iloc[[h.row for h in pool]].tolist())
            if self.rerank_blend == "fuse":
                # Merge the reranker's order with the hybrid order, so a reranker trained on another domain
                # cannot throw away what the first-stage search got right.
                by_rerank = [(pool[i].row, float(scores[i])) for i in np.argsort(-scores)]
                fused = rrf({"hybrid": [(h.row, h.score) for h in pool], "rerank": by_rerank}, self.rrf_k)
                src = {h.row: h.sources for h in pool}
                hits = [Hit(h.row, h.score, src[h.row]) for h in fused]
            else:
                for h, sc in zip(pool, scores):
                    h.score = float(sc)
                hits = sorted(pool, key=lambda h: -h.score)

        hits = hits[:k]
        t["rerank"] = time.perf_counter()

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
        t["end"] = time.perf_counter()
        stages = list(t)
        timings = {b: round(t[b] - t[a], 4) for a, b in zip(stages, stages[1:])} | {"total": round(t["end"] - t["start"], 4)}
        timings["pin"] = timings.pop("end")
        scope = None if user is None or user.role in ("admin", "researcher") else allowed  # researchers: aggregates over everyone
        return Result(hits=df, filters=f, subset=rows, notes=notes, scope=scope, timings=timings, denied_patients=denied)
