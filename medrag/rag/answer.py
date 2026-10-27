"""Answer a question: retrieve, add exact computed facts, prompt the LLM, return answer + sources."""
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from medrag.data import catalog as C
from medrag.providers.llms import LLM, get_llm
from medrag.retrieval.retriever import FILTER_ORDER, Result, Retriever

SYSTEM = """You are a clinical assistant answering questions about a hospital's radiology records.

Rules:
- COMPUTED FACTS are exact counts from the full database for the question's filters. For any "how many",
  count, breakdown or "latest" question, answer directly from them, e.g. "There were 633 such studies."
- Otherwise use only the SOURCES. Never invent patients, dates, findings or numbers.
- Cite sources inline as [S1], [S2] after each claim taken from them.
- Only if neither the COMPUTED FACTS nor the SOURCES contain the answer, say
  "The records do not contain this information."
- Be concise and clinical: short paragraphs or bullet points, dates as YYYY-MM-DD.
- This is decision support, not a diagnosis; do not give treatment advice beyond what reports recommend."""

MAX_TIMELINE = 60


@dataclass
class Answer:
    question: str
    text: str
    result: Result
    facts: str
    llm: str
    retrieval_s: float
    generation_s: float


def computed_facts(r: Result, retriever: Retriever) -> str:
    """Exact statistics for the filtered records, so counts never depend on what retrieval happened to return."""
    f, meta = r.filters, retriever.meta
    if not (f.aggregate or f.patient_ids or f.latest):
        return ""
    # Count with every filter applied, even when search relaxed one to find something to show:
    # "how many MRIs" for a patient with no MRI must be 0, not the count of their other studies.
    scope = meta[retriever._mask(f, set(FILTER_ORDER))]
    st = scope[(scope["doc_type"] == "study") & (scope["part"] == 0)]
    if f.findings:  # count by what the radiologist reported, not by the (sometimes wrong) AI label
        st = st[retriever.finding_mask(f.findings).loc[st.index]]
    lines = [f"Filters applied: {f.describe()}",
             f"Matching studies: {len(st):,} across {st['patient_id'].nunique():,} patients "
             f"(this is the exact number of studies that match the question)"]
    if len(st):
        lines.append(f"Date range: {st['study_date'].min():%Y-%m-%d} to {st['study_date'].max():%Y-%m-%d}")
        lines.append("By modality: " + ", ".join(f"{C.MODALITY_NAMES.get(k, k)} {v}" for k, v in st["modality"].value_counts().items()))
        lines.append("By priority: " + ", ".join(f"{k} {v}" for k, v in st["priority"].value_counts().items()))
        imp = st["impression"].fillna("")
        abnormal = st[~imp.str.lower().str.contains("no acute abnormality") & (imp != "")]
        lines.append(f"Studies with abnormal findings: {len(abnormal):,} ({100 * len(abnormal) / len(st):.1f}%)")
        if len(abnormal) and not f.patient_ids:
            lines.append("Most common AI-predicted findings: " + ", ".join(
                f"{k} {v}" for k, v in abnormal["ai_label"].value_counts().head(8).items()))
    if len(f.patient_ids) > 1 and len(st):
        lines.append(f"WARNING: {len(f.patient_ids)} different patients match this name. Answer separately for each patient (give MRN):")
        for pid, g in st.groupby("patient_id"):
            mods = ", ".join(f"{C.MODALITY_NAMES.get(k, k)} {v}" for k, v in g["modality"].value_counts().items())
            last = g.sort_values("study_date").iloc[-1]
            lines.append(f"- {last.patient_name} (MRN {last.mrn}, id {pid}): {len(g)} studies ({mods}); most recent {last.study_date:%Y-%m-%d} "
                         f"{C.MODALITY_NAMES.get(last.modality, last.modality)} {last.body_part}: {last.impression}")
    elif f.patient_ids and len(st):
        last = st.sort_values("study_date").iloc[-1]
        lines.append(f"Most recent study: {last.study_date:%Y-%m-%d} {C.MODALITY_NAMES.get(last.modality, last.modality)} "
                     f"{last.body_part} ({last.priority}, study {last.study_id}): {last.impression}")
    if f.patient_ids:
        docs = meta[(meta["doc_type"] == "document") & (meta["part"] == 0) & meta["patient_id"].isin(f.patient_ids)]
        if len(docs):
            lines.append(f"Uploaded documents ({len(docs)}), most recent first:")
            for x in docs.sort_values("study_date", ascending=False).head(20).itertuples():
                lines.append(f"- {x.study_date:%Y-%m-%d} {x.title} ({x.specialty}, {x.doc_id}): {x.impression or 'see document'}")
    if f.patient_ids and len(st):
        lines.append(f"Study timeline (most recent first, up to {MAX_TIMELINE}):")
        for x in st.sort_values("study_date", ascending=False).head(MAX_TIMELINE).itertuples():
            who = f"[{x.mrn}] " if len(f.patient_ids) > 1 else ""
            lines.append(f"- {who}{x.study_date:%Y-%m-%d} {C.MODALITY_NAMES.get(x.modality, x.modality)} {x.body_part} "
                         f"({x.priority}, study {x.study_id}): {x.impression}")
    return "\n".join(lines)


def build_prompt(question: str, r: Result, facts: str) -> str:
    sources = "\n\n".join(f"[S{i}] ({row.doc_id})\n{row.text}" for i, row in enumerate(r.hits.itertuples(), 1))
    parts = [f"QUESTION: {question}"]
    if r.notes:
        parts.append("NOTES: " + " ".join(r.notes))
    if facts:
        parts.append(f"COMPUTED FACTS:\n{facts}")
    parts.append(f"SOURCES:\n{sources or '(none)'}")
    return "\n\n".join(parts)


class RAG:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm()

    def prepare(self, question: str, mode: str = "hybrid+rerank", k: int | None = None) -> tuple[Result, str, str, float]:
        t0 = time.time()
        r = self.retriever.search(question, k=k, mode=mode)
        facts = computed_facts(r, self.retriever)
        return r, facts, build_prompt(question, r, facts), time.time() - t0

    def stream(self, question: str, mode: str = "hybrid+rerank") -> tuple[Result, str, Iterator[str]]:
        r, facts, prompt, _ = self.prepare(question, mode)
        if r.filters.unresolved_patient:
            msg = f"No patient matching '{r.filters.unresolved_patient}' was found in the records."
            return r, facts, iter([msg])
        return r, facts, self.llm.stream(SYSTEM, prompt)

    def ask(self, question: str, mode: str = "hybrid+rerank") -> Answer:
        r, facts, prompt, t_ret = self.prepare(question, mode)
        t0 = time.time()
        if r.filters.unresolved_patient:
            text = f"No patient matching '{r.filters.unresolved_patient}' was found in the records."
        else:
            text = self.llm.complete(SYSTEM, prompt)
        return Answer(question, text.strip(), r, facts, self.llm.label, t_ret, time.time() - t0)
