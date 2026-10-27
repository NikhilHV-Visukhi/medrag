"""Streamlit UI:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from medrag.config import get_settings
from medrag.data import catalog as C
from medrag.index.store import Index
from medrag.providers.llms import get_llm
from medrag.rag.answer import RAG
from medrag.retrieval.retriever import MODES, Retriever

s = get_settings()
st.set_page_config(page_title="MedRAG", page_icon="🩺", layout="wide")

indexes = sorted(p for p in s.index_dir.glob("*") if (p / "manifest.json").exists()) if s.index_dir.exists() else []
if not indexes:
    st.error("No index found. Build one first:  `python -m medrag pipeline`  (or `clean` + `index`).")
    st.stop()


@st.cache_resource(show_spinner="Loading index...")
def load_retriever(path: str, built_at: str) -> Retriever:
    return Retriever(Index(Path(path)))


with st.sidebar:
    st.header("Settings")
    default_idx = next((i for i, p in enumerate(indexes) if p == s.index_path()), 0)
    idx_path = st.selectbox("Embedding index", indexes, index=default_idx, format_func=lambda p: p.name)
    manifest = json.loads((idx_path / "manifest.json").read_text())
    st.caption(f"{manifest['chunks']:,} chunks · {manifest['dim']}-d · built {manifest['built_at']}")
    llm_spec = st.text_input("LLM (provider:model)", f"{s.llm_provider}:{s.llm_model}",
                             help="e.g. ollama:qwen2.5:3b · anthropic:claude-opus-5-5 · openai:<model> · gemini:<model>")
    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index("hybrid+rerank"))
    top_k = st.slider("Sources per answer", 3, 20, s.top_k)
    if st.button("Clear chat"):
        st.session_state.messages = []

retriever = load_retriever(str(idx_path), manifest["built_at"])
meta = retriever.meta

chat_tab, patient_tab, quality_tab, eval_tab = st.tabs(["Chat", "Patients", "Data quality", "Evaluation"])

# ---------------- chat ----------------
with chat_tab:
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question := st.chat_input("Ask about a patient, a study, or the whole cohort..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                rag = RAG(retriever, get_llm(llm_spec))
                r, facts, stream = rag.stream(question, mode)
                st.caption(f"Filters: {r.filters.describe()}" + (f" · {' '.join(r.notes)}" if r.notes else ""))
                text = st.write_stream(stream)
            except Exception as e:
                text = f"Error: {e}"
                st.error(text)
                r = facts = None
            st.session_state.messages.append({"role": "assistant", "content": text})
            if r is not None:
                with st.expander(f"Sources ({len(r.hits)})"):
                    show = r.hits[["doc_id", "via", "score", "patient_name", "mrn", "study_date", "modality", "body_part", "priority"]].copy()
                    show.insert(0, "src", [f"S{i}" for i in range(1, len(show) + 1)])
                    st.dataframe(show, hide_index=True, use_container_width=True)
                    for i, row in enumerate(r.hits.itertuples(), 1):
                        st.text(f"[S{i}] {row.doc_id}\n{row.text}")
                if facts:
                    with st.expander("Computed facts (exact database counts)"):
                        st.text(facts)

# ---------------- patients ----------------
with patient_tab:
    q = st.text_input("Search by name, MRN or patient ID")
    if q:
        people = meta[meta["doc_type"] == "patient"].drop_duplicates("patient_id")
        ql = q.strip().lower()
        found = people[people["patient_name"].str.lower().str.contains(ql, regex=False) | (people["mrn"].str.lower() == ql)
                       | (people["patient_id"].astype(str) == ql)]
        st.write(f"{len(found)} match(es)")
        for pt in found.head(10).itertuples():
            with st.expander(f"{pt.patient_name} · {pt.mrn} · id {pt.patient_id}", expanded=len(found) == 1):
                for t in meta[(meta["doc_id"] == f"patient:{pt.patient_id}")].sort_values("part")["text"]:
                    st.text(t)
                studies = meta[(meta["patient_id"] == pt.patient_id) & (meta["doc_type"] == "study") & (meta["part"] == 0)]
                studies = studies.sort_values("study_date", ascending=False)
                studies = studies.assign(modality=studies["modality"].map(C.MODALITY_NAMES))
                st.dataframe(studies[["study_id", "study_date", "modality", "body_part", "priority", "impression", "ai_label", "ai_confidence"]],
                             hide_index=True, use_container_width=True)

# ---------------- data quality ----------------
with quality_tab:
    st.metric("Patients", f"{meta['patient_id'].nunique():,}")
    st.metric("Studies indexed", f"{meta.loc[meta['doc_type'] == 'study', 'study_id'].nunique():,}")
    for name in ("cleaning_log.md", "quality_raw.md", "quality_clean.md"):
        f = s.reports_dir / name
        if f.exists():
            with st.expander(name, expanded=name == "cleaning_log.md"):
                st.markdown(f.read_text(encoding="utf-8"))

# ---------------- evaluation ----------------
with eval_tab:
    runs = sorted(s.reports_dir.glob("eval_*/summary.md"), reverse=True) if s.reports_dir.exists() else []
    if not runs:
        st.info("No evaluation yet. Run `python -m medrag eval`.")
    else:
        pick = st.selectbox("Run", runs, format_func=lambda p: p.parent.name)
        st.markdown(pick.read_text(encoding="utf-8"))
        gen = pick.parent / "generation.csv"
        if gen.exists():
            st.subheader("Answers")
            st.dataframe(pd.read_csv(gen), use_container_width=True)
