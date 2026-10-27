"""End-to-end check on a small dataset: offline (parquet), hash embeddings, no reranker, no LLM."""
import pandas as pd
import pytest

from medrag.config import get_settings


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("medrag")
    mp = pytest.MonkeyPatch()
    for k, v in {"DATA_BACKEND": "parquet", "DATA_DIR": str(tmp / "data"), "INDEX_DIR": str(tmp / "idx"),
                 "REPORTS_DIR": str(tmp / "reports"), "EMBED_PROVIDER": "hash", "EMBED_MODEL": "bow", "RERANK_MODEL": "none"}.items():
        mp.setenv(k, v)
    get_settings.cache_clear()
    yield get_settings()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def built(env):
    from medrag.data import store
    from medrag.data.clean import clean
    from medrag.data.generate import generate
    from medrag.index.store import build
    from medrag.ingest.documents import build_chunks
    from medrag.providers.embeddings import get_embedder

    raw = generate(n_patients=400, mean_studies=4, dirty_rate=0.05, seed=3)
    store.write_all("raw", raw)
    cleaned, log = clean(store.read_all("raw"))
    store.write_all("clean", cleaned)
    chunks = build_chunks(store.read_all("clean"))
    manifest = build(chunks, get_embedder(), env.index_path())
    return raw, cleaned, log, chunks, manifest


def test_raw_data_is_dirty_and_clean_data_passes(built):
    from medrag.data.quality import check

    raw, cleaned, log, _, _ = built
    raw_errors = [i for i in check(raw) if i["severity"] == "error"]
    clean_errors = [i for i in check(cleaned) if i["severity"] == "error"]
    assert len(raw_errors) >= 5
    assert clean_errors == [], clean_errors
    assert any("merged duplicate registrations" in line for line in log)


def test_chunks_have_context_headers(built):
    chunks = built[3]
    study = chunks[chunks["doc_type"] == "study"].iloc[0]
    assert study["text"].startswith("Patient: ") and "MRN" in study["text"] and "IMPRESSION" in study["text"]
    assert chunks["chunk_id"].is_unique
    assert set(chunks["doc_type"]) == {"study", "patient"}


def test_incremental_rebuild_reuses_vectors(env, built):
    from medrag.index.store import build
    from medrag.providers.embeddings import get_embedder

    m = build(built[3], get_embedder(), env.index_path())
    assert m["embedded_now"] == 0 and m["reused"] == m["chunks"]


def test_parser_and_retrieval(env, built):
    from medrag.retrieval.retriever import Retriever

    ret = Retriever()
    st = ret.meta[(ret.meta["doc_type"] == "study") & (ret.meta["part"] == 0)].iloc[5]
    q = f"What did the {st.modality} {st.body_part} on {st.study_date:%Y-%m-%d} show for {st.patient_name}?"
    r = ret.search(q, k=5, mode="hybrid")
    assert st.patient_id in r.filters.patient_ids
    assert st.doc_id in r.hits["doc_id"].tolist()
    assert set(r.hits["patient_id"]) <= set(r.filters.patient_ids)  # no other patients leak in

    r = ret.search(f"latest scan for {st.mrn}", k=5)
    assert r.filters.latest and r.hits.iloc[0]["doc_type"] == "patient"

    r = ret.search("Summarise the imaging history of patient Zebulon Quixote")
    assert r.filters.unresolved_patient == "Zebulon Quixote"


def test_computed_facts_count_exactly(env, built):
    from medrag.rag.answer import computed_facts
    from medrag.retrieval.retriever import Retriever

    ret = Retriever()
    pid = ret.meta["patient_id"].value_counts().index[0]
    name = ret.meta.loc[ret.meta["patient_id"] == pid, "patient_name"].iloc[0]
    r = ret.search(f"How many studies does {name} have?")
    facts = computed_facts(r, ret.meta)
    n = ret.meta[(ret.meta["patient_id"] == pid) & (ret.meta["doc_type"] == "study") & (ret.meta["part"] == 0)].shape[0]
    assert f"Matching studies: {n:,}" in facts


def test_eval_metrics():
    from medrag.eval.run import retrieval_metrics

    m = retrieval_metrics(["a", "b", "c"], ["b"], 3)
    assert m["hit"] == 1 and m["mrr"] == 0.5 and m["recall"] == 1
    assert retrieval_metrics(["a"], ["z"], 3)["hit"] == 0


def test_chunking_splits_long_text():
    from medrag.ingest.chunking import chunk_document

    body = "FINDINGS: " + " ".join(f"Sentence number {i} is here." for i in range(200)) + "\nIMPRESSION: Fine."
    chunks = chunk_document("Patient: X", body, max_words=120)
    assert len(chunks) > 3 and all(c.startswith("Patient: X") for c in chunks)
    assert all(len(c.split()) <= 140 for c in chunks)
    assert isinstance(pd.Series(chunks), pd.Series)
