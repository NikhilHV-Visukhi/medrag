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

    from medrag.index.store import Index

    m = build(built[3], get_embedder(), env.index_path())
    assert m["embedded_now"] == 0 and m["reused"] == m["chunks"]
    assert len(list(env.index_path().glob("vectors*.npy"))) == 1  # the previous build's files are cleaned up
    assert len(Index(env.index_path()).meta) == m["chunks"]


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
    facts = computed_facts(r, ret)
    n = ret.meta[(ret.meta["patient_id"] == pid) & (ret.meta["doc_type"] == "study") & (ret.meta["part"] == 0)].shape[0]
    assert f"Matching studies: {n:,}" in facts


def test_eval_metrics():
    from medrag.eval.run import retrieval_metrics

    m = retrieval_metrics(["a", "b", "c"], ["b"], 3)
    assert m["hit"] == 1 and m["mrr"] == 0.5 and m["recall"] == 1
    assert retrieval_metrics(["a"], ["z"], 3)["hit"] == 0


def test_query_expansion_maps_lay_terms():
    from medrag.retrieval.expansion import detect_findings, expand

    q, notes = expand("Has she ever had a slipped disc or fatty liver?")
    assert "disc extrusion" in q and "hepatic steatosis" in q and len(notes) == 2
    assert [lbl for _, lbl in detect_findings("brain bleed")] == ["intracranial haemorrhage"]
    assert expand("CT chest on 2024-01-01")[1] == []


def test_answer_scoring():
    from medrag.eval.run import ci, score_answer

    assert score_answer({"number": 1234}, "There were 1,234 studies.")["auto_correct"] == 1
    assert score_answer({"expect_no": True}, "No, she has never had one.")["auto_correct"] == 1
    assert score_answer({"facts": ["improv|better"]}, "It has improved.")["auto_correct"] == 1
    assert score_answer({"facts": ["2024-01-01", "ct|x"]}, "On 2024-01-01 an MRI")["fact_recall"] == 0.5
    assert ci([1, 0, 1, 1]).startswith("0.750 [")


def test_hard_testset(env, built):
    from medrag.eval import testset
    from medrag.retrieval.retriever import Retriever

    items = testset.build(Retriever().meta, per_type=5)
    kinds = {i["type"] for i in items if i["difficulty"] == "hard"}
    assert {"hard_paraphrase_patient", "hard_negative", "hard_typo_name", "hard_count_finding"} <= kinds


def test_document_cleaning_and_linking(built):
    from medrag.documents.prepare import clean, link_to_patients

    raw = pd.DataFrame({
        "source": ["mtsamples", "mtsamples", "openi", "openi", "openi"],
        "source_id": ["a", "b", "c", "d", "e"], "title": ["Consult"] * 2 + ["CXR"] * 3,
        "specialty": ["Cardiology", "Surgery", "Radiology", "Radiology", "Radiology"],
        "text": ["HISTORY OF PRESENT ILLNESS:,  This 45-year-old female presents with chest pain over two days and mild dyspnea on exertion today.,"] * 2
                + ["FINDINGS: Normal heart size. Lungs are clear. No effusion. IMPRESSION: No acute disease XXXX seen today at all."] * 2 + ["short"],
        "labels": None, "description": "x",
    })
    docs, log = clean(raw)
    assert len(docs) == 3  # mtsamples duplicate merged, identical Open-i reports kept, short one dropped
    assert docs.iloc[0]["text"].startswith("HISTORY OF PRESENT ILLNESS: This") and "Surgery" in docs.iloc[0]["specialty"]
    assert "[redacted]" in docs.iloc[1]["text"]
    linked, _ = link_to_patients(docs, built[1]["patients"])
    assert linked["patient_id"].notna().all()
    pt = built[1]["patients"].set_index("patient_id").loc[linked.iloc[0]["patient_id"]]
    assert pt["gender"] == "female"


def test_chunking_splits_long_text():
    from medrag.ingest.chunking import chunk_document

    body = "FINDINGS: " + " ".join(f"Sentence number {i} is here." for i in range(200)) + "\nIMPRESSION: Fine."
    chunks = chunk_document("Patient: X", body, max_words=120)
    assert len(chunks) > 3 and all(c.startswith("Patient: X") for c in chunks)
    assert all(len(c.split()) <= 140 for c in chunks)
    assert isinstance(pd.Series(chunks), pd.Series)


def test_count_is_zero_when_patient_lacks_the_modality(env, built):
    from medrag.rag.answer import computed_facts
    from medrag.retrieval.retriever import Retriever

    ret = Retriever()
    st = ret.meta[ret.meta["doc_type"] == "study"]
    no_mri = set(st["patient_id"]) - set(st.loc[st["modality"] == "MRI", "patient_id"])
    mrn = st.loc[st["patient_id"].isin(no_mri), "mrn"].iloc[0]
    r = ret.search(f"How many MRI studies does {mrn} have?")
    assert r.filters.modality == "MRI" and any("relaxed" in n for n in r.notes)  # search still shows other studies
    assert "Matching studies: 0 " in computed_facts(r, ret)  # but the count is not taken from them


def test_short_modality_codes_only_in_capitals():
    from medrag.retrieval.query_parser import QueryParser

    meta = pd.DataFrame({"patient_id": [1], "patient_name": ["John Smith"], "mrn": ["MRN0000001"],
                         "study_id": [pd.NA], "text": ["x"], "specialty": [None]})
    qp = QueryParser(meta)
    assert qp.parse("Show the US abdomen for John Smith").modality == "US"
    assert qp.parse("Any PET scans last year?").modality == "PET"
    assert qp.parse("Can you tell us about John Smith").modality is None
    assert qp.parse("Mr Smith had an ultrasound").modality == "US"


def test_clean_keeps_patients_without_mrn():
    from medrag.data.clean import clean
    from medrag.data.store import COLUMNS

    p = pd.DataFrame({"patient_id": [1, 2, 3], "mrn": ["MRN1", None, "MRN1"], "full_name": ["Ann Lee", "Bob Ray", "Ann Lee"],
                      "gender": ["F", "M", "F"], "date_of_birth": pd.to_datetime(["1980-01-01", "1970-01-01", "1980-01-01"]),
                      "blood_group": None, "city": None})
    s = pd.DataFrame({"study_id": [10, 20, 30], "patient_id": [1, 2, 3], "accession": ["a", "b", "c"],
                      "study_date": pd.to_datetime(["2024-01-01"] * 3), "modality": "CT", "body_part": "chest",
                      "indication": "x", "priority": "routine", "referring_physician": None})
    empty = {k: pd.DataFrame(columns=list(COLUMNS[k])) for k in ("conditions", "ai_analysis", "reports")}
    out, log = clean({"patients": p, "studies": s} | empty)
    assert sorted(out["patients"]["patient_id"]) == [1, 2]  # 3 merged into 1, 2 kept despite no MRN
    assert sorted(out["studies"]["patient_id"]) == [1, 1, 2]
    assert "patients: merged duplicate registrations (same MRN): 1" in log
