"""Turn clean tables into retrievable chunks: one set per study report + one per patient profile."""
import hashlib
import re
from datetime import datetime

import pandas as pd

from medrag.data import catalog as C
from medrag.ingest.chunking import chunk_document

TIMELINE_ROWS = 40  # most recent studies listed in a patient profile


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def _age(dob, at) -> str:
    if pd.isna(dob):
        return "unknown"
    return str(int((pd.Timestamp(at) - pd.Timestamp(dob)).days // 365))


def _impression(text: str) -> str:
    for line in str(text).splitlines():
        if line.upper().startswith("IMPRESSION:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_chunks(t: dict[str, pd.DataFrame], max_words: int = 220) -> pd.DataFrame:
    p = t["patients"].set_index("patient_id")
    s = t["studies"].merge(t["reports"][["study_id", "report_text", "status", "radiologist"]], on="study_id", how="left")
    ai = t["ai_analysis"].sort_values("analysis_id").drop_duplicates("study_id", keep="last")
    s = s.merge(ai[["study_id", "model_name", "predicted_label", "confidence"]], on="study_id", how="left")
    s = s.join(p[["mrn", "full_name", "gender", "date_of_birth"]], on="patient_id")
    conds = t["conditions"].groupby("patient_id")["condition"].apply(lambda x: ", ".join(sorted(x)))

    rows = []
    today = datetime.now()
    docs = t.get("documents")
    docs_by_patient = {} if docs is None else {pid: g.sort_values("doc_date", ascending=False) for pid, g in docs.groupby("patient_id")}

    # ---- study report chunks ----
    for r in s.itertuples(index=False):
        if pd.isna(r.report_text):
            continue
        date = pd.Timestamp(r.study_date).strftime("%Y-%m-%d")
        modality = C.MODALITY_NAMES.get(r.modality, r.modality)
        ai_line = ""
        if pd.notna(r.predicted_label):
            conf = f"{r.confidence:.2f}" if pd.notna(r.confidence) else "n/a"
            ai_line = f"\nAI MODEL ({r.model_name}): predicted '{r.predicted_label}' (confidence {conf})."
        header = (
            f"Patient: {r.full_name} | MRN: {r.mrn} | Patient ID: {r.patient_id} | {r.gender}, age {_age(r.date_of_birth, r.study_date)}\n"
            f"Study: {modality} {r.body_part} | Date: {date} | Accession: {r.accession} | Priority: {r.priority} | Report status: {r.status}\n"
            f"Indication: {r.indication}"
        )
        for i, text in enumerate(chunk_document(header, str(r.report_text) + ai_line, max_words)):
            rows.append({
                "chunk_id": _hash(text), "doc_id": f"study:{r.study_id}", "doc_type": "study", "part": i,
                "patient_id": int(r.patient_id), "study_id": int(r.study_id), "mrn": r.mrn, "patient_name": r.full_name,
                "gender": r.gender, "modality": r.modality, "body_part": r.body_part, "priority": r.priority,
                "study_date": pd.Timestamp(r.study_date), "status": r.status,
                "ai_label": r.predicted_label if pd.notna(r.predicted_label) else None,
                "ai_confidence": float(r.confidence) if pd.notna(r.confidence) else None,
                "impression": _impression(r.report_text), "text": text,
            })

    # ---- patient profile chunks ----
    for pid, grp in s.sort_values("study_date", ascending=False).groupby("patient_id", sort=False):
        pt = p.loc[pid]
        counts = grp["modality"].map(lambda m: C.MODALITY_NAMES.get(m, m)).value_counts()
        header = (
            f"PATIENT PROFILE: {pt.full_name} | MRN: {pt.mrn} | Patient ID: {pid} | {pt.gender}, age {_age(pt.date_of_birth, today)}"
            f" | Blood group: {pt.blood_group or 'unknown'} | City: {pt.city or 'unknown'}"
        )
        lines = [
            f"HISTORY: Chronic conditions: {conds.get(pid, 'none recorded')}.",
            f"HISTORY: {len(grp)} imaging studies from {grp['study_date'].min():%Y-%m-%d} to {grp['study_date'].max():%Y-%m-%d}: "
            + ", ".join(f"{k} x{v}" for k, v in counts.items()) + ".",
        ]
        timeline = [
            f"{pd.Timestamp(x.study_date):%Y-%m-%d} {C.MODALITY_NAMES.get(x.modality, x.modality)} {x.body_part} ({x.priority}): {_impression(x.report_text) or 'no report'}"
            for x in grp.head(TIMELINE_ROWS).itertuples(index=False)
        ]
        pdocs = docs_by_patient.get(pid)
        if pdocs is not None:
            lines.append(f"HISTORY: {len(pdocs)} uploaded documents: " + "; ".join(
                f"{d.doc_date:%Y-%m-%d} {d.title} ({str(d.specialty).split(';')[0]})" for d in pdocs.head(15).itertuples()) + ".")
        body = "\n".join(lines) + "\nFINDINGS: Most recent studies first. " + " ".join(f"{t_}." if not t_.endswith(".") else t_ for t_ in timeline)
        for i, text in enumerate(chunk_document(header, body, max_words + 80)):
            rows.append({
                "chunk_id": _hash(text), "doc_id": f"patient:{pid}", "doc_type": "patient", "part": i,
                "patient_id": int(pid), "study_id": None, "mrn": pt.mrn, "patient_name": pt.full_name, "gender": pt.gender,
                "modality": None, "body_part": None, "priority": None, "study_date": grp["study_date"].max(), "status": None,
                "ai_label": None, "ai_confidence": None, "impression": None, "text": text,
            })

    if docs is not None:
        rows += document_chunks(docs, p, max_words)

    df = pd.DataFrame(rows)
    for col in ("source", "title", "specialty"):
        if col not in df:
            df[col] = None
    df = df.drop_duplicates("chunk_id").reset_index(drop=True)
    df["study_id"] = df["study_id"].astype("Int64")
    return df


def _summary_line(text: str) -> str:
    """First line of the section that states the conclusion (impression / diagnosis / assessment)."""
    m = re.search(r"^(?:IMPRESSION|DIAGNOS[IE]S|POSTOPERATIVE DIAGNOSIS|FINAL DIAGNOSIS|ASSESSMENT(?: AND PLAN)?|CONCLUSION)S?:\s*(.+)$",
                  str(text), re.M)
    return m.group(1).strip()[:300] if m else ""


def document_chunks(docs: pd.DataFrame, patients: pd.DataFrame, max_words: int = 220) -> list[dict]:
    rows = []
    for d in docs.itertuples(index=False):
        if d.patient_id not in patients.index:
            continue
        pt = patients.loc[d.patient_id]
        specialty = str(d.specialty).split(";")[0]
        header = (
            f"Patient: {pt.full_name} | MRN: {pt.mrn} | Patient ID: {d.patient_id} | {pt.gender}, age {_age(pt.date_of_birth, d.doc_date)}\n"
            f"Document: {d.title} | Type: {specialty} | Date: {d.doc_date:%Y-%m-%d} | Source: {d.source}"
            + (f"\nSummary: {d.description}" if d.source == "mtsamples" and isinstance(d.description, str) and d.description else "")
        )
        is_cxr = d.source == "openi"
        for i, text in enumerate(chunk_document(header, d.text, max_words)):
            rows.append({
                "chunk_id": _hash(text), "doc_id": f"doc:{d.document_id}", "doc_type": "document", "part": i,
                "patient_id": int(d.patient_id), "study_id": None, "mrn": pt.mrn, "patient_name": pt.full_name, "gender": pt.gender,
                "modality": "XR" if is_cxr else None, "body_part": "chest" if is_cxr else None, "priority": None,
                "study_date": pd.Timestamp(d.doc_date), "status": None, "ai_label": None, "ai_confidence": None,
                "impression": _summary_line(d.text) or None, "text": text,
                "source": d.source, "title": d.title, "specialty": specialty,
            })
    return rows
