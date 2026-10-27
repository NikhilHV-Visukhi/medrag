"""Check, clean and link outside documents to patients."""
import re

import numpy as np
import pandas as pd

HEADING_CSV = re.compile(r"\s*([A-Z][A-Z0-9 /&(),'-]{2,60}):,\s*")  # MTSamples: "HISTORY OF PRESENT ILLNESS:,  The..."
AGE_RE = re.compile(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\b", re.I)
MALE_RE = re.compile(r"\b(he|his|him|mr|male|man|gentleman|boy)\b", re.I)
FEMALE_RE = re.compile(r"\b(she|her|hers|mrs|ms|miss|female|woman|lady|girl)\b", re.I)
WINDOW = (pd.Timestamp("2020-10-01"), pd.Timestamp("2026-09-30"))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def check(docs: pd.DataFrame) -> list[dict]:
    """Same issue format as medrag.data.quality.check."""
    text = docs["text"].fillna("").astype(str)
    words = text.str.split().str.len()
    n = len(docs)

    def add(name, count, severity="warn"):
        count = int(count)
        return {"table": "documents", "check": name, "count": count, "pct": round(100 * count / n, 2) if n else 0,
                "severity": severity if count else "ok"}

    mts = docs["source"] == "mtsamples"
    return [
        add("empty text", (text.str.strip() == "").sum(), "error"),
        add("very short (< 15 words)", ((words < 15) & (text.str.strip() != "")).sum(), "error"),
        add("duplicate document (same text, same source)", (mts & docs.assign(_n=text.map(_norm)).duplicated(["source", "_n"])).sum(), "error"),
        add("transcription artefacts (':,' headings)", text.str.contains(r"[A-Z]:,", regex=True).sum()),
        add("de-identification tokens (XXXX)", text.str.contains("XXXX").sum(), "info"),
        add("Open-i report without IMPRESSION", ((docs["source"] == "openi") & ~text.str.contains("IMPRESSION:")).sum()),
        add("missing patient link", docs["patient_id"].isna().sum() if "patient_id" in docs else n, "info"),
    ]


def clean(docs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log: list[str] = []
    d = docs.copy()
    d["text"] = d["text"].fillna("").astype(str)
    fixed = d["text"].str.contains(r"[A-Z]:,", regex=True)
    d["text"] = d["text"].map(lambda t: HEADING_CSV.sub(lambda m: f"\n{m.group(1)}: ", t))
    d["text"] = d["text"].str.replace(r",\s*,", ",", regex=True)
    log.append(f"documents: rebuilt section headings in transcriptions: {int(fixed.sum()):,}")
    red = d["text"].str.contains("XXXX")
    d["text"] = d["text"].str.replace(r"\bX{2,}\b", "[redacted]", regex=True).str.replace(r"X{4}", "[redacted]", regex=True)
    log.append(f"documents: marked de-identified words as [redacted]: {int(red.sum()):,}")
    d["text"] = d["text"].map(lambda t: "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.splitlines() if ln.strip()))
    n = len(d)
    d = d[d["text"].str.split().str.len() >= 15]
    log.append(f"documents: removed empty / very short: {n - len(d):,}")
    # The same transcription is filed under several specialties in MTSamples: keep one, remember the others.
    # Only MTSamples: identical Open-i reports ("No acute disease.") are different patients.
    d = d.assign(_n=d["text"].map(_norm).where(d["source"] == "mtsamples", d.index.astype(str)))
    also = d.groupby(["source", "_n"])["specialty"].agg(lambda s: "; ".join(dict.fromkeys(s)))
    n = len(d)
    d = d.drop_duplicates(["source", "_n"]).copy()
    d["specialty"] = [also[(s, k)] for s, k in zip(d["source"], d["_n"])]
    log.append(f"documents: merged duplicate transcriptions: {n - len(d):,}")
    return d.drop(columns="_n").reset_index(drop=True), log


def link_to_patients(docs: pd.DataFrame, patients: pd.DataFrame, seed: int = 13) -> tuple[pd.DataFrame, list[str]]:
    """Attach each document to a plausible synthetic patient: same gender and age (when the text states them).

    The text is real; the patient link is synthetic, so questions can be asked per patient.
    """
    rng = np.random.default_rng(seed)
    p = patients.dropna(subset=["date_of_birth"]).copy()
    p = p[p["gender"].isin(["male", "female"])]
    dob = p["date_of_birth"].to_numpy(dtype="datetime64[ns]")
    gender = p["gender"].to_numpy()
    pids = p["patient_id"].to_numpy()
    span = (WINDOW[1] - WINDOW[0]).days
    out_pid, out_date, matched = [], [], {"age+gender": 0, "gender": 0, "none": 0}
    for t in docs["text"]:
        m, f = len(MALE_RE.findall(t)), len(FEMALE_RE.findall(t))
        g = "male" if m > f else "female" if f > m else None
        when = WINDOW[0] + pd.Timedelta(days=int(rng.integers(span)))
        cand = np.ones(len(p), bool) if g is None else gender == g
        if (a := AGE_RE.search(t)) and 0 < int(a.group(1)) < 100:
            age = (np.datetime64(when) - dob).astype("timedelta64[D]").astype(int) / 365.25
            near = cand & (np.abs(age - int(a.group(1))) <= 2)
            if near.any():
                cand = near
                matched["age+gender"] += 1
            else:
                matched["gender" if g else "none"] += 1
        else:
            matched["gender" if g else "none"] += 1
        out_pid.append(int(pids[rng.choice(np.flatnonzero(cand))]))
        out_date.append(when)
    d = docs.copy()
    d.insert(0, "document_id", np.arange(1, len(d) + 1))
    d.insert(1, "patient_id", out_pid)
    d["doc_date"] = out_date
    log = [f"documents: linked to patients by age+gender {matched['age+gender']:,}, by gender {matched['gender']:,}, "
           f"at random (no gender/age in text) {matched['none']:,}"]
    return d, log
