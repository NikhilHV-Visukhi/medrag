"""Cleaning rules. Each rule logs how many rows it changed or removed."""
import html
import re
from datetime import datetime

import pandas as pd

from medrag.data import catalog as C
from medrag.data.text_unicode import clean_text as _unicode_clean

NOW = pd.Timestamp(datetime.now())
TEST_NAME = re.compile(r"\btest\b|dummy|do not use", re.I)
NON_ASCII = re.compile(r"[^\x00-\x7f]")


def strip_hidden_unicode(text: pd.Series) -> pd.Series:
    """Remove invisible characters (zero-width, bidi overrides, tag chars) and turn exotic spaces into spaces.

    They are invisible to a reader but reach BM25, the embedder and the LLM, where they can hide instructions.
    The character-level cleaner is slow, so only texts with a non-ASCII character go through it.
    """
    text = text.fillna("").astype(str)
    has = text.str.contains(NON_ASCII)
    return text.where(~has, text[has].map(lambda t: _unicode_clean(t)[0]))


def _norm(series: pd.Series, synonyms: dict, default: str) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().map(synonyms).fillna(default)


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return "\n".join(re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip())


def clean(t: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    log: list[str] = []

    def note(msg, n):
        if n:
            log.append(f"{msg}: {n:,}")

    # ---------------- patients ----------------
    p = t["patients"].copy()
    n = len(p)
    p = p[~p["full_name"].fillna("").str.contains(TEST_NAME)]
    note("patients: removed test/dummy records", n - len(p))
    p = p[p["full_name"].fillna("").str.strip() != ""]
    fixed = p["full_name"].str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    note("patients: normalised name whitespace/case", int((fixed != p["full_name"]).sum()))
    p["full_name"] = fixed
    g = _norm(p["gender"], C.GENDER_SYNONYMS, "unknown")
    note("patients: normalised gender values", int((g != p["gender"]).sum()))
    p["gender"] = g
    bad_dob = p["date_of_birth"] > NOW
    note("patients: cleared impossible future date of birth", int(bad_dob.sum()))
    p.loc[bad_dob, "date_of_birth"] = pd.NaT

    # Same MRN under several patient ids = one person registered twice. Keep the lowest id, remap the rest.
    # Patients without an MRN cannot be matched, so they are kept as they are.
    p = p.sort_values("patient_id")
    has_mrn = p["mrn"].fillna("").astype(str).str.strip() != ""
    d = p[has_mrn]
    keep_id = d.groupby("mrn")["patient_id"].transform("min")
    remap = dict(zip(d.loc[keep_id != d["patient_id"], "patient_id"], keep_id[keep_id != d["patient_id"]]))
    # Fill gaps in the surviving record from its duplicates before dropping them.
    d = d.groupby("mrn", as_index=False).agg({c: "first" for c in d.columns if c != "mrn"} | {"patient_id": "min"})
    note("patients: kept without MRN (cannot be de-duplicated)", int((~has_mrn).sum()))
    p = pd.concat([d, p[~has_mrn]], ignore_index=True)
    note("patients: merged duplicate registrations (same MRN)", len(remap))
    t_cond = t["conditions"].copy()
    t_cond["patient_id"] = t_cond["patient_id"].replace(remap)
    t_cond = t_cond.drop_duplicates(["patient_id", "condition"])

    # ---------------- studies ----------------
    s = t["studies"].copy()
    n = len(s)
    s = s.drop_duplicates()
    s = s.drop_duplicates("study_id")
    note("studies: removed duplicate rows", n - len(s))
    note("studies: re-pointed to merged patient", int(s["patient_id"].isin(set(remap)).sum()))
    s["patient_id"] = s["patient_id"].replace(remap)
    n = len(s)
    s = s[s["patient_id"].isin(set(p["patient_id"]))]
    note("studies: removed orphans (unknown patient)", n - len(s))
    mod = s["modality"].fillna("").str.strip().str.lower().map(C.MODALITY_SYNONYMS)
    mod = mod.fillna(s["modality"].where(s["modality"].isin(C.MODALITIES)))
    note("studies: normalised modality names", int((mod != s["modality"]).sum()))
    s["modality"] = mod
    n = len(s)
    s = s[s["modality"].notna()]
    note("studies: removed unknown modality", n - len(s))
    pr = _norm(s["priority"], C.PRIORITY_SYNONYMS, "routine")
    note("studies: normalised priority values", int((pr != s["priority"]).sum()))
    s["priority"] = pr
    n = len(s)
    s = s[s["study_date"].notna() & (s["study_date"] <= NOW)]
    note("studies: removed missing / future study dates", n - len(s))

    # ---------------- reports ----------------
    r = t["reports"].copy()
    has_html = r["report_text"].fillna("").str.contains(r"<[a-zA-Z/][^>]*>", regex=True)
    r.loc[has_html, "report_text"] = r.loc[has_html, "report_text"].map(_strip_html)
    note("reports: stripped HTML markup", int(has_html.sum()))
    fixed = strip_hidden_unicode(r["report_text"])
    note("reports: removed hidden Unicode characters", int((fixed != r["report_text"].fillna("")).sum()))
    r["report_text"] = fixed
    n = len(r)
    r = r[r["report_text"].fillna("").str.strip() != ""]
    note("reports: removed empty reports", n - len(r))
    r = r[r["study_id"].isin(set(s["study_id"]))]
    st = _norm(r["status"], C.STATUS_SYNONYMS, "preliminary")
    note("reports: normalised status values", int((st != r["status"]).sum()))
    r["status"] = st
    # One report per study: prefer final > amended > preliminary, then latest signed.
    rank = r["status"].map({"final": 0, "amended": 1, "preliminary": 2})
    n = len(r)
    r = r.assign(_rank=rank).sort_values(["study_id", "_rank", "signed_at"], ascending=[True, True, False])
    r = r.drop_duplicates("study_id").drop(columns="_rank")
    note("reports: removed duplicate reports per study", n - len(r))

    # ---------------- ai_analysis ----------------
    a = t["ai_analysis"].copy()
    a = a[a["study_id"].isin(set(s["study_id"]))]
    conf = pd.to_numeric(a["confidence"], errors="coerce")
    pct = (conf > 1) & (conf <= 100)
    note("ai_analysis: converted percent confidence to 0-1", int(pct.sum()))
    conf = conf.where(~pct, conf / 100)
    invalid = (conf < 0) | (conf > 1)
    note("ai_analysis: cleared invalid confidence", int(invalid.sum()))
    a["confidence"] = conf.where(~invalid)

    # Studies with no usable report carry no clinical text; keep them (metadata is still useful).
    note("studies: without a report after cleaning", int((~s["study_id"].isin(set(r["study_id"]))).sum()))
    return {"patients": p, "conditions": t_cond, "studies": s, "ai_analysis": a, "reports": r}, log
