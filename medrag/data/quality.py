"""Data quality checks. Run on raw data before cleaning and on clean data after, and compare."""
import json
from datetime import datetime

import pandas as pd

from medrag.data import catalog as C
from medrag.data.clean import strip_hidden_unicode

NOW = pd.Timestamp(datetime.now())


def _pct(n: int, total: int) -> float:
    return round(100 * n / total, 2) if total else 0.0


def check(t: dict[str, pd.DataFrame]) -> list[dict]:
    """Return a list of issues: {table, check, count, pct, severity}."""
    issues = []

    def add(table, name, mask_or_count, severity="warn"):
        total = len(t[table])
        count = int(mask_or_count.sum()) if hasattr(mask_or_count, "sum") else int(mask_or_count)
        issues.append({"table": table, "check": name, "count": count, "pct": _pct(count, total), "severity": severity if count else "ok"})

    p, s, r, a = t["patients"], t["studies"], t["reports"], t["ai_analysis"]
    names = p["full_name"].fillna("")

    add("patients", "duplicate patient_id", p["patient_id"].duplicated(), "error")
    add("patients", "duplicate MRN (same person, several ids)", p["mrn"].duplicated(keep=False))
    add("patients", "duplicate name + date of birth", p.dropna(subset=["date_of_birth"]).duplicated(["full_name", "date_of_birth"], keep=False))
    add("patients", "name has stray whitespace or wrong case", (names != names.str.strip()) | (names != names.str.strip().str.title()))
    add("patients", "test / dummy record", names.str.lower().str.contains(r"\btest\b|dummy|do not use", regex=True), "error")
    add("patients", "missing name", names.str.strip() == "", "error")
    add("patients", "non-standard gender value", ~p["gender"].fillna("").isin(["male", "female", "other", "unknown"]))
    add("patients", "missing date of birth", p["date_of_birth"].isna())
    add("patients", "date of birth in the future", p["date_of_birth"] > NOW, "error")

    known_pids = set(p["patient_id"])
    add("studies", "duplicate study row", s.duplicated(), "error")
    add("studies", "duplicate study_id", s["study_id"].duplicated(), "error")
    add("studies", "orphan (patient does not exist)", ~s["patient_id"].isin(known_pids), "error")
    add("studies", "non-standard modality", ~s["modality"].fillna("").isin(C.MODALITIES))
    add("studies", "non-standard priority", ~s["priority"].fillna("").isin(C.PRIORITIES))
    add("studies", "missing study date", s["study_date"].isna(), "error")
    add("studies", "study date in the future", s["study_date"] > NOW, "error")
    add("studies", "missing body part", s["body_part"].isna())

    known_sids = set(s["study_id"])
    text = r["report_text"].fillna("")
    add("reports", "orphan (study does not exist)", ~r["study_id"].isin(known_sids), "error")
    add("reports", "empty report text", text.str.strip() == "", "error")
    add("reports", "contains HTML markup", text.str.contains(r"<[a-zA-Z/][^>]*>", regex=True))
    add("reports", "hidden Unicode characters (zero-width, bidi, tag)", strip_hidden_unicode(text) != text)
    add("reports", "duplicate report for same study", r["study_id"].duplicated(keep=False))
    add("reports", "non-standard status", ~r["status"].fillna("").isin(C.REPORT_STATUSES))
    add("reports", "missing IMPRESSION section", ~text.str.contains("IMPRESSION", case=False) & (text.str.strip() != ""))

    conf = pd.to_numeric(a["confidence"], errors="coerce")
    add("ai_analysis", "orphan (study does not exist)", ~a["study_id"].isin(known_sids), "error")
    add("ai_analysis", "missing confidence", conf.isna())
    add("ai_analysis", "confidence outside 0-1", (conf < 0) | (conf > 1), "error")
    add("ai_analysis", "studies without AI analysis", int((~s["study_id"].isin(set(a["study_id"]))).sum()), "info")
    return issues


def summary(t: dict[str, pd.DataFrame]) -> dict:
    s = t["studies"]
    return {
        "rows": {k: len(v) for k, v in t.items()},
        "studies_per_patient": s.groupby("patient_id").size().describe().round(2).to_dict() if len(s) else {},
        "modality_counts": s["modality"].value_counts().head(15).to_dict(),
        "date_range": [str(s["study_date"].min()), str(s["study_date"].max())],
    }


def to_markdown(stage: str, issues: list[dict], info: dict) -> str:
    lines = [f"# Data quality report: {stage}", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}", "", "## Row counts", ""]
    lines += [f"- **{k}**: {v:,}" for k, v in info["rows"].items()]
    lines += ["", f"Study dates: {info['date_range'][0]} to {info['date_range'][1]}", "", "## Checks", "",
              "| Table | Check | Rows | % | Status |", "|---|---|---:|---:|---|"]
    icon = {"ok": "OK", "info": "info", "warn": "WARN", "error": "ERROR"}
    for i in issues:
        lines.append(f"| {i['table']} | {i['check']} | {i['count']:,} | {i['pct']} | {icon[i['severity']]} |")
    bad = [i for i in issues if i["severity"] == "error"]
    lines += ["", f"**{len(bad)} error checks failing.**" if bad else "**All error-level checks pass.**"]
    return "\n".join(lines)


def run(stage: str, t: dict[str, pd.DataFrame], out_dir) -> tuple[list[dict], str]:
    issues, info = check(t), summary(t)
    md = to_markdown(stage, issues, info)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"quality_{stage}.md").write_text(md, encoding="utf-8")
    (out_dir / f"quality_{stage}.json").write_text(json.dumps({"issues": issues, "summary": info}, indent=2, default=str), encoding="utf-8")
    return issues, md
