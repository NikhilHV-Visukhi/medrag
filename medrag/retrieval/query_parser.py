"""Pull structured filters out of a natural-language question (patient, modality, dates, ...)."""
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
from rapidfuzz import fuzz, process

from medrag.data import catalog as C
from medrag.retrieval.expansion import detect_findings

BODY_PARTS = sorted({bp for parts in C.MODALITY_BODY_PARTS.values() for bp in parts}, key=len, reverse=True)
BODY_SYNONYMS = {"lung": "chest", "lungs": "chest", "thorax": "chest", "liver": "abdomen", "kidney": "abdomen",
                 "abdominal": "abdomen", "brain": "brain", "lumbar": "spine", "back": "spine", "breasts": "breast"}
STOP_AFTER_PATIENT = {"id", "with", "who", "whose", "has", "had", "have", "that", "in", "on", "the", "a", "an", "named", "called",
                      "history", "summary", "patient", "patients", "imaging", "scan", "scans", "study", "studies", "records", "record", "report", "reports", "profile", "and", "of", "is", "was", "for"}


@dataclass
class Filters:
    patient_ids: list[int] = field(default_factory=list)
    patient_names: list[str] = field(default_factory=list)
    unresolved_patient: str | None = None
    modality: str | None = None
    body_part: str | None = None
    priority: str | None = None
    date_from: pd.Timestamp | None = None
    date_to: pd.Timestamp | None = None
    latest: bool = False
    aggregate: bool = False
    abnormal_only: bool = False
    findings: list[str] = field(default_factory=list)  # canonical finding labels
    documents: bool = False  # question is about uploaded documents / notes
    specialty: str | None = None

    def describe(self) -> str:
        parts = []
        if self.patient_ids:
            parts.append(f"patient {', '.join(self.patient_names)} (id {', '.join(map(str, self.patient_ids))})")
        for k in ("modality", "body_part", "priority"):
            if getattr(self, k):
                parts.append(f"{k}={getattr(self, k)}")
        if self.date_from is not None or self.date_to is not None:
            a = self.date_from.date() if self.date_from is not None else "..."
            b = self.date_to.date() if self.date_to is not None else "..."
            parts.append(f"dates {a} to {b}")
        if self.documents:
            parts.append("documents" + (f" ({self.specialty})" if self.specialty else ""))
        if self.findings:
            parts.append(f"finding={', '.join(self.findings)}")
        for k in ("latest", "aggregate", "abnormal_only"):
            if getattr(self, k):
                parts.append(k)
        return ", ".join(parts) or "none"


class QueryParser:
    def __init__(self, meta: pd.DataFrame):
        people = meta[["patient_id", "patient_name", "mrn"]].drop_duplicates("patient_id")
        self.by_mrn = dict(zip(people["mrn"].str.upper(), people["patient_id"]))
        self.ids = set(people["patient_id"])
        self.name_of = dict(zip(people["patient_id"], people["patient_name"]))
        self.by_name: dict[str, list[int]] = {}
        for pid, name in zip(people["patient_id"], people["patient_name"]):
            self.by_name.setdefault(name.lower(), []).append(int(pid))
        self.names = list(self.by_name)
        acc = meta.dropna(subset=["study_id"]).copy()
        acc["acc"] = acc["text"].str.extract(r"Accession: (\S+)", expand=False)
        self.by_accession = dict(zip(acc["acc"].str.upper(), acc["patient_id"]))
        spec = meta["specialty"].dropna().unique() if "specialty" in meta else []
        self.specialties = sorted((x for x in spec if len(x) > 3 and x.lower() not in ("radiology",)), key=len, reverse=True)

    # ---------------- patient ----------------
    def _patients(self, q: str, f: Filters) -> None:
        ids: list[int] = []
        for mrn in re.findall(r"\b(MRN\d{4,}|LEG\d{4,})\b", q, re.I):
            if mrn.upper() in self.by_mrn:
                ids.append(int(self.by_mrn[mrn.upper()]))
            else:
                f.unresolved_patient = mrn
        for acc in re.findall(r"\b(ACC\d{4,})\b", q, re.I):
            if acc.upper() in self.by_accession:
                ids.append(int(self.by_accession[acc.upper()]))
        for num in re.findall(r"\bpatient\s*(?:id|#|no\.?|number)?\s*[:#]?\s*(\d+)\b", q, re.I):
            if int(num) in self.ids:
                ids.append(int(num))
            else:
                f.unresolved_patient = f"patient id {num}"

        # Names: try every 2-3 word window against the patient name list.
        words = re.findall(r"[A-Za-z][A-Za-z'.-]+", q)
        best: tuple[float, str] | None = None
        for n in (3, 2):
            for i in range(len(words) - n + 1):
                cand = " ".join(words[i:i + n]).lower()
                if cand in self.by_name:
                    best = (100.0, cand)
                    break
                hit = process.extractOne(cand, self.names, scorer=fuzz.ratio, score_cutoff=90)
                if hit and (best is None or hit[1] > best[0]):
                    best = (hit[1], hit[0])
            if best and best[0] == 100.0:
                break
        if best:
            ids.extend(self.by_name[best[1]])
        elif not ids:
            for m in re.finditer(r"\b(?:patient|for|of|about)\s+([A-Za-z][a-z'.-]+(?:\s+[A-Za-z][a-z'.-]+){1,2})", q, re.I):
                cand = [w.strip(".'-") for w in m.group(1).split() if w.lower().strip(".") not in STOP_AFTER_PATIENT]
                if len(cand) >= 2 and all(w[0].isupper() for w in cand):
                    f.unresolved_patient = " ".join(cand)
                    break

        f.patient_ids = sorted(set(ids))
        f.patient_names = [self.name_of[i] for i in f.patient_ids]
        if f.patient_ids:
            f.unresolved_patient = None

    # ---------------- everything else ----------------
    def parse(self, q: str, now: datetime | None = None) -> Filters:
        now = now or datetime.now()
        f = Filters()
        ql = q.lower()
        self._patients(q, f)

        for syn in sorted(C.MODALITY_SYNONYMS, key=len, reverse=True):
            if re.search(rf"(?<![a-z]){re.escape(syn)}(?![a-z])", ql):
                if syn in ("us", "mr", "mg", "pet") and not re.search(rf"\b{syn}\b", q):  # 'us' as a word is ambiguous
                    continue
                f.modality = C.MODALITY_SYNONYMS[syn]
                break
        for bp in BODY_PARTS:
            if re.search(rf"\b{bp}\b", ql):
                f.body_part = bp
                break
        if not f.body_part:
            for syn, bp in BODY_SYNONYMS.items():
                if re.search(rf"\b{syn}\b", ql):
                    f.body_part = bp
                    break
        if m := re.search(r"\b(stat|urgent|emergency|routine)\b", ql) or re.search(r"\b(critical) priority\b", ql):
            f.priority = C.PRIORITY_SYNONYMS[m.group(1)]

        if m := re.search(r"\b(?:last|past|previous)\s+(\d+|a|one|two|three|six)\s+(day|week|month|year)s?\b", ql):
            n = {"a": 1, "one": 1, "two": 2, "three": 3, "six": 6}.get(m.group(1), None) or int(m.group(1))
            days = {"day": 1, "week": 7, "month": 30.4, "year": 365.25}[m.group(2)] * n
            f.date_from = pd.Timestamp(now - timedelta(days=days))
        elif m := re.search(r"\b(?:since|after|from)\s+(20\d\d)\b", ql):
            f.date_from = pd.Timestamp(f"{m.group(1)}-01-01")
        elif m := re.search(r"\bbetween\s+(20\d\d)\s+and\s+(20\d\d)\b", ql):
            f.date_from, f.date_to = pd.Timestamp(f"{m.group(1)}-01-01"), pd.Timestamp(f"{m.group(2)}-12-31 23:59")
        elif m := re.search(r"\b(?:in|during)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d\d)\b", ql):
            f.date_from = pd.Timestamp(f"{m.group(1)} 1 {m.group(2)}")
            f.date_to = f.date_from + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59)
        elif m := re.search(r"\b(?:in|during|of)\s+(20\d\d)\b", ql):
            f.date_from, f.date_to = pd.Timestamp(f"{m.group(1)}-01-01"), pd.Timestamp(f"{m.group(1)}-12-31 23:59")
        if m := re.search(r"\b(?:on|dated)\s+(20\d\d-\d\d-\d\d)\b", ql):
            f.date_from = pd.Timestamp(m.group(1))
            f.date_to = f.date_from + pd.Timedelta(hours=23, minutes=59)

        f.documents = bool(re.search(r"\b(document|documents|uploaded|note|notes|transcription|letter|discharge summary|consult\w*|operative report|x-ray report)\b", ql))
        for sp in self.specialties:
            if re.search(rf"(?<![a-z]){re.escape(sp.lower())}(?![a-z])", ql):
                f.specialty, f.documents = sp, True
                break
        f.latest = bool(re.search(r"\b(latest|most recent|newest|last (study|scan|report|imaging|visit))\b", ql))
        f.aggregate = bool(re.search(r"\b(how many|count|number of|total|percentage|proportion|statistics|most common|distribution|breakdown)\b", ql))
        f.findings = [label for _, label in detect_findings(q)]
        f.abnormal_only = bool(re.search(r"\b(abnormal|positive|patholog\w*|significant findings?)\b", ql))
        return f
