"""Generate a large, realistic synthetic radiology dataset (with optional dirty records).

Every patient has demographics, chronic conditions and a long imaging history; every study has
an AI model prediction and a structured radiology report (technique, comparison, findings,
impression, recommendation). Follow-up studies reference earlier ones so the history is coherent.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from medrag.data import catalog as C
from medrag.data.store import COLUMNS, coerce_types

NOW = datetime(2026, 10, 1)


def _name_pool(rng: np.random.Generator, n: int) -> list[str]:
    firsts, lasts = set(), set()
    for locale in ("en_IN", "en_US", "en_GB"):
        f = Faker(locale)
        f.seed_instance(int(rng.integers(1 << 30)))
        for _ in range(400):
            firsts.add(f.first_name())
            lasts.add(f.last_name())
    firsts, lasts = sorted(firsts), sorted(lasts)
    return [f"{firsts[i]} {lasts[j]}" for i, j in zip(rng.integers(len(firsts), size=n), rng.integers(len(lasts), size=n))]


def _fill(template: str, rng: np.random.Generator) -> str:
    lvl = int(rng.integers(1, 5))
    return template.format(
        side=rng.choice(["right", "left"]), size=int(rng.integers(4, 28)), big=round(float(rng.uniform(1.2, 6.5)), 1),
        seg=int(rng.integers(2, 9)), lvl=lvl, lvl1=lvl + 1, bone=rng.choice(C.BONES), clock=int(rng.integers(1, 13)),
        suv=round(float(rng.uniform(4, 18)), 1),
    )


def _report(rng, modality, body_part, finding, comparison) -> tuple[str, str]:
    """Return (report_text, label)."""
    technique = C.TECHNIQUES[(modality, body_part)]
    if finding is None:
        findings, impression, rec, label = C.NORMAL_FINDINGS[body_part], "No acute abnormality.", "No follow-up imaging required.", "normal"
    else:
        label, _, f_t, i_t, rec, _ = finding
        seed = int(rng.integers(1 << 30))  # same random fill for findings and impression
        findings = _fill(f_t, np.random.default_rng(seed))
        impression = _fill(i_t, np.random.default_rng(seed))
    text = (
        f"TECHNIQUE: {technique}\n"
        f"COMPARISON: {comparison or 'None available.'}\n"
        f"FINDINGS: {findings}\n"
        f"IMPRESSION: {impression}\n"
        f"RECOMMENDATION: {rec}"
    )
    return text, label


def generate(n_patients: int = 20_000, mean_studies: float = 5.0, dirty_rate: float = 0.03, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = _name_pool(rng, n_patients)
    radiologists = [f"Dr. {n}" for n in _name_pool(rng, 60)]
    physicians = [f"Dr. {n}" for n in _name_pool(rng, 300)]
    cities = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata", "Mysuru", "London", "New York"]

    modalities = list(C.MODALITY_WEIGHTS)
    mod_p = np.array(list(C.MODALITY_WEIGHTS.values()), float)
    mod_p /= mod_p.sum()
    by_part: dict[str, list] = {}
    for f in C.FINDINGS:
        for bp in f[1]:
            by_part.setdefault(bp, []).append(f)

    patients, conditions, studies, analyses, reports = [], [], [], [], []
    study_id = analysis_id = report_id = 0

    for pid in tqdm(range(1, n_patients + 1), desc="patients", unit="pt"):
        gender = rng.choice(["male", "female"], p=[0.49, 0.51])
        dob = NOW - timedelta(days=int(rng.integers(18 * 365, 92 * 365)))
        age = (NOW - dob).days // 365
        patients.append((pid, f"MRN{pid:07d}", names[pid - 1], gender, dob.date(), rng.choice(C.BLOOD_GROUPS), rng.choice(cities)))

        n_cond = int(rng.poisson(0.4 + age / 40))
        for cond in rng.choice(C.CHRONIC_CONDITIONS, size=min(n_cond, 6), replace=False):
            conditions.append((pid, str(cond), (NOW - timedelta(days=int(rng.integers(30, 365 * 15)))).date()))

        # Heavy-tailed history: most patients have a few studies, some have dozens.
        n_studies = max(1, int(rng.gamma(1.2, mean_studies / 1.2)))
        dates = sorted(NOW - timedelta(days=int(d), minutes=int(rng.integers(0, 1440))) for d in rng.integers(1, 6 * 365, size=n_studies))
        last_abnormal: dict[tuple, tuple] = {}

        for when in dates:
            study_id += 1
            modality = rng.choice(modalities, p=mod_p)
            if modality == "MG" and gender == "male":
                modality = "XR"
            parts = C.MODALITY_BODY_PARTS[modality]
            if gender == "male":
                parts = {k: v for k, v in parts.items() if k != "pelvis"}
            body_part = rng.choice(list(parts), p=np.array(list(parts.values())) / sum(parts.values()))

            prior = last_abnormal.get((modality, body_part))
            comparison, finding = None, None
            if prior and rng.random() < 0.7:  # follow-up of a known abnormality
                finding = prior[1]
                trend = rng.choice(["stable", "improved", "worse"], p=[0.5, 0.3, 0.2])
                comparison = C.PROGRESSION[trend].format(prior=prior[0])
            elif (options := [f for f in by_part[body_part] if modality in C.FINDING_MODALITIES.get(f[0], {modality})]) \
                    and rng.random() < 0.45 + min(age, 90) / 400:
                finding = options[int(rng.integers(len(options)))]
            if finding is not None:
                last_abnormal[(modality, body_part)] = (when.strftime("%Y-%m-%d"), finding)

            severity = finding[5] if finding else 0
            priority = {0: "routine", 1: "routine", 2: rng.choice(["routine", "urgent"]), 3: rng.choice(["urgent", "stat"], p=[0.4, 0.6])}[severity]
            indication = rng.choice(C.INDICATIONS[body_part])
            studies.append((study_id, pid, f"ACC{study_id:09d}", when, modality, body_part, indication, priority, rng.choice(physicians)))

            text, label = _report(rng, modality, body_part, finding, comparison)
            report_id += 1
            status = rng.choice(C.REPORT_STATUSES, p=[0.88, 0.07, 0.05])
            reports.append((report_id, study_id, rng.choice(radiologists), text, status, when + timedelta(hours=int(rng.integers(1, 72)))))

            analysis_id += 1
            correct = rng.random() < 0.86
            pred = label if correct else rng.choice([f[0] for f in by_part[body_part]] + ["normal"])
            conf = float(np.clip(rng.normal(0.88 if correct else 0.62, 0.08), 0.05, 0.999))
            analyses.append((analysis_id, study_id, C.AI_MODELS[modality], pred, round(conf, 3)))

    cols = {k: list(v) for k, v in COLUMNS.items()}
    tables = {
        "patients": pd.DataFrame(patients, columns=cols["patients"]),
        "conditions": pd.DataFrame(conditions, columns=cols["conditions"]),
        "studies": pd.DataFrame(studies, columns=cols["studies"]),
        "ai_analysis": pd.DataFrame(analyses, columns=cols["ai_analysis"]),
        "reports": pd.DataFrame(reports, columns=cols["reports"]),
    }
    tables["studies"]["modality"] = tables["studies"]["modality"].map(C.MODALITY_NAMES).where(rng.random(len(studies)) < 0.5, tables["studies"]["modality"])
    if dirty_rate > 0:
        tables = make_dirty(tables, dirty_rate, rng)
    return {name: coerce_types(name, df) for name, df in tables.items()}


def make_dirty(t: dict[str, pd.DataFrame], rate: float, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Inject the kinds of problems real hospital exports have, so the cleaning step has work to do."""
    def pick(df, frac=1.0):
        return rng.random(len(df)) < rate * frac

    p = t["patients"].copy()
    m = pick(p)
    p.loc[m, "full_name"] = ["  " + n.upper() + " " if i % 2 else n.lower() for i, n in enumerate(p.loc[m, "full_name"])]
    p.loc[pick(p, 0.3), "gender"] = None
    m = pick(p)
    p.loc[m, "gender"] = rng.choice(["M", "F", "Male", "FEMALE", " female", "unknown"], size=int(m.sum()))
    p.loc[pick(p, 0.3), "date_of_birth"] = None
    m = pick(p, 0.1)
    p.loc[m, "date_of_birth"] = (NOW + timedelta(days=400)).date()  # impossible future DOB
    dup = p[pick(p, 0.5)].copy()  # same person registered twice under a new id
    dup["patient_id"] = np.arange(len(p) + 1, len(p) + 1 + len(dup))
    tests = pd.DataFrame({"patient_id": np.arange(len(p) + len(dup) + 1, len(p) + len(dup) + 6),
                          "mrn": [f"TEST{i}" for i in range(5)], "full_name": ["Test Patient", "Test Check", "DUMMY", "test user", "Do Not Use"],
                          "gender": "male", "date_of_birth": None, "blood_group": None, "city": None})
    t["patients"] = pd.concat([p, dup, tests], ignore_index=True)

    s = t["studies"].copy()
    m = pick(s)
    variants = {"CT": ["CT scan", "ct", "Cat Scan"], "MRI": ["mri ", "MR", "Magnetic Resonance"], "XR": ["Xray", "x-ray", "CXR"],
                "US": ["USG", "ultrasound", "Sonography"], "MG": ["mammo", "Mammogram"], "PET": ["pet-ct", "PET scan"]}
    canon = s.loc[m, "modality"].map(lambda v: C.MODALITY_SYNONYMS.get(str(v).lower(), v))
    s.loc[m, "modality"] = [rng.choice(variants[c]) for c in canon]  # misspelt, but still the true modality
    m = pick(s)
    s.loc[m, "priority"] = rng.choice(["HIGH", "Stat", "low", "Critical", "medium", ""], size=int(m.sum()))
    s.loc[pick(s, 0.2), "study_date"] = NOW + timedelta(days=900)  # future date
    s.loc[pick(s, 0.2), "study_date"] = None
    s.loc[pick(s, 0.2), "patient_id"] = 10_000_000  # orphan: patient does not exist
    # Studies of duplicate patients point at the duplicate id for some rows.
    remap = dict(zip(p.loc[p.index.isin(dup.index), "patient_id"], dup["patient_id"]))
    m = s["patient_id"].isin(remap) & (rng.random(len(s)) < 0.5)
    s.loc[m, "patient_id"] = s.loc[m, "patient_id"].map(remap)
    t["studies"] = pd.concat([s, s[pick(s, 0.5)]], ignore_index=True)  # exact duplicate rows

    r = t["reports"].copy()
    r.loc[pick(r, 0.3), "report_text"] = ""
    m = pick(r)
    r.loc[m, "report_text"] = "<p>" + r.loc[m, "report_text"].str.replace("\n", "<br>  ") + "</p>"
    m = pick(r)
    r.loc[m, "status"] = rng.choice(["approved", "draft", "FINAL", "signed", "pending"], size=int(m.sum()))
    t["reports"] = pd.concat([r, r[pick(r, 0.5)].assign(report_id=lambda d: d["report_id"] + 10_000_000)], ignore_index=True)

    a = t["ai_analysis"].copy()
    m = pick(a)
    a.loc[m, "confidence"] = (a.loc[m, "confidence"] * 100).round(1)  # percent instead of fraction
    a.loc[pick(a, 0.3), "confidence"] = -1.0
    a.loc[pick(a, 0.3), "confidence"] = None
    t["ai_analysis"] = a

    # Last, so the random draws above (and every dataset generated before this rule) stay the same.
    r = t["reports"]
    m = pick(r) & (r["report_text"] != "")
    r.loc[m, "report_text"] = r.loc[m, "report_text"].str.replace(" ", "​ ", n=3).str.replace(":", ": ", n=1) + "‮"
    return t
