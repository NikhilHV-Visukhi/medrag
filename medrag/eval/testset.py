"""Build an evaluation set straight from the indexed data, so every question has a known right answer."""
import json
import random
import re

import pandas as pd

from medrag.data import catalog as C

STOP = {"with", "without", "from", "that", "this", "consistent", "findings", "likely", "suspicious", "compared", "study",
        "prior", "there", "which", "highly", "measuring", "noted", "acute", "minimal", "mild", "moderate"}


def keywords(text: str, n: int = 4) -> list[str]:
    words = [w for w in re.findall(r"[a-z][a-z-]{3,}", str(text).lower()) if w not in STOP]
    return list(dict.fromkeys(words))[:n]


def build(meta: pd.DataFrame, per_type: int = 20, seed: int = 11) -> list[dict]:
    rnd = random.Random(seed)
    studies = meta[(meta["doc_type"] == "study") & (meta["part"] == 0)]
    abnormal = studies[~studies["impression"].fillna("").str.contains("No acute abnormality") & studies["impression"].notna()]
    per_patient = studies.groupby("patient_id")
    items: list[dict] = []

    def sample(df, n):
        return df.sample(min(n, len(df)), random_state=rnd.randint(0, 10**6)) if len(df) else df

    names = meta[meta["doc_type"] == "patient"].drop_duplicates("patient_id")["patient_name"]
    unique_name = set(names[~names.duplicated(keep=False)])
    shared_name = sorted(set(names[names.duplicated(keep=False)]))
    # A semantic question must point at exactly one study: same day + modality + finding must be unique.
    day = abnormal["study_date"].dt.strftime("%Y-%m-%d")
    abnormal = abnormal.assign(_key=day + abnormal["modality"] + abnormal["ai_label"].fillna(""))
    unique_case = abnormal[~abnormal["_key"].duplicated(keep=False)]
    abnormal = abnormal[abnormal["patient_name"].isin(unique_name)]
    all_studies = studies  # cohort counts must include every patient, not only those with a unique name
    studies = studies[studies["patient_name"].isin(unique_name)]
    per_patient = studies.groupby("patient_id")

    for x in sample(abnormal, per_type).itertuples():
        items.append({"type": "lookup_by_name", "question": f"What did the {C.MODALITY_NAMES[x.modality]} {x.body_part} on {x.study_date:%Y-%m-%d} show for {x.patient_name}?",
                      "gold_docs": [x.doc_id], "facts": keywords(x.impression), "answer": x.impression})
    for x in sample(abnormal, per_type).itertuples():
        items.append({"type": "lookup_by_mrn", "question": f"Summarise the {C.MODALITY_NAMES[x.modality]} report for patient {x.mrn} dated {x.study_date:%Y-%m-%d}.",
                      "gold_docs": [x.doc_id], "facts": keywords(x.impression), "answer": x.impression})

    many = [pid for pid, g in per_patient if len(g) >= 3]
    for pid in rnd.sample(many, min(per_type, len(many))):
        g = per_patient.get_group(pid).sort_values("study_date")
        last = g.iloc[-1]
        items.append({"type": "latest", "question": f"What was the most recent imaging study for {last.patient_name} and what did it show?",
                      "gold_docs": [last.doc_id], "facts": [f"{last.study_date:%Y-%m-%d}"] + keywords(last.impression, 3), "answer": last.impression})
    for pid in rnd.sample(many, min(per_type, len(many))):
        g = per_patient.get_group(pid)
        mod = g["modality"].value_counts().index[0]
        n = int((g["modality"] == mod).sum())
        name = g.iloc[0].patient_name
        items.append({"type": "count_patient", "question": f"How many {C.MODALITY_NAMES[mod]} studies does {name} have?",
                      "gold_docs": [f"patient:{pid}"], "facts": [], "number": n, "answer": str(n)})
    for name in rnd.sample(shared_name, min(per_type // 2 or 1, len(shared_name))):
        mrns = sorted(meta.loc[(meta["doc_type"] == "patient") & (meta["patient_name"] == name), "mrn"].unique())
        items.append({"type": "ambiguous_name", "question": f"How many imaging studies does {name} have?",
                      "gold_docs": [], "facts": [m.lower() for m in mrns], "answer": f"{len(mrns)} different patients: {', '.join(mrns)}"})
    for x in sample(unique_case[unique_case["patient_name"].isin(unique_name)], per_type).itertuples():
        label = x.ai_label if x.ai_label and x.ai_label != "normal" else keywords(x.impression, 2)[-1]
        items.append({"type": "semantic", "question": f"Which patient had a {C.MODALITY_NAMES[x.modality]} on {x.study_date:%Y-%m-%d} showing {label}?",
                      "gold_docs": [x.doc_id], "facts": [x.patient_name.split()[0].lower()], "answer": x.patient_name})

    years = sorted(studies["study_date"].dt.year.unique())
    for _ in range(min(per_type // 2 or 1, 10)):
        y, mod, pr = rnd.choice(years), rnd.choice(C.MODALITIES), rnd.choice(["stat", "urgent"])
        n = int(((all_studies["study_date"].dt.year == y) & (all_studies["modality"] == mod) & (all_studies["priority"] == pr)).sum())
        items.append({"type": "count_cohort", "question": f"How many {pr} priority {C.MODALITY_NAMES[mod]} studies were performed in {y}?",
                      "gold_docs": [], "facts": [], "number": n, "answer": str(n)})

    fake = ["Zebulon Quixote", "Marigold Featherstone", "Thaddeus Wobblebottom", "Xiomara Pemberton-Lake", "Ignatius Fernsby"]
    for name in fake[: max(2, per_type // 4)]:
        items.append({"type": "not_found", "question": f"Summarise the imaging history of patient {name}.",
                      "gold_docs": [], "facts": [], "abstain": True, "answer": "not found"})
    for it in items:
        it["difficulty"] = "easy"
    items += build_hard(meta, per_type, seed)
    items += build_documents(meta, per_type, seed)
    for i, it in enumerate(items):
        it["id"] = i
    return items


# Lay / alternative wording used in hard questions. Some phrases are deliberately NOT in the expansion
# dictionary (marked by in_dictionary=False on each item) so the effect of expansion is measured honestly.
TEST_PHRASES = {
    "pneumonia": ["a chest infection", "a lung infection"],
    "pulmonary nodule": ["a spot on the lung", "a small lung nodule"],
    "lung mass": ["a lung tumour", "a suspected lung cancer"],
    "pleural effusion": ["fluid around the lung", "water on the lung"],
    "pneumothorax": ["a collapsed lung"],
    "cardiomegaly": ["an enlarged heart", "a big heart"],
    "pulmonary embolism": ["a blood clot in the lung", "a clot in the pulmonary arteries"],
    "hepatic steatosis": ["fatty liver", "fat in the liver"],
    "liver lesion": ["a suspected liver cancer", "a liver tumour"],
    "gallstones": ["gall stones", "stones in the gallbladder"],
    "renal calculus": ["a kidney stone", "a stone in the kidney"],
    "intracranial haemorrhage": ["a brain bleed", "bleeding inside the skull"],
    "ischaemic stroke": ["a stroke", "a cerebral infarct"],
    "disc herniation": ["a slipped disc", "a herniated disk"],
    "compression fracture": ["a collapsed vertebra", "a crushed vertebra"],
    "meniscal tear": ["a torn meniscus", "a knee cartilage tear"],
    "ACL tear": ["a torn ACL", "a ruptured cruciate ligament"],
    "thyroid nodule": ["a thyroid lump", "a lump in the neck gland"],
    "uterine fibroid": ["fibroids", "a benign uterine growth"],
    "breast mass": ["a breast lump", "a suspicious lump in the breast"],
    "metastatic disease": ["cancer spread", "secondary tumours"],
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def build_hard(meta: pd.DataFrame, per_type: int = 20, seed: int = 11) -> list[dict]:
    from medrag.retrieval.expansion import detect_findings, impression_pattern

    rnd = random.Random(seed + 1)
    studies = meta[(meta["doc_type"] == "study") & (meta["part"] == 0)].copy()
    names = meta[meta["doc_type"] == "patient"].drop_duplicates("patient_id")["patient_name"]
    unique_name = set(names[~names.duplicated(keep=False)])
    imp = studies["impression"].fillna("")
    label_of = pd.Series(None, index=studies.index, dtype=object)
    for label in TEST_PHRASES:
        label_of[imp.str.contains(impression_pattern(label), case=False, regex=True) & label_of.isna()] = label
    studies["finding"] = label_of
    found = studies[studies["finding"].notna()]
    items: list[dict] = []

    def phrase(label):
        p = rnd.choice(TEST_PHRASES[label])
        return p, any(lbl == label for _, lbl in detect_findings(p))

    def add(kind, q, gold, label=None, in_dict=None, **kw):
        items.append({"type": kind, "difficulty": "hard", "question": q, "gold_docs": gold, "label": label,
                      "in_dictionary": in_dict, **kw})

    # 1. paraphrased finding inside a long patient history, no date or modality given
    cand = found[found["patient_name"].isin(unique_name)]
    sizes = studies.groupby("patient_id").size()
    cand = cand[cand["patient_id"].map(sizes) >= 6]
    cand = cand[~cand.duplicated(["patient_id", "finding"], keep=False)]
    for x in cand.sample(min(per_type, len(cand)), random_state=rnd.randint(0, 10**6)).itertuples():
        p, d = phrase(x.finding)
        add("hard_paraphrase_patient", f"When did {x.patient_name} have imaging that showed {p}, and what kind of scan was it?",
            [x.doc_id], x.finding, d, facts=[f"{x.study_date:%Y-%m-%d}", C.MODALITY_NAMES[x.modality].lower() + "|" + x.modality.lower()],
            answer=f"{x.study_date:%Y-%m-%d} {C.MODALITY_NAMES[x.modality]} {x.body_part}: {x.impression}")

    # 2. cohort question by paraphrased finding + modality + month, 1-3 right answers
    found_m = found.assign(month=found["study_date"].dt.to_period("M"))
    groups = [g for _, g in found_m.groupby(["finding", "modality", "month"]) if 1 <= len(g) <= 3]
    for g in rnd.sample(groups, min(per_type, len(groups))):
        x = g.iloc[0]
        p, d = phrase(x.finding)
        add("hard_semantic_cohort", f"Which patients had {p} on a {C.MODALITY_NAMES[x.modality]} scan in {MONTHS[x.month.month - 1]} {x.month.year}?",
            g["doc_id"].tolist(), x.finding, d, facts=[n.split()[-1].lower() for n in g["patient_name"]],
            answer="; ".join(f"{r.patient_name} ({r.study_date:%Y-%m-%d})" for r in g.itertuples()))

    # 3. follow-up trend: needs the comparison line of a later study
    trend = studies[studies["text"].str.contains(r"COMPARISON: (?:Interval|Unchanged)", regex=True) & studies["finding"].notna()
                    & studies["patient_name"].isin(unique_name)]
    trend = trend.assign(trend=trend["text"].str.extract(r"COMPARISON: (Interval improvement|Interval progression|Unchanged)", expand=False))
    word = {"Interval improvement": "improv|better", "Interval progression": "progress|worse", "Unchanged": "unchanged|stable|same"}
    for x in trend.sample(min(per_type, len(trend)), random_state=rnd.randint(0, 10**6)).itertuples():
        p, d = phrase(x.finding)
        add("hard_trend", f"On the follow-up scan of {x.study_date:%Y-%m-%d}, had {x.patient_name}'s {p.split(' ', 1)[-1] if p.startswith(('a ', 'an ')) else p} improved, worsened or stayed the same?",
            [x.doc_id], x.finding, d, facts=[word[x.trend]], answer=x.trend)

    # 4. exact cohort count by paraphrased finding and year
    found_y = found.assign(year=found["study_date"].dt.year)
    for _ in range(min(per_type // 2 or 1, 10)):
        label, year = rnd.choice(sorted(TEST_PHRASES)), rnd.choice(sorted(found_y["year"].unique()))
        n = int(found_y[(found_y["finding"] == label) & (found_y["year"] == year)]["patient_id"].nunique())
        p, d = phrase(label)
        add("hard_count_finding", f"How many patients had {p} reported on imaging in {year}?", [], label, d, facts=[], number=n, answer=str(n))

    # 5. negative: the patient never had this finding
    many = [pid for pid, c in sizes.items() if c >= 4]
    rnd.shuffle(many)
    for pid in many[: per_type * 3]:
        if len([i for i in items if i["type"] == "hard_negative"]) >= per_type:
            break
        g = studies[studies["patient_id"] == pid]
        if g.iloc[0].patient_name not in unique_name:
            continue
        label = rnd.choice(sorted(TEST_PHRASES))
        if (g["finding"] == label).any():
            continue
        p, d = phrase(label)
        add("hard_negative", f"Has {g.iloc[0].patient_name} ever had {p} on any scan?", [], label, d, facts=[], expect_no=True,
            answer=f"No: none of {len(g)} studies report {label}")

    # 6. misspelt patient name
    cand = found[found["patient_name"].isin(unique_name)]
    for x in cand.sample(min(per_type, len(cand)), random_state=rnd.randint(0, 10**6)).itertuples():
        first, last = x.patient_name.split(" ", 1)
        if len(last) < 5:
            continue
        i = rnd.randint(1, len(last) - 2)
        typo = f"{first} {last[:i] + last[i + 1:]}"
        add("hard_typo_name", f"What did the {C.MODALITY_NAMES[x.modality]} on {x.study_date:%Y-%m-%d} show for {typo}?",
            [x.doc_id], x.finding, None, facts=keywords(x.impression), answer=x.impression)
    return items


def build_documents(meta: pd.DataFrame, per_type: int = 20, seed: int = 11) -> list[dict]:
    """Questions about uploaded outside documents (Open-i reports, MTSamples transcriptions)."""
    if "source" not in meta or not (meta["doc_type"] == "document").any():
        return []
    rnd = random.Random(seed + 2)
    docs = meta[(meta["doc_type"] == "document") & (meta["part"] == 0)]
    names = meta[meta["doc_type"] == "patient"].drop_duplicates("patient_id")["patient_name"]
    unique_name = set(names[~names.duplicated(keep=False)])
    docs = docs[docs["patient_name"].isin(unique_name) & docs["impression"].notna()]
    items: list[dict] = []

    def sample(df, n):
        return df.sample(min(n, len(df)), random_state=rnd.randint(0, 10**6)) if len(df) else df

    def add(kind, q, gold, facts, answer):
        items.append({"type": kind, "difficulty": "documents", "question": q, "gold_docs": gold, "facts": facts, "answer": answer})

    cxr = docs[docs["source"] == "openi"]
    abnormal = cxr[~cxr["impression"].str.contains(r"no acute|normal|negative|clear", case=False, regex=True)]
    for x in sample(abnormal, per_type).itertuples():
        add("doc_patient_lookup", f"What did {x.patient_name}'s uploaded chest X-ray report from {x.study_date:%Y-%m-%d} conclude?",
            [x.doc_id], keywords(x.impression, 3), x.impression)
    # Find a report by its content alone: a distinctive impression phrase, no name or date.
    imp_counts = cxr["impression"].str.lower().value_counts()
    distinct = abnormal[abnormal["impression"].str.lower().map(imp_counts) == 1]
    distinct = distinct[distinct["impression"].str.split().str.len().between(6, 30)]
    # The keywords must single out one report: a keyword set that also fits other reports (e.g. 116 reports
    # mention "stable silhouette pneumothorax focal") measures luck, not retrieval. Use 5 keywords, or up to
    # 9 if needed; skip impressions that stay ambiguous. "[redacted]" (de-identified text) is never a keyword.
    every_cxr = meta[(meta["doc_type"] == "document") & (meta["source"] == "openi")]
    all_imp = every_cxr.groupby("doc_id")["text"].apply(lambda t: " ".join(t).split("\n", 2)[-1].lower())  # report text, header dropped
    added = 0
    for x in sample(distinct, 3 * per_type).itertuples():
        if added == per_type:
            break
        words = [w for w in keywords(x.impression, 12) if w != "redacted"]
        for n in (5, 7, 9):
            kw = words[:n]
            if len(kw) < min(n, 5) or all_imp.apply(lambda t: all(re.search(r"\b" + re.escape(w), t) for w in kw)).sum() == 1:
                break
        else:
            continue
        if len(kw) < 5:
            continue
        added += 1
        add("doc_semantic", f"Which patient has an uploaded chest X-ray report describing {' '.join(kw)}?",
            [x.doc_id], [x.patient_name.split()[-1].lower()], f"{x.patient_name}: {x.impression}")
    mts = docs[docs["source"] == "mtsamples"]
    for x in sample(mts, per_type).itertuples():
        add("doc_transcription_lookup", f"Summarise {x.patient_name}'s {x.title.strip()} note.",
            [x.doc_id], keywords(x.impression, 3), x.impression)
    desc = meta[(meta["doc_type"] == "document") & (meta["part"] == 0) & (meta["source"] == "mtsamples")]
    desc = desc.assign(summary=desc["text"].str.extract(r"Summary: (.+)", expand=False))
    desc = desc[desc["summary"].str.split().str.len().between(6, 40) & desc["patient_name"].isin(unique_name)]
    for x in sample(desc, per_type).itertuples():
        add("doc_transcription_semantic", f"Which patient had a {x.specialty.lower()} document about: {x.summary.rstrip('.')}?",
            [x.doc_id], [x.patient_name.split()[-1].lower()], x.patient_name)
    return items


def review_sheet(items: list[dict], meta: pd.DataFrame, path, n: int = 30, seed: int = 5) -> None:
    """CSV of hard questions with the gold evidence, for a person to check the expected answers."""
    rnd = random.Random(seed)
    hard = [i for i in items if i.get("difficulty") != "easy"]
    texts = meta.drop_duplicates("doc_id").set_index("doc_id")["text"]
    rows = [{"id": it["id"], "type": it["type"], "question": it["question"], "expected_answer": it["answer"],
             "gold_evidence": "\n---\n".join(texts.get(d, "") for d in it["gold_docs"])[:1500],
             "reviewer_ok (y/n)": "", "reviewer_note": ""} for it in rnd.sample(hard, min(n, len(hard)))]
    pd.DataFrame(rows).to_csv(path, index=False)


def save(items: list[dict], path) -> None:
    path.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")


def load(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
