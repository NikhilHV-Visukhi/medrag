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
        n = int(((studies["study_date"].dt.year == y) & (studies["modality"] == mod) & (studies["priority"] == pr)).sum())
        items.append({"type": "count_cohort", "question": f"How many {pr} priority {C.MODALITY_NAMES[mod]} studies were performed in {y}?",
                      "gold_docs": [], "facts": [], "number": n, "answer": str(n)})

    fake = ["Zebulon Quixote", "Marigold Featherstone", "Thaddeus Wobblebottom", "Xiomara Pemberton-Lake", "Ignatius Fernsby"]
    for name in fake[: max(2, per_type // 4)]:
        items.append({"type": "not_found", "question": f"Summarise the imaging history of patient {name}.",
                      "gold_docs": [], "facts": [], "abstain": True, "answer": "not found"})
    for i, it in enumerate(items):
        it["id"] = i
    return items


def save(items: list[dict], path) -> None:
    path.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")


def load(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
