"""Medical query expansion: map lay terms and clinical synonyms onto the vocabulary used in the reports.

"slipped disc" -> disc herniation (AI label) + "disc extrusion" (radiologist wording). Expanded terms are
appended to the query for BM25, the embedding search and the reranker, and passed to the LLM as a note,
so a question and a report that describe the same finding in different words still meet.
"""
import re

# canonical finding label -> synonyms people use in questions
SYNONYMS = {
    "pneumonia": ["chest infection", "lung infection", "consolidation", "lobar pneumonia"],
    "pulmonary nodule": ["lung nodule", "spot on the lung", "lung spot", "coin lesion"],
    "lung mass": ["lung cancer", "lung tumour", "lung tumor", "lung malignancy", "bronchogenic carcinoma"],
    "pleural effusion": ["fluid around the lung", "fluid in the chest", "fluid on the lung"],
    "pneumothorax": ["collapsed lung", "punctured lung"],
    "cardiomegaly": ["enlarged heart", "big heart", "heart enlargement"],
    "emphysema": ["copd changes", "emphysematous"],
    "pulmonary embolism": ["blood clot in the lung", "lung clot", "clot in the lungs", "pulmonary embolus"],
    "hepatic steatosis": ["fatty liver", "liver fat", "steatosis"],
    "liver lesion": ["liver tumour", "liver tumor", "liver cancer", "hepatocellular carcinoma", "hcc", "liver mass"],
    "gallstones": ["cholelithiasis", "gall stones", "gallbladder stones"],
    "renal calculus": ["kidney stone", "renal stone", "nephrolithiasis"],
    "appendicitis": ["inflamed appendix"],
    "splenomegaly": ["enlarged spleen", "big spleen"],
    "intracranial haemorrhage": ["brain bleed", "bleeding in the brain", "intracranial hemorrhage", "haemorrhagic stroke", "hemorrhagic stroke"],
    "ischaemic stroke": ["ischemic stroke", "cerebral infarct", "brain infarct", "stroke"],
    "brain tumour": ["brain tumor", "brain mass", "glioma", "brain cancer"],
    "small vessel disease": ["white matter disease", "microvascular changes", "white matter changes"],
    "disc herniation": ["slipped disc", "herniated disc", "disc prolapse", "prolapsed disc", "disc bulge"],
    "compression fracture": ["vertebral fracture", "collapsed vertebra", "spinal fracture", "vertebral collapse"],
    "degenerative spondylosis": ["spinal arthritis", "spondylosis", "spine degeneration"],
    "meniscal tear": ["torn meniscus", "meniscus tear", "torn cartilage in the knee"],
    "ACL tear": ["torn acl", "anterior cruciate ligament tear", "acl rupture"],
    "osteoarthritis": ["arthritis", "joint degeneration", "wear and tear"],
    "fracture": ["broken bone", "fractured"],
    "thyroid nodule": ["thyroid lump", "lump in the thyroid"],
    "ovarian cyst": ["cyst on the ovary", "ovary cyst"],
    "uterine fibroid": ["fibroid", "fibroids", "leiomyoma", "myoma"],
    "breast mass": ["breast lump", "breast cancer", "breast tumour", "breast tumor"],
    "breast calcifications": ["microcalcifications", "calcium deposits in the breast"],
    "metastatic disease": ["metastases", "cancer spread", "metastatic cancer", "mets"],
    "lymphoma": ["lymph node cancer", "enlarged lymph nodes"],
}

# How each finding is worded in report impressions (regex). Used for exact cohort counts and expansion terms.
IMPRESSION_PATTERNS = {
    "gallstones": r"cholelithiasis", "intracranial haemorrhage": r"haemorrhage", "ischaemic stroke": r"infarct",
    "brain tumour": r"frontal mass|glioma", "disc herniation": r"disc extrusion", "meniscal tear": r"meniscus .*tear",
    "fracture": r"(?<!compression )fracture", "lung mass": r"lung mass", "liver lesion": r"liver lesion",
}
REPORT_TERMS = {
    "gallstones": "cholelithiasis", "intracranial haemorrhage": "haemorrhage", "ischaemic stroke": "infarct MCA territory",
    "brain tumour": "enhancing frontal mass glioma", "disc herniation": "disc extrusion nerve root compression",
    "meniscal tear": "meniscus posterior horn tear", "liver lesion": "arterially enhancing lesion washout LI-RADS",
    "lung mass": "spiculated mass malignancy", "pulmonary embolism": "filling defects segmental pulmonary arteries",
    "small vessel disease": "white matter hyperintensities", "breast mass": "spiculated mass BI-RADS",
}


def impression_pattern(label: str) -> str:
    return IMPRESSION_PATTERNS.get(label, re.escape(label))


_TERMS = sorted(((syn, label) for label, syns in SYNONYMS.items() for syn in syns + [label]), key=lambda x: -len(x[0]))


def detect_findings(query: str) -> list[tuple[str, str]]:
    """[(phrase found in query, canonical label)], longest phrases first, no overlaps."""
    ql, found, taken = query.lower(), [], []
    for phrase, label in _TERMS:
        for m in re.finditer(rf"(?<![a-z]){re.escape(phrase.lower())}(?![a-z])", ql):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            if label not in [lbl for _, lbl in found]:
                found.append((phrase, label))
    return found


def expand(query: str) -> tuple[str, list[str]]:
    """Return (expanded query, notes for the LLM)."""
    extra, notes = [], []
    for phrase, label in detect_findings(query):
        terms = [label] + ([REPORT_TERMS[label]] if label in REPORT_TERMS else [])
        new = [t for t in terms if t.lower() not in query.lower()]
        if new:
            extra += new
            notes.append(f"'{phrase}' means {label}" + (f" (reports may say: {REPORT_TERMS[label]})" if label in REPORT_TERMS else ""))
    return (f"{query} ({'; '.join(extra)})" if extra else query), notes
