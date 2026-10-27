"""Load outside document collections into one DataFrame layout.

Columns: source, source_id, title, specialty, text, labels, description

- Open-i (Indiana University chest X-ray reports, NLM): 3,955 de-identified reports with MeSH labels.
  https://openi.nlm.nih.gov/  (download: NLMCXR_reports.tgz, licence CC BY-NC-ND 4.0)
- MTSamples (Kaggle "tboyle10/medicaltranscriptions", mtsamples.csv): ~5,000 transcriptions in 40 specialties.
"""
import tarfile
from pathlib import Path

import pandas as pd
from defusedxml import ElementTree as ET

OPENI_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
SECTIONS = ["COMPARISON", "INDICATION", "FINDINGS", "IMPRESSION"]


def load_openi(source: Path) -> pd.DataFrame:
    """source: the NLMCXR_reports.tgz archive or a folder of its XML files."""
    if source.suffix == ".tgz":
        xml_dir = source.parent / "xml"
        if not xml_dir.exists():
            with tarfile.open(source) as t:
                t.extractall(xml_dir, filter="data")  # 'data' filter blocks path traversal / links
        source = xml_dir
    rows = []
    for f in sorted(source.rglob("*.xml"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        root = ET.parse(f).getroot()
        parts = {el.get("Label"): (el.text or "").strip() for el in root.iter("AbstractText")}
        text = "\n".join(f"{s}: {parts[s]}" for s in SECTIONS if parts.get(s))
        labels = [m.text for m in root.iter("major") if m.text]
        uid = root.find("uId")
        rows.append({"source": "openi", "source_id": uid.get("id") if uid is not None else f.stem,
                     "title": "Chest X-ray report", "specialty": "Radiology", "text": text,
                     "labels": "; ".join(labels), "description": parts.get("IMPRESSION", "")})
    return pd.DataFrame(rows)


def load_mtsamples(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df.rename(columns={"transcription": "text", "medical_specialty": "specialty", "sample_name": "title"})
    return pd.DataFrame({
        "source": "mtsamples", "source_id": "MTS" + df.index.astype(str), "title": df["title"].astype(str).str.strip(),
        "specialty": df["specialty"].astype(str).str.strip(), "text": df["text"], "labels": df.get("keywords"),
        "description": df["description"].astype(str).str.strip(),
    })


def load_all(doc_dir: Path, openi: Path | None = None) -> pd.DataFrame:
    frames = []
    openi = openi or next(iter(sorted(doc_dir.parent.glob("source/openi/*.tgz"))), None)
    if openi and openi.exists():
        frames.append(load_openi(openi))
    mts = doc_dir / "mtsamples.csv"
    if mts.exists():
        frames.append(load_mtsamples(mts))
    if not frames:
        raise FileNotFoundError(f"No documents found. Put NLMCXR_reports.tgz in data/source/openi/ ({OPENI_URL}) "
                                f"and/or mtsamples.csv in {doc_dir}")
    return pd.concat(frames, ignore_index=True)
