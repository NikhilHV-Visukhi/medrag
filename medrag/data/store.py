"""Read/write the dataset tables as DataFrames, backed by PostgreSQL or parquet files.

Two stages:
  raw   -> as generated/imported, may be dirty   (postgres schema medrag_raw)
  clean -> after cleaning, used for indexing     (postgres schema medrag)
"""
import io
from contextlib import contextmanager

import pandas as pd

from medrag.config import get_settings

SCHEMAS = {"raw": "medrag_raw", "clean": "medrag"}

# Raw tables keep everything as loose types (text) so dirty values survive the load.
COLUMNS = {
    "patients": {
        "patient_id": "bigint", "mrn": "text", "full_name": "text", "gender": "text",
        "date_of_birth": "date", "blood_group": "text", "city": "text",
    },
    "conditions": {"patient_id": "bigint", "condition": "text", "diagnosed_on": "date"},
    "studies": {
        "study_id": "bigint", "patient_id": "bigint", "accession": "text",
        "study_date": "timestamp", "modality": "text", "body_part": "text",
        "indication": "text", "priority": "text", "referring_physician": "text",
    },
    "ai_analysis": {
        "analysis_id": "bigint", "study_id": "bigint", "model_name": "text",
        "predicted_label": "text", "confidence": "double precision",
    },
    "reports": {
        "report_id": "bigint", "study_id": "bigint", "radiologist": "text",
        "report_text": "text", "status": "text", "signed_at": "timestamp",
    },
    "documents": {
        "document_id": "bigint", "patient_id": "bigint", "source": "text", "source_id": "text", "title": "text",
        "specialty": "text", "doc_date": "timestamp", "text": "text", "labels": "text", "description": "text",
    },
}
CORE_TABLES = ["patients", "conditions", "studies", "ai_analysis", "reports"]  # documents are optional
DATE_COLS = {"date_of_birth", "diagnosed_on", "study_date", "signed_at", "doc_date"}


@contextmanager
def connect():
    import psycopg

    s = get_settings()
    conn = psycopg.connect(
        host=s.db_host, port=s.db_port, dbname=s.db_name, user=s.db_user,
        password=s.db_password, connect_timeout=10,
    )
    try:
        yield conn
    finally:
        conn.close()


def coerce_types(name: str, df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS[name]:
        if col in DATE_COLS and col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def write_table(stage: str, name: str, df: pd.DataFrame) -> None:
    df = df[list(COLUMNS[name])].copy()
    for col, typ in COLUMNS[name].items():
        if typ == "bigint":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")  # no "12.0" in CSV
    s = get_settings()
    if s.data_backend == "parquet":
        out = s.data_dir / stage
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / f"{name}.parquet", index=False)
        return

    schema = SCHEMAS[stage]
    cols = ", ".join(f"{c} {t}" for c, t in COLUMNS[name].items())
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        cur.execute(f"DROP TABLE IF EXISTS {schema}.{name}")
        cur.execute(f"CREATE TABLE {schema}.{name} ({cols})")
        copy_sql = f"COPY {schema}.{name} ({', '.join(df.columns)}) FROM STDIN WITH (FORMAT csv)"
        with cur.copy(copy_sql) as copy:
            for start in range(0, len(df), 50_000):
                buf = io.StringIO()
                df.iloc[start:start + 50_000].to_csv(buf, index=False, header=False)
                copy.write(buf.getvalue())
        if stage == "clean":
            key = next(iter(COLUMNS[name]))
            if key.endswith("_id") and name != "conditions":
                cur.execute(f"ALTER TABLE {schema}.{name} ADD PRIMARY KEY ({key})")
            for fk in ("patient_id", "study_id"):
                if fk in df.columns and fk != key:
                    cur.execute(f"CREATE INDEX ON {schema}.{name} ({fk})")
        conn.commit()


def read_table(stage: str, name: str) -> pd.DataFrame:
    s = get_settings()
    if s.data_backend == "parquet":
        return coerce_types(name, pd.read_parquet(s.data_dir / stage / f"{name}.parquet"))
    with connect() as conn, conn.cursor() as cur:
        buf = io.StringIO()
        with cur.copy(f"COPY {SCHEMAS[stage]}.{name} TO STDOUT WITH (FORMAT csv, HEADER)") as copy:
            for chunk in copy:
                buf.write(bytes(chunk).decode("utf-8"))
        buf.seek(0)
        return coerce_types(name, pd.read_csv(buf, low_memory=False))


def read_all(stage: str) -> dict[str, pd.DataFrame]:
    """Core tables, plus 'documents' when that stage has them."""
    t = {name: read_table(stage, name) for name in CORE_TABLES}
    try:
        t["documents"] = read_table(stage, "documents")
    except Exception:  # no documents ingested yet (missing file or table)
        pass
    return t


def write_all(stage: str, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        write_table(stage, name, df)


def import_legacy_oads() -> dict[str, pd.DataFrame]:
    """Map the original oads_db tables (oads.*) onto the raw MedRAG layout."""
    with connect() as conn:
        q = lambda sql: pd.read_sql(sql, conn)  # noqa: E731
        patients = q("SELECT patient_id, full_name, gender FROM oads.patients")
        studies = q("""
            SELECT s.study_id, s.patient_id, s.study_date, s.priority,
                   (SELECT i.image_type FROM oads.images i WHERE i.study_id = s.study_id LIMIT 1) AS modality
            FROM oads.studies s""")
        analysis = q("""
            SELECT a.analysis_id, i.study_id, a.findings_summary AS predicted_label, a.confidence_score AS confidence
            FROM oads.analysis a JOIN oads.images i ON i.image_id = a.image_id""")
        reports = q("SELECT report_id, study_id, report_content AS report_text, status FROM oads.reports")

    patients["mrn"] = "LEG" + patients["patient_id"].astype(str).str.zfill(7)
    studies["accession"] = "LEG-ACC-" + studies["study_id"].astype(str)
    analysis["model_name"] = "legacy"
    tables = {
        "patients": patients, "conditions": pd.DataFrame(columns=list(COLUMNS["conditions"])),
        "studies": studies, "ai_analysis": analysis, "reports": reports,
    }
    for name, df in tables.items():
        for col in COLUMNS[name]:
            if col not in df:
                df[col] = None
    return tables
