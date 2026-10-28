# MedRAG: full project report

Everything done in this project, end to end: data collection, quality checks, cleaning, outside documents, chunking, embedding, indexing, retrieval, answering, evaluation (all question types and metrics), every experiment and result, problems hit along the way, fixes, deployment options and known limits.

Repository: https://github.com/nikhilhv414/medrag (branch `main`). Report date: 2026-10-28.

---

## Contents

1. [Goal](#1-goal)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [Environment and configuration](#3-environment-and-configuration)
4. [Data collection](#4-data-collection)
5. [Data quality checks](#5-data-quality-checks)
6. [Data cleaning](#6-data-cleaning)
7. [Outside documents: Open-i and MTSamples](#7-outside-documents-open-i-and-mtsamples)
8. [Storage layer](#8-storage-layer)
9. [Chunking](#9-chunking)
10. [Embeddings](#10-embeddings)
11. [Index (dense + BM25)](#11-index-dense--bm25)
12. [Keeping the index in sync](#12-keeping-the-index-in-sync)
13. [Query understanding](#13-query-understanding)
14. [Retrieval pipeline](#14-retrieval-pipeline)
15. [Answer generation](#15-answer-generation)
16. [LLM providers](#16-llm-providers)
17. [User interface](#17-user-interface)
18. [Evaluation design](#18-evaluation-design)
19. [Evaluation metrics](#19-evaluation-metrics)
20. [All evaluation runs and results](#20-all-evaluation-runs-and-results)
21. [Code audit and fixes](#21-code-audit-and-fixes)
22. [Tests](#22-tests)
23. [Problems hit and how they were solved](#23-problems-hit-and-how-they-were-solved)
24. [Deployment: AWS Bedrock](#24-deployment-aws-bedrock)
25. [Security and data handling](#25-security-and-data-handling)
26. [Known limits and next steps](#26-known-limits-and-next-steps)
27. [Command reference](#27-command-reference)
28. [Git history](#28-git-history)

---

## 1. Goal

A question-answering assistant over a hospital's radiology records that a clinician can ask in plain English:

- "What did the CT chest on 2024-03-02 show for Asha Rao?"
- "How many stat MRI studies were done in 2023?"
- "Has this patient ever had a slipped disc?"
- "Summarise the uploaded discharge note for MRN0001234."

Requirements set during the project:

- Answers must come from the records only, with the sources cited, and say "not found" instead of guessing.
- Counts must be exact (computed from the database, not from whatever a search happened to return).
- It must work fully on a CPU laptop with local models, and be able to switch to cloud models (Anthropic, OpenAI, Gemini, Groq, OpenRouter, AWS Bedrock) by configuration only.
- Every design choice must be measured: a reproducible evaluation with known right answers.

---

## 2. Architecture at a glance

```
 DATA                         PREPARE                           SEARCH + ANSWER
 ----                         -------                           ---------------
 synthetic generator ─┐
 legacy oads_db       ├─> raw stage ─> quality check ─> clean ─> clean stage ─┐
                      ┘                (report)        (log)                  │
 Open-i XML reports ──┐                                                       ├─> chunks (+ context header)
 MTSamples CSV ───────┴─> load ─> check ─> clean ─> link to patient ──────────┘        │
                                                                                       v
                                                         embed (bge-small) + BM25 ─> versioned index
                                                                                       │
 question ─> parse filters ─> filter rows ─> dense + BM25 ─> RRF ─> MedCPT rerank ─> pin profile/latest
                                                                                       │
                                     exact computed facts (counts, timeline) ──────────┤
                                                                                       v
                                                                LLM (local or cloud) ─> answer with [S1] citations
```

Code layout (about 3,300 lines of Python):

| Folder / file | What it does |
|---|---|
| `medrag/config.py` | All settings from `.env` (pydantic-settings) |
| `medrag/__main__.py` | Command line: `generate`, `import-legacy`, `check`, `clean`, `documents`, `index`, `ask`, `watch`, `eval`, `pipeline` |
| `medrag/data/catalog.py` | Clinical vocabulary: modalities, body parts, 32 findings, synonyms, techniques |
| `medrag/data/generate.py` | Synthetic dataset generator, plus deliberate "dirty" records |
| `medrag/data/quality.py` | 27 data quality checks, Markdown + JSON report |
| `medrag/data/clean.py` | Cleaning rules, each one logged with a row count |
| `medrag/data/store.py` | Read/write tables to PostgreSQL or parquet; import of the legacy `oads` schema |
| `medrag/documents/loaders.py` | Open-i XML and MTSamples CSV loaders |
| `medrag/documents/prepare.py` | Document checks, cleaning, linking to patients |
| `medrag/ingest/chunking.py` | Section-aware chunking |
| `medrag/ingest/documents.py` | Turns clean tables into study, patient-profile and document chunks |
| `medrag/providers/embeddings.py` | Embedding providers (sentence-transformers, Ollama, OpenAI, Gemini, hash) |
| `medrag/providers/llms.py` | LLM providers (Ollama, Anthropic, OpenAI, Groq, OpenRouter, Gemini, Bedrock) |
| `medrag/index/store.py` | Versioned hybrid index: FAISS + BM25 + chunk metadata |
| `medrag/retrieval/query_parser.py` | Pulls patient, modality, body part, priority, dates, intent out of a question |
| `medrag/retrieval/expansion.py` | Medical query expansion (lay terms to report wording) |
| `medrag/retrieval/retriever.py` | Filtering, hybrid search, RRF, reranking, pinning |
| `medrag/rag/answer.py` | Computed facts, prompt, answer |
| `medrag/eval/testset.py` | Builds the test set from the data, with known answers |
| `medrag/eval/run.py` | Retrieval metrics, answer checks, LLM judge, bootstrap CIs, report |
| `app.py` | Streamlit UI (Chat, Patients, Data quality, Evaluation tabs) |
| `tests/test_pipeline.py` | 15 end-to-end and unit tests |

---

## 3. Environment and configuration

- Windows 11, CPU only (no GPU), Python 3.14 virtual environment at `D:\LLM\medrag\.venv`.
- Ollama at `http://localhost:11434` for local LLMs.
- Data backend used for this run: **parquet** (`DATA_BACKEND=parquet`), so no database is needed. PostgreSQL is fully supported (`DATA_BACKEND=postgres`, schemas `medrag_raw` and `medrag`).
- Dependencies (`requirements.txt`): numpy, pandas, pyarrow, tabulate, pydantic-settings, python-dotenv, tqdm, faker, rapidfuzz, psycopg, faiss-cpu, bm25s, PyStemmer, sentence-transformers, httpx, openai, anthropic, google-genai, boto3, streamlit, pytest, defusedxml.

Main settings (`.env`, template in `.env.example`):

| Setting | Value used | Meaning |
|---|---|---|
| `DATA_BACKEND` | parquet | parquet files in `data/` or PostgreSQL |
| `EMBED_PROVIDER` / `EMBED_MODEL` | sentence-transformers / `BAAI/bge-small-en-v1.5` | embedding model (384 dimensions) |
| `EMBED_BATCH_SIZE` | 64 | |
| `RERANK_MODEL` | `ncbi/MedCPT-Cross-Encoder` | cross-encoder reranker (`none` to switch off) |
| `RERANK_BLEND` | fuse | fuse reranker order with hybrid order (RRF), or `replace` |
| `LLM_PROVIDER` / `LLM_MODEL` | ollama / `qwen2.5:3b` | answer model |
| `JUDGE_PROVIDER` / `JUDGE_MODEL` | gemini / `gemini-3-flash-preview` (set per run) | judge for evaluation |
| `TOP_K` | 8 | sources given to the LLM |
| `CANDIDATES` | 60 | candidates from each of dense and BM25 |
| `RERANK_POOL` | 30 | fused candidates re-scored by the reranker |
| `AWS_REGION` / `AWS_PROFILE` | us-east-1 / empty | Bedrock; credentials come from the AWS chain, never `.env` |

API keys live only in the git-ignored `.env`.

---

## 4. Data collection

Three sources, plus an optional fourth.

### 4.1 Synthetic radiology data (`python -m medrag generate`)

No real hospital data was available (and real data would need a HIPAA BAA), so a realistic synthetic dataset was generated. Defaults: 20,000 patients, mean 5 studies per patient, 3% dirty rate, seed 7. "Now" is fixed at 2026-10-01 so runs are reproducible.

**Patients**
- Names from Faker pools in three locales (`en_IN`, `en_US`, `en_GB`), about 400 first and 400 last names each, combined at random. This deliberately produces some **shared names** (two different patients called the same), which the evaluation uses.
- MRN `MRN0000001` style, gender 49% male / 51% female, age 18 to 92, blood group, city (Bengaluru, Mumbai, Delhi, Chennai, Hyderabad, Pune, Kolkata, Mysuru, London, New York).
- Chronic conditions: Poisson count rising with age (mean 0.4 + age/40, max 6) from 24 conditions (diabetes, hypertension, COPD, heart failure, treated cancers, and so on), each with a diagnosis date.

**Studies**
- Heavy-tailed history: number of studies per patient from a gamma distribution (most have a few, some have dozens), spread over 6 years.
- Modality weights: XR 35, CT 25, US 18, MRI 15, MG 4, PET 3. Body part chosen per modality with weights (for example XR: chest 6, knee 2, spine 1, hand 1). Rules: no mammography or pelvic ultrasound for men.
- Accession `ACC000000001`, indication from a body-part list (for example chest: "Cough and fever", "Haemoptysis"), referring physician from 300 synthetic doctors.

**Findings and reports**
- 32 findings in `catalog.FINDINGS` (pneumonia, pulmonary nodule, lung mass, pleural effusion, pneumothorax, cardiomegaly, emphysema, pulmonary embolism, hepatic steatosis, liver lesion, gallstones, renal calculus, appendicitis, splenomegaly, intracranial haemorrhage, ischaemic stroke, brain tumour, small vessel disease, disc herniation, compression fracture, degenerative spondylosis, meniscal tear, ACL tear, osteoarthritis, fracture, thyroid nodule, ovarian cyst, uterine fibroid, breast mass, breast calcifications, metastatic disease, lymphoma). Each has a findings template, impression, recommendation and severity 1 to 3.
- Findings are only allowed where they make sense: body part, and for some, modality (pulmonary embolism only on CT, meniscal and ACL tears only on MRI, and so on).
- Chance of an abnormal study rises with age (45% + age/400).
- **Follow-ups**: if a patient had an abnormal study of the same modality and body part, 70% of later studies are follow-ups with a comparison line: "Unchanged" (50%), "Interval improvement" (30%) or "Interval progression" (20%) compared with the prior date. This creates real patient histories and enables the trend questions.
- Priority follows severity (severity 3: 60% stat, 40% urgent).
- Every report has five sections: `TECHNIQUE`, `COMPARISON`, `FINDINGS`, `IMPRESSION`, `RECOMMENDATION`. Templates are filled with random side, size, segment, vertebral level, bone, clock position or SUV.
- Report status: 88% final, 7% preliminary, 5% amended; signed 1 to 72 hours after the study; 60 synthetic radiologists.

**AI analysis**
- One AI prediction per study from a modality-specific model name (for example "ChestXR-Vision v3.0"). 86% correct; wrong predictions get lower confidence (mean 0.62 vs 0.88). This mimics a real imperfect AI, which is why counts use the radiologist's impression, not the AI label (section 15).

### 4.2 Deliberately dirty records (`make_dirty`)

Real hospital exports are messy, so the generator injects realistic problems at about the 3% rate, giving the cleaning step real work:

| Table | Problems injected |
|---|---|
| patients | names in upper/lower case with extra spaces; gender as `M`, `F`, `Male`, `FEMALE`, ` female`, `unknown` or missing; missing date of birth; impossible future date of birth; the same person registered twice under a new id (same MRN); 5 test records ("Test Patient", "DUMMY", "Do Not Use"...) |
| studies | modality spelt differently (`CT scan`, `Cat Scan`, `mri `, `MR`, `Xray`, `CXR`, `USG`, `Sonography`, `mammo`, `pet-ct`...); priority as `HIGH`, `Stat`, `low`, `Critical`, `medium` or empty; future and missing study dates; orphan studies pointing at a patient that does not exist; studies pointing at the duplicate patient id; exact duplicate rows |
| reports | empty report text; HTML markup (`<p>`, `<br>`); status as `approved`, `draft`, `FINAL`, `signed`, `pending`; second duplicate report for the same study |
| ai_analysis | confidence as a percentage (88.0 instead of 0.88); confidence -1; missing confidence |

Also, half of all clean modality values are written as the long name ("Ultrasound", "X-ray") instead of the code, as happens when systems are merged.

Raw output: 20,316 patients, 35,417 conditions, 92,944 studies, 91,528 AI analyses, 92,935 reports.

### 4.3 Legacy database (`python -m medrag import-legacy`)

The original `oads_db` PostgreSQL database (`oads.patients`, `oads.studies`, `oads.images`, `oads.analysis`, `oads.reports`; SQL dump in `data/source/oads_db.sql`) can be mapped onto the same raw layout: MRNs become `LEG0000001`, accessions `LEG-ACC-<id>`, modality is taken from the first image of the study. It then goes through the same check, clean and index steps. The main runs used the synthetic data because the legacy data is small.

### 4.4 Outside documents (real text)

- **Open-i** (Indiana University chest X-ray reports, US National Library of Medicine): `NLMCXR_reports.tgz` in `data/source/openi/`, 3,955 de-identified reports in XML with MeSH labels. Licence CC BY-NC-ND 4.0.
- **MTSamples** medical transcriptions: `mtsamples.csv` (4,999 rows, about 40 specialties) downloaded with the user's approval from the socd06/medical-nlp GitHub copy of the Kaggle dataset, saved in `data/documents/`.

Neither file is committed to git (the licence of MTSamples is unclear); `.gitignore` excludes `data/documents/*.csv`.

---

## 5. Data quality checks

`python -m medrag check --stage raw|clean` runs 27 checks and writes `reports/quality_<stage>.md` and `.json`. `--strict` exits with code 1 if any error-level check fails, so it can gate a pipeline.

Severity: **error** = must be fixed before use; **warn** = fix or normalise; **info** = worth knowing.

Results on the raw data:

| Table | Check | Raw rows | % | Severity |
|---|---|---:|---:|---|
| patients | duplicate patient_id | 0 | 0.00 | error |
| patients | duplicate MRN (same person, several ids) | 622 | 3.06 | warn |
| patients | duplicate name + date of birth | 614 | 3.02 | warn |
| patients | name has stray whitespace or wrong case | 656 | 3.23 | warn |
| patients | test / dummy record | 5 | 0.02 | error |
| patients | missing name | 0 | 0.00 | error |
| patients | non-standard gender value | 694 | 3.42 | warn |
| patients | missing date of birth | 196 | 0.96 | warn |
| patients | date of birth in the future | 66 | 0.32 | error |
| studies | duplicate study row | 1,416 | 1.52 | error |
| studies | duplicate study_id | 1,416 | 1.52 | error |
| studies | orphan (patient does not exist) | 567 | 0.61 | error |
| studies | non-standard modality | 29,846 | 32.11 | warn |
| studies | non-standard priority | 2,826 | 3.04 | warn |
| studies | missing study date | 609 | 0.66 | error |
| studies | study date in the future | 524 | 0.56 | error |
| studies | missing body part | 0 | 0.00 | warn |
| reports | orphan (study does not exist) | 0 | 0.00 | error |
| reports | empty report text | 858 | 0.92 | error |
| reports | contains HTML markup | 2,687 | 2.89 | warn |
| reports | duplicate report for same study | 2,814 | 3.03 | warn |
| reports | non-standard status | 2,847 | 3.06 | warn |
| reports | missing IMPRESSION section | 26 | 0.03 | warn |
| ai_analysis | orphan (study does not exist) | 0 | 0.00 | error |
| ai_analysis | missing confidence | 855 | 0.93 | warn |
| ai_analysis | confidence outside 0-1 | 3,468 | 3.79 | error |
| ai_analysis | studies without AI analysis | 0 | 0.00 | info |

Raw study dates ran from 2020-10-02 to 2029-03-19 (the future dates are the injected errors).

After cleaning, the same checks run again automatically: **all error-level checks pass** (`reports/quality_clean.md`).

---

## 6. Data cleaning

`python -m medrag clean` reads the raw stage, applies the rules below in order, writes the clean stage, writes `reports/cleaning_log.md`, then re-runs the quality checks.

### 6.1 Rules

**Patients**
1. Remove test / dummy records (name matches `test`, `dummy`, `do not use`).
2. Remove records with no name.
3. Normalise names: trim, collapse spaces, title case.
4. Normalise gender through a synonym map (`m`, `man` → male; `f`, `woman` → female; `o`, `non-binary` → other); anything else → `unknown`.
5. Clear impossible future dates of birth (set to missing, keep the patient).
6. **Merge duplicate registrations**: the same MRN under several patient ids is one person. Keep the lowest id, fill gaps in the surviving record from its duplicates (first non-empty value per column), and remember a remap (old id → kept id). Patients with no MRN cannot be matched, so they are kept as they are (this was audit finding 3; previously they were silently dropped).
7. Re-point conditions to the kept id and drop duplicate (patient, condition) pairs.

**Studies**
1. Drop exact duplicate rows, then duplicate study ids.
2. Re-point studies of merged patients to the kept id.
3. Remove orphans (patient does not exist).
4. Normalise modality with the synonym map (`ct scan`, `cat scan`, `computed tomography` → CT; `xray`, `cxr`, `radiograph` → XR; `usg`, `sonography` → US; and so on). Remove rows whose modality is still unknown.
5. Normalise priority (`low`, `normal`, `medium`, `elective` → routine; `high` → urgent; `critical`, `emergency`, `immediate` → stat; empty → routine).
6. Remove studies with missing or future dates.

**Reports**
1. Strip HTML (`<br>` → newline, remove tags, unescape entities, tidy spaces per line).
2. Remove empty reports.
3. Keep only reports of studies that survived cleaning.
4. Normalise status (`approved`, `signed`, `finalized` → final; `draft`, `prelim`, `pending` → preliminary; `addendum`, `corrected` → amended).
5. **One report per study**: prefer final, then amended, then preliminary; within that the latest signed.

**AI analysis**
1. Keep only analyses of surviving studies.
2. Convert percentage confidence (1 < value ≤ 100) to a fraction.
3. Clear confidence still outside 0 to 1.

Studies left with no usable report are kept: their metadata (date, modality, priority) still answers count questions.

### 6.2 Cleaning log (actual run)

| Step | Rows |
|---|---:|
| patients: removed test/dummy records | 5 |
| patients: normalised name whitespace/case | 654 |
| patients: normalised gender values | 694 |
| patients: cleared impossible future date of birth | 66 |
| patients: merged duplicate registrations (same MRN) | 311 |
| studies: removed duplicate rows | 1,416 |
| studies: re-pointed to merged patient | 632 |
| studies: removed orphans (unknown patient) | 558 |
| studies: normalised modality names | 29,221 |
| studies: normalised priority values | 2,761 |
| studies: removed missing / future study dates | 1,102 |
| reports: stripped HTML markup | 2,687 |
| reports: removed empty reports | 884 |
| reports: normalised status values | 2,768 |
| reports: removed duplicate reports per study | 1,369 |
| ai_analysis: converted percent confidence to 0-1 | 2,602 |
| ai_analysis: cleared invalid confidence | 800 |
| studies: without a report after cleaning | 854 |

### 6.3 Clean data

| Table | Raw rows | Clean rows |
|---|---:|---:|
| patients | 20,316 | 20,000 |
| conditions | 35,417 | 35,417 |
| studies | 92,944 | 89,868 |
| ai_analysis | 91,528 | 89,868 |
| reports | 92,935 | 89,014 |

Clean study dates: 2020-10-02 to 2026-09-29.

---

## 7. Outside documents: Open-i and MTSamples

`python -m medrag documents` loads, checks, cleans and links the outside documents, then writes `reports/quality_documents.md`. Run `index` afterwards.

### 7.1 Loading

- **Open-i**: the `.tgz` is extracted once with Python's `data` filter (blocks path traversal and links in the archive). XML is parsed with `defusedxml` (safe against XML attacks). For each report the `COMPARISON`, `INDICATION`, `FINDINGS` and `IMPRESSION` sections are joined into the text; MeSH "major" terms become labels; the impression is the description. Title "Chest X-ray report", specialty Radiology.
- **MTSamples**: columns renamed (`transcription` → text, `medical_specialty` → specialty, `sample_name` → title), keywords as labels, the dataset's one-line description kept.

### 7.2 Checks (before → after cleaning)

| Check | Before | After |
|---|---:|---:|
| empty text | 61 | 0 |
| very short (< 15 words) | 99 | 0 |
| duplicate document (same text, same source) | 2,643 | 0 |
| transcription artefacts (`:,` headings) | 4,118 | 69 |
| de-identification tokens (XXXX) | 3,423 | 0 |
| Open-i report without IMPRESSION | 34 | 6 |
| missing patient link | 8,954 | 0 |

### 7.3 Cleaning

1. **Rebuild section headings** in transcriptions. MTSamples text has CSV artefacts like `HISTORY OF PRESENT ILLNESS:,  The patient...`; these are turned back into `\nHISTORY OF PRESENT ILLNESS: The patient...` so the chunker can split on sections (4,118 documents).
2. **Mark de-identified words**: Open-i replaces names and dates with `XXXX`; these become `[redacted]` so the LLM does not treat them as a word (3,423 documents).
3. Tidy whitespace per line.
4. **Remove empty and very short** documents, under 15 words (158).
5. **Merge duplicate transcriptions**: MTSamples files the same transcription under several specialties (for example both "Surgery" and "Orthopedic"). One copy is kept and the other specialties are remembered in the specialty field (2,585 merged). Only MTSamples: identical Open-i reports ("No acute cardiopulmonary disease.") belong to different patients and are kept.

Result: 3,878 Open-i reports and 2,333 unique MTSamples transcriptions.

### 7.4 Linking to patients

The document text is real, but it has no link to the synthetic patients. To make per-patient questions possible, each document is attached to a plausible synthetic patient:

- **Gender** from the text: counts of male words (he, his, him, Mr, male, man, gentleman, boy) vs female words (she, her, Mrs, Ms, female, woman, lady, girl).
- **Age** from phrases like "45-year-old".
- A random date in the data window (2020-10-01 to 2026-09-30).
- The patient is picked at random among those of the same gender and within 2 years of the stated age at that date. If none match, gender only; if no gender either, any patient.

Linked by age + gender 1,058, by gender only 1,879, at random 3,274 (mostly Open-i reports, which are de-identified). Seeded (13), so it is reproducible. This link is synthetic and is documented as such.

---

## 8. Storage layer

`medrag/data/store.py`:

- Two stages: **raw** (as generated or imported, may be dirty) and **clean** (used for indexing).
- **parquet** backend: `data/raw/*.parquet`, `data/clean/*.parquet`.
- **PostgreSQL** backend: schemas `medrag_raw` and `medrag`. Raw tables use loose types so dirty values survive the load. Bulk load with `COPY` in 50,000-row batches. Clean tables get primary keys and indexes on `patient_id` and `study_id`.
- Id columns are written as nullable integers, so CSV never contains `12.0`.
- `read_all` loads the core tables plus `documents` when present. It only ignores "not ingested yet" (missing file, or PostgreSQL error 42P01 "table does not exist"); any other error is raised so documents are never silently dropped (audit finding 5).

---

## 9. Chunking

`medrag/ingest/chunking.py` and `medrag/ingest/documents.py`. `python -m medrag index --max-words 220`.

### 9.1 How text is split

1. Split on **section headings**: any upper-case heading at the start of a line (`FINDINGS:`, `IMPRESSION:`, `HISTORY OF PRESENT ILLNESS:`, `PLAN:`).
2. A section longer than the budget is split into **sentences**.
3. Units are packed into chunks of at most `max_words` (220) with a **one-sentence overlap** (only when that sentence is short, under a third of the budget).

### 9.2 Contextual header on every chunk

Each chunk starts with a short header so it can be understood and retrieved on its own ("contextual retrieval"). The header's words count against the budget (minimum 60 words of body).

### 9.3 Three kinds of chunk

**Study chunks** (one or more per report):
```
Patient: <name> | MRN: <mrn> | Patient ID: <id> | <gender>, age <age at study>
Study: <modality> <body part> | Date: <YYYY-MM-DD> | Accession: <acc> | Priority: <p> | Report status: <s>
Indication: <indication>
TECHNIQUE: ... COMPARISON: ... FINDINGS: ... IMPRESSION: ... RECOMMENDATION: ...
AI MODEL (<model>): predicted '<label>' (confidence 0.91).
```
Metadata stored with it: patient, study, MRN, name, gender, modality, body part, priority, date, status, AI label and confidence, impression.

**Patient profile chunks** (budget 300 words):
```
PATIENT PROFILE: <name> | MRN | Patient ID | gender, age | Blood group | City
HISTORY: Chronic conditions: ...
HISTORY: 12 imaging studies from 2021-01-03 to 2026-08-11: X-ray x5, CT x4, ...
HISTORY: 2 uploaded documents: 2025-03-01 Discharge Summary (Cardiology); ...
FINDINGS: Most recent studies first. <date> <modality> <body part> (<priority>): <impression>. ... (up to 40)
```

**Document chunks** (Open-i and MTSamples):
```
Patient: <name> | MRN | Patient ID | gender, age
Document: <title> | Type: <specialty> | Date: <date> | Source: openi|mtsamples
Summary: <MTSamples description>
<cleaned text>
```
Open-i chunks are tagged modality XR and body part chest, so filters like "chest X-ray" find them. The conclusion line (IMPRESSION, DIAGNOSIS, ASSESSMENT, CONCLUSION...) is stored as the document's impression.

Each chunk id is a SHA-1 hash of its text (used for incremental re-indexing). Duplicate chunks are dropped.

### 9.4 Final chunk counts

| Kind | Source | Chunks | Documents |
|---|---|---:|---:|
| study | synthetic reports | 89,014 | 89,014 |
| patient | profiles | 19,923 | 19,897 patients |
| document | Open-i | 3,880 | 3,878 |
| document | MTSamples | 8,518 | 2,333 |
| **Total** | | **121,335** | |

(19,897 rather than 20,000 profiles: patients with no study left after cleaning have no profile.)

---

## 10. Embeddings

`medrag/providers/embeddings.py`. One interface: `embed_documents` and `embed_query`, both return L2-normalised float32 vectors, so the inner product is the cosine similarity.

| Provider | Notes |
|---|---|
| sentence-transformers | **used**: `BAAI/bge-small-en-v1.5`, 384 dimensions, runs on CPU |
| ollama | any Ollama embedding model (for example `nomic-embed-text`) |
| openai | OpenAI embeddings API |
| gemini | Gemini embeddings with `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types |
| hash | deterministic bag-of-words hashing, 512 dimensions, no download: for tests and smoke runs only |

Models trained with asymmetric instructions get their prefixes automatically (for example bge queries get "Represent this sentence for searching relevant passages: "; e5 gets "query: " / "passage: "; nomic gets "search_query: " / "search_document: ").

Why bge-small: good retrieval quality for its size, fast enough to embed 121k chunks on a CPU. A 2026-10-27 comparison showed it matched or beat the hash baseline everywhere (section 20.2).

---

## 11. Index (dense + BM25)

`medrag/index/store.py`. One directory per embedding model under `indexes/` (for example `indexes/sentence-transformers__BAAI_bge-small-en-v1.5/`, 442 MB), so different models can be built and compared side by side.

Files in each build:

| File | Content |
|---|---|
| `chunks-<version>.parquet` | chunk text and metadata |
| `vectors-<version>.npy` | embeddings (memory-mapped when read) |
| `dense-<version>.faiss` | FAISS index: exact inner product (`IndexFlatIP`); above 150,000 chunks an approximate HNSW graph (M 32, efConstruction 80, efSearch 128) |
| `bm25-<version>/` | BM25 index (`bm25s`, k1 1.2, b 0.75, English stop words, Snowball stemmer) |
| `manifest.json` | provider, model, dimension, chunk count, embedded vs reused, index type, build time, version |

**Incremental re-build**: vectors of chunks whose text (hash) has not changed are reused; only new or changed chunks are embedded.

**Safe swap** (audit finding 6): each build writes new versioned files and switches `manifest.json` last with an atomic `os.replace`. An app reading the old build is never shown a half-written build, and memory-mapped vector files that are in use are not overwritten. Old builds are deleted afterwards; on Windows a file still open is skipped and removed by the next build. Indexes built before versioning still load.

Build history:

| Build | Chunks embedded | Time |
|---|---:|---|
| First full build (synthetic data) | ~107k | 2 h 50 min |
| Adding Open-i | 7,428 new (rest reused) | 12 min |
| Adding MTSamples | 10,749 new, 110,586 reused | 50 min (3,025 s) |

Current manifest: 121,335 chunks, 384 dimensions, `IndexFlatIP`, built 2026-10-28 01:06.

---

## 12. Keeping the index in sync

`python -m medrag watch --interval 30` polls the raw data and, when it changes, re-runs clean and index (incrementally).

How a change is detected cheaply, without reading table data (audit finding 4):
- parquet: name, modification time and size of each raw file.
- PostgreSQL: insert, update and delete counters from `pg_stat_user_tables` for the raw schema. (Previously only row counts were used, so an update that kept the count the same was missed.)

Errors (for example the database briefly down) are printed and the watcher keeps running.

---

## 13. Query understanding

`medrag/retrieval/query_parser.py` turns a question into structured filters.

| Filter | How it is found |
|---|---|
| Patient | MRN (`MRN1234567`, `LEG...`), accession (`ACC...`), "patient id 42", or a name: every 2-3 word window is compared with all patient names, exact first, then fuzzy (rapidfuzz ratio ≥ 90, which tolerates typos). Several patients with the same name are all returned. |
| Unknown patient | A capitalised name after "patient/for/of/about" that matches nobody, or an MRN/id that does not exist → the system answers "No patient matching '...' was found" without calling the LLM. |
| Modality | Synonym map (`cat scan` → CT, `cxr` → XR, `sonography` → US...). Short codes US, MR, MG, PET count **only in capitals**, so "tell us" is not ultrasound (audit finding 2). |
| Body part | Direct names, or synonyms (lung → chest, liver → abdomen, back → spine...). |
| Priority | stat, urgent, emergency, routine, "critical priority". |
| Dates | "last 3 months", "past year", "since 2023", "between 2022 and 2024", "in March 2025", "in 2024", "on 2025-03-02". |
| Documents | words like document, uploaded, note, transcription, letter, discharge summary, consult, operative report, x-ray report; or a specialty name. |
| Latest | latest, most recent, newest, last scan/study/report. |
| Aggregate | how many, count, number of, total, percentage, most common, distribution, breakdown. |
| Abnormal | abnormal, positive, pathological, significant findings. |
| Findings | from the expansion dictionary (next section). |

### Medical query expansion

`medrag/retrieval/expansion.py` maps lay and alternative terms to the vocabulary used in reports, so a question and a report describing the same thing in different words still meet.

- 32 findings, each with synonyms ("slipped disc", "herniated disc", "disc prolapse" → disc herniation; "brain bleed" → intracranial haemorrhage; "fatty liver" → hepatic steatosis; "kidney stone" → renal calculus...).
- Report wording for findings whose radiologist phrasing differs from the label (disc herniation → "disc extrusion nerve root compression"; gallstones → "cholelithiasis"; liver lesion → "arterially enhancing lesion washout LI-RADS"...).
- Longest phrase wins, no overlaps.
- The expanded terms are appended to the query for BM25, dense search and the reranker, and a note is passed to the LLM ("'slipped disc' means disc herniation (reports may say: disc extrusion...)").
- Regex patterns for how each finding appears in impressions are used for exact cohort counts.

---

## 14. Retrieval pipeline

`medrag/retrieval/retriever.py`, mode `hybrid+rerank` by default (also `hybrid`, `dense`, `bm25`).

1. **Parse** the question into filters.
2. **Filter**: restrict the search to rows matching all filters. If nothing matches, drop the least important filter and try again, in this order: abnormal, documents, priority, dates, body part, modality, patient. Each relaxation is noted ("No records match every filter; relaxed the 'modality' filter.").
3. **Expand** the query (section 13).
4. **Dense search**: top 60 by cosine similarity. Inside a filter this is an exact search over the subset's vectors.
5. **BM25 search**: top 60 over the same subset.
6. **Reciprocal Rank Fusion** (RRF, k = 60): score = Σ 1 / (60 + rank) over both lists.
7. **Rerank**: the top 30 fused candidates are re-scored by the MedCPT cross-encoder (max length 512). In `fuse` mode (default) the reranker's order is fused again with the hybrid order by RRF, so a reranker cannot throw away what the first stage got right. In `replace` mode the reranker's order is used alone.
8. **Keep top k** (8).
9. **Pin**: for a patient question, that patient's profile chunk is always included; for a "latest" question, their 3 most recent studies too. At least half of the k slots stay for search results.

Each hit records how it was found (`dense+bm25`, `profile`, `latest`), shown in the UI.

**Why MedCPT**: it is a cross-encoder trained on PubMed search logs, so it understands medical wording. A general web-trained reranker (`BAAI/bge-reranker-base`) was tested first and made hard-question MRR worse than no reranker (0.845 vs 0.876) while being slower; the fast general `ms-marco-MiniLM-L-6-v2` was roughly neutral. MedCPT with fusion gave the best overall result (section 20.3).

---

## 15. Answer generation

`medrag/rag/answer.py`.

### 15.1 Computed facts

For count, patient and "latest" questions, exact statistics are computed from the full index metadata, not from the 8 retrieved chunks:

- Filters applied; number of matching studies and patients; date range; counts by modality and priority; number and percentage of abnormal studies; most common AI findings among them (cohort questions).
- **All filters are applied when counting**, even if search relaxed one to find something to show. So "how many MRIs does X have" for a patient with no MRI gives 0, not the count of their other studies (audit finding 1).
- When a finding is asked about, studies are counted by the **radiologist's impression**, not the AI label (the AI is wrong 14% of the time).
- One patient: most recent study, list of uploaded documents, timeline of up to 60 studies.
- Several patients with the same name: a warning and a separate summary per patient (with MRN), so the LLM answers for each.

### 15.2 Prompt

System rules:
- Use COMPUTED FACTS for every count, breakdown or "latest" question.
- Otherwise use only the SOURCES; never invent patients, dates, findings or numbers.
- Cite sources inline as [S1], [S2].
- If neither contains the answer, say "The records do not contain this information."
- Concise, clinical, dates as YYYY-MM-DD.
- Decision support only; no treatment advice beyond what reports recommend.

User message: the question, expansion notes, computed facts, then the 8 sources labelled [S1]..[S8] with their document id.

### 15.3 Unknown patient

If the parser found a patient name or MRN that does not exist, the answer is returned directly ("No patient matching 'X' was found in the records.") without calling the LLM: faster and no chance of a made-up patient.

---

## 16. LLM providers

`medrag/providers/llms.py`. One interface: `stream(system, user)` yields text; `complete()` joins it and retries on rate limits (HTTP 429, RESOURCE_EXHAUSTED, "rate limit", ThrottlingException) with waits of 15, 30, 45, 60 seconds.

| Provider | Spec example | Notes |
|---|---|---|
| ollama | `ollama:qwen2.5:3b` | local; temperature 0.1, context 8,192; thinking switched off and any `<think>` text removed |
| anthropic | `anthropic:claude-opus-5-5` | streaming, medium effort; server-side refusal fallback for supported Claude models |
| openai | `openai:<model>` | |
| groq | `groq:llama-3.3-70b-versatile` | free tier, OpenAI-compatible |
| openrouter | `openrouter:<model>:free` | free models available |
| gemini | `gemini:gemini-3-flash-preview` | used as the judge |
| bedrock | `bedrock:us.amazon.nova-lite-v1:0` | AWS Converse streaming API; works with every Bedrock model (Claude, Llama, Nova, Mistral) |

Local models installed in Ollama: qwen2.5:3b, qwen3:1.7b, qwen3:4b, phi4-mini (3.8B, 2.5 GB), llama3.2:3b (2.0 GB).

---

## 17. User interface

`streamlit run app.py`.

- **Sidebar**: choose the embedding index (shows chunk count, dimension, build time), the LLM (`provider:model`), retrieval mode, number of sources (3 to 20), clear chat.
- **Chat**: streamed answers; the parsed filters and expansion notes shown above each answer; an expander with the sources (document id, how found, score, text) and the computed facts.
- **Patients**: pick a patient, see profile, studies table (date, modality, body part, priority, impression, AI label and confidence) and uploaded documents.
- **Data quality**: patient and study counts and the quality / cleaning reports.
- **Evaluation**: pick any `reports/eval_*` run and read its summary and answer table.

The index is cached and reloaded automatically when a new build appears.

---

## 18. Evaluation design

`python -m medrag eval`. The test set is built from the indexed data itself (`medrag/eval/testset.py`), so every question has a known right answer and the gold document. Seeded (11), so the same questions come back every run.

### 18.1 Question types

**Easy (125 questions at 20 per type)**

| Type | Example | Gold | Check |
|---|---|---|---|
| `lookup_by_name` | What did the CT chest on 2024-03-02 show for Asha Rao? | that study | impression keywords |
| `lookup_by_mrn` | Summarise the MRI report for patient MRN0001234 dated 2023-07-11. | that study | impression keywords |
| `latest` | What was the most recent imaging study for X and what did it show? | latest study | date + keywords |
| `count_patient` | How many X-ray studies does X have? | patient profile | exact number |
| `ambiguous_name` | How many imaging studies does <shared name> have? | none | every MRN must be named |
| `semantic` | Which patient had a CT on 2024-03-02 showing pneumonia? (unique day + modality + finding) | that study | patient's first name |
| `count_cohort` | How many stat priority MRI studies were performed in 2023? | none | exact number |
| `not_found` | Summarise the imaging history of patient Zebulon Quixote. | none | must say not found |

Only patients with a unique name are used for lookups, so the right answer is not ambiguous.

**Hard (108 questions)**: findings are described in **lay or alternative words** ("a slipped disc", "water on the lung", "a crushed vertebra"). Some phrasings are deliberately *not* in the expansion dictionary, and each question records `in_dictionary`, so the effect of expansion is measured honestly.

| Type | Example | Check |
|---|---|---|
| `hard_paraphrase_patient` | When did X have imaging that showed a slipped disc, and what kind of scan was it? (patient with 6+ studies) | date + modality |
| `hard_semantic_cohort` | Which patients had fluid around the lung on a CT scan in March 2024? (1-3 right answers) | every surname |
| `hard_trend` | On the follow-up scan of 2025-01-04, had X's fatty liver improved, worsened or stayed the same? | right trend word |
| `hard_count_finding` | How many patients had a kidney stone reported on imaging in 2023? | exact number |
| `hard_negative` | Has X ever had a torn ACL on any scan? (they never had) | must say no |
| `hard_typo_name` | What did the CT on 2024-03-02 show for "Asha Ro"? (one letter removed from the surname) | impression keywords |

**Documents (80 questions)**

| Type | Example | Check |
|---|---|---|
| `doc_patient_lookup` | What did X's uploaded chest X-ray report from 2025-02-10 conclude? (abnormal Open-i report) | impression keywords |
| `doc_semantic` | Which patient has an uploaded chest X-ray report describing <5 distinctive words>? (impression unique in the collection) | surname |
| `doc_transcription_lookup` | Summarise X's Laparoscopic Cholecystectomy note. | conclusion keywords |
| `doc_transcription_semantic` | Which patient had a cardiology document about: <MTSamples description>? | surname |

Total: 313 questions (125 easy, 108 hard, 80 documents). 258 have a gold document and are used for retrieval scoring; all types are used for answer scoring.

### 18.2 Human review sheet

Each run writes `golden_review.csv`: 30 random non-easy questions with the expected answer and the gold evidence text, plus empty "reviewer_ok (y/n)" and "reviewer_note" columns, so a person can check that the expected answers are right.

### 18.3 Configurations compared

- `--indexes`: several embedding indexes (for example bge vs hash).
- `--rerankers`: several cross-encoders (first is the main one).
- `--blends`: each reranker also in `replace` mode.
- `--baselines`: ablations on the main configuration: hybrid without reranker, dense only, BM25 only, no query expansion, no filters.
- `--llms`: several answer models on the same questions.
- `--judge`: also score answers with the judge LLM.
- `--difficulty`: only easy, hard or documents.
- `--max-answers`: how many questions get an LLM answer, stratified: the same number from every type (16 = one per type).

---

## 19. Evaluation metrics

### 19.1 Retrieval (k = 8)

For each question with a gold document, after removing duplicates and (unless the gold is a profile) the always-pinned profile chunk, so the search itself is scored:

| Metric | Meaning |
|---|---|
| **Hit@8** | 1 if any gold document is in the top 8 |
| **MRR** | 1 / rank of the first gold document (0 if none in the top 8) |
| **Recall@8** | share of the gold documents found in the top 8 (matters for cohort questions with several right answers) |
| **nDCG@8** | rewards gold documents near the top: Σ rel / log2(rank + 1), divided by the best possible |
| Latency | seconds per search |

Reported per difficulty, per question type (MRR), and for paraphrases split by whether the wording is in the expansion dictionary (Recall).

### 19.2 Answers: deterministic checks

| Metric | Meaning |
|---|---|
| `fact_recall` | share of expected key facts present in the answer (a fact may list alternatives, `ct|computed tomography`) |
| `number_correct` | the exact expected number appears (with or without thousands separator) |
| `negative_correct` | the answer says no/never/none and does not start with "yes" |
| `abstain_correct` | the answer says "not found / no record / does not contain" exactly when it should |
| `cites_sources` | the answer cites at least one [S#] |
| **`auto_correct`** | one pass/fail per question from the check that fits it: number → number_correct; negative → negative_correct; not-found → abstain_correct; otherwise all facts present |
| Gen. time | seconds to generate the answer |

### 19.3 Answers: LLM judge

A different, stronger model gets the question, the reference answer, the computed facts and the retrieved sources, and scores 1 to 5:

- **correctness**: agrees with the reference answer
- **faithfulness**: every claim supported by the sources or computed facts (nothing invented)
- **answer_relevance**: addresses what was asked, without padding
- **context_relevance**: the retrieved sources contain what is needed

It replies in JSON with a one-sentence reason. A judge failure does not lose the answer's other metrics.

**Judge agreement**: the share of answers where "judge correctness ≥ 4" and "auto_correct = pass" agree. Low agreement means one of the two needs review.

### 19.4 Confidence intervals

Every mean is reported with a **bootstrap 95% confidence interval** (1,000 resamples, seeded), for example `0.809 [0.73-0.88]`. With small samples (16 answers) the intervals are wide, and the report says so.

### 19.5 Outputs of each run

`reports/eval_<date>_<time>/`: `testset.json`, `golden_review.csv`, `retrieval.csv` (one row per question per configuration), `generation.csv` (one row per answer with the text, timings, every check and judge scores), `summary.md`.

---

## 20. All evaluation runs and results

### 20.1 Run history

| Run | What | Notes |
|---|---|---|
| `eval_20261027_1415` | first smoke run | incomplete |
| `eval_20261027_1428` | hash embedder, all modes + no filters | pipeline check before the real model was downloaded |
| `eval_20261027_1726` | bge vs hash, all modes | bge ≥ hash everywhere |
| `eval_20261027_1821` | bge, easy set, first answers from qwen2.5:3b | |
| `eval_20261027_1958` | 233 questions (easy + hard): rerankers bge-reranker-base and MiniLM, ablations | general rerankers hurt hard questions |
| `eval_20261027_2033` | 273 questions incl. 40 documents (Open-i): MedCPT, MiniLM, fuse vs replace, all ablations | main retrieval table |
| `eval_20261027_2146` | MedCPT default + judged answers | Gemini quota ran out mid-run |
| `eval_20261028_0006` | complete judged answer run, qwen2.5:3b, judge gemini-3-flash-preview | main answers table |
| `eval_20261028_0107` | documents with baselines | stalled for hours (machine sleep), stopped |
| `eval_20261028_0753` | documents only, 80 questions after MTSamples | |
| `eval_20261028_1044` | qwen2.5:3b, qwen3:1.7b, qwen3:4b on the same 18 questions | model comparison |
| `eval_20261028_1350` | phi4-mini, llama3.2:3b on the same 18 questions | model comparison |

### 20.2 Embedding model: bge-small vs hash baseline (`eval_20261027_1726`, easy set)

| Index | Mode | MRR | Latency |
|---|---|---:|---:|
| bge-small | hybrid+rerank | 0.987 | 0.39 s |
| hash | hybrid+rerank | 0.987 | 0.34 s |
| bge-small | dense only | 0.967 | 0.09 s |
| hash | dense only | 0.950 | 0.04 s |
| bge-small | no filters | 0.645 | 5.7 s |

On easy questions metadata filters do almost all the work, so even a hashing "embedding" scores well. The difference shows on hard and document questions (dense-only hard MRR 0.913 with bge), which is why bge-small is used.

### 20.3 Retrieval: all configurations (`eval_20261027_2033`, MRR@8)

| Configuration | Easy | Hard | Documents (Open-i) | Latency |
|---|---:|---:|---:|---:|
| **hybrid + MedCPT reranker (fuse), default** | 1.000 | 0.905 | 0.808 | 2.8 s |
| hybrid + MedCPT reranker (replace) | 1.000 | 0.915 | 0.793 | 2.8 s |
| hybrid + MiniLM reranker (fuse) | 1.000 | 0.885 | 0.797 | 0.7 s |
| hybrid, no reranker | 1.000 | 0.875 | 0.762 | 0.1 s |
| dense only | 0.995 | 0.913 | 0.621 | 0.1 s |
| BM25 only | 1.000 | 0.865 | 0.837 | 0.05 s |
| hybrid + rerank, no query expansion | 0.995 | 0.793 | 0.797 | 0.6 s |
| hybrid + rerank, no filters | 0.495 | 0.509 | 0.507 | 4.1 s |

Earlier (`eval_20261027_1958`, hard questions): general reranker `bge-reranker-base` 0.845, hybrid with no reranker 0.876, MiniLM 0.874, no expansion 0.817.

Findings:
- **Metadata filtering is the biggest win**: it roughly doubles MRR (0.5 → 0.9-1.0).
- **Query expansion** adds about 0.09-0.11 MRR on hard questions. On wording that is not in the dictionary, hybrid search still finds 92.6% of gold documents (Recall) vs 100% for wording in it.
- **Reranker choice matters**: a general web-trained reranker made results worse; the medical MedCPT helps; fusing its order with the hybrid order is the safest.
- **Dense alone** is weak on keyword-style document questions (`doc_semantic` 0.27); **BM25 alone** is weak on paraphrases. Hybrid covers both.
- Differences of a few hundredths are within the confidence intervals; the top three rerank configurations are effectively equivalent.

nDCG@8 for the default: easy 1.000, hard 0.920, documents 0.831.

### 20.4 Retrieval: default configuration on the full current test set (`eval_20261028_1044`)

| Difficulty | Questions | Hit@8 | MRR | Recall@8 | nDCG@8 | Latency |
|---|---:|---|---|---|---|---:|
| Easy | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.7 s |
| Hard | 78 | 0.987 [0.96-1.00] | 0.905 [0.85-0.95] | 0.983 [0.95-1.00] | 0.920 [0.88-0.96] | 2.6 s |
| Documents | 80 | 0.875 [0.80-0.94] | 0.809 [0.73-0.88] | 0.875 [0.80-0.94] | 0.825 [0.75-0.89] | 8.0 s |

MRR by type: count_patient 1.00, lookup by name/MRN 1.00, latest 1.00, semantic 1.00, typo name 1.00, hard paraphrase 0.938, hard trend 0.875, hard semantic cohort 0.815, doc patient lookup 0.975, doc transcription lookup 1.00, doc transcription semantic 0.70, doc semantic (Open-i by content) 0.562.

Finding a document by content alone is the hardest case: many Open-i reports use near-identical wording ("no acute cardiopulmonary disease").

### 20.5 Answers judged by an LLM (`eval_20261028_0006`)

qwen2.5:3b answering, gemini-3-flash-preview judging, 16 questions (8 easy, 6 hard, 2 documents; one per type, before the two MTSamples types existed):

| | Easy | Hard | Documents |
|---|---:|---:|---:|
| Pass the exact check (auto_correct) | 63% | 83% | 50% |
| Judge correctness (1-5) | 3.6 | 3.2 | 3.0 |
| Judge faithfulness (1-5) | 4.1 | 3.7 | 3.0 |
| Judge answer relevance (1-5) | 4.9 | 5.0 | 5.0 |
| Judge context relevance (1-5) | 4.5 | 4.5 | 5.0 |
| Says "not found" correctly | 100% | 83% | 100% |
| Cites sources | 13% | 17% | 100% |
| Seconds per answer (CPU) | 76 | 97 | 82 |

- Scored correctly: lookups by name and MRN, latest study, per-patient counts, not found, misspelt names, paraphrases, document lookups.
- Failed: cohort counts, ambiguous names (shared by two patients), semantic search across many records, finding counts, document content search.
- Judge vs exact checks agree on **88%** of answers; both disagreements are on the looser checks ("never had", trends), where the judge is probably right.

### 20.6 Five local models on the same 18 questions (`eval_20261028_1044`, `eval_20261028_1350`)

18 questions, one per type (8 easy, 6 hard, 4 documents); exact checks only (no judge), CPU:

| Model | Easy | Hard | Documents | Correct "not found" | Cites sources | Seconds per answer |
|---|---:|---:|---:|---:|---:|---:|
| **qwen3:1.7b** | 88% | 83% | 75% | 25-63% | 50-63% | 44 |
| qwen3:4b | 88% | 83% | 75% | 75-88% | 75-100% | 450 |
| llama3.2:3b | 88% | 83% | 50% | 50-88% | 0-50% | 57 |
| phi4-mini (3.8B) | 88% | 50% | 75% | 50-100% | 33-50% | 56 |
| qwen2.5:3b (default) | 63% | 67% | 50% | 75-100% | 0-50% | 55 |

(Ranges are across the three difficulty groups.)

Pass/fail by question type:

| Type | qwen2.5:3b | qwen3:1.7b | qwen3:4b | llama3.2:3b | phi4-mini |
|---|:-:|:-:|:-:|:-:|:-:|
| ambiguous_name | ✗ | ✓ | ✓ | ✓ | ✓ |
| count_cohort | ✗ | ✗ | ✗ | ✗ | ✗ |
| count_patient | ✓ | ✓ | ✓ | ✓ | ✓ |
| doc_patient_lookup | ✓ | ✓ | ✓ | ✓ | ✓ |
| doc_semantic | ✗ | ✓ | ✓ | ✗ | ✓ |
| doc_transcription_lookup | ✓ | ✓ | ✓ | ✓ | ✓ |
| doc_transcription_semantic | ✗ | ✗ | ✗ | ✗ | ✗ |
| hard_count_finding | ✗ | ✗ | ✗ | ✗ | ✗ |
| hard_negative | ✓ | ✓ | ✓ | ✓ | ✓ |
| hard_paraphrase_patient | ✓ | ✓ | ✓ | ✓ | ✓ |
| hard_semantic_cohort | ✗ | ✓ | ✓ | ✓ | ✗ |
| hard_trend | ✓ | ✓ | ✓ | ✓ | ✗ |
| hard_typo_name | ✓ | ✓ | ✓ | ✓ | ✓ |
| latest | ✓ | ✓ | ✓ | ✓ | ✓ |
| lookup_by_mrn | ✓ | ✓ | ✓ | ✓ | ✓ |
| lookup_by_name | ✓ | ✓ | ✓ | ✓ | ✓ |
| not_found | ✓ | ✓ | ✓ | ✓ | ✓ |
| semantic | ✗ | ✓ | ✓ | ✓ | ✓ |

Conclusions:
- **qwen3:1.7b** is the best speed/accuracy trade-off; its weakness is answering when it should say "not found".
- **qwen3:4b** scores the same, is the most careful and cites best, but is 10 times slower (7-9 minutes per answer on CPU).
- **phi4-mini** is reliable at "not found" on easy and document questions but misses hard cohort and trend questions.
- **llama3.2:3b** matches qwen3 on easy and hard but is weaker on documents and rarely cites.
- **Every model fails** cohort counts, finding counts and transcription content search: those need retrieval or computed-fact work, not a different model.
- The default was kept at qwen2.5:3b by the user's choice; `LLM_MODEL=qwen3:1.7b` in `.env` switches.
- With one question per type, one answer moves a row by 12 points (easy), 17 (hard) or 25 (documents): treat differences as a direction, not proof. (qwen2.5:3b scored 83% on hard in the judged run and 67% here because of one question.)

---

## 21. Code audit and fixes

A whole-repository audit was run. Findings 1-6 were fixed with tests (finding 7); 8-12 are recorded but not fixed.

| # | Finding | Fix |
|---|---|---|
| 1 | Counts used the *relaxed* filter: "how many MRIs does X have" for a patient with no MRI returned the count of their other studies | Count with every filter applied (`FILTER_ORDER` shared by search and counting) |
| 2 | "tell **us**" parsed as ultrasound (also MR, MG, PET) | Short modality codes count only in capitals |
| 3 | Patients with no MRN were dropped during MRN de-duplication | Kept as they are; logged |
| 4 | `watch` detected changes by row counts only, so updates were missed | Signature from file times and sizes (parquet) or insert/update/delete counters (PostgreSQL) |
| 5 | `read_all` swallowed every error when reading documents, so a real failure silently dropped all documents | Only "not ingested yet" is ignored |
| 6 | Re-indexing overwrote files a running app had memory-mapped; a reader could see half a build | Versioned files + atomic manifest swap + deferred clean-up |
| 7 | Risky code without tests | 4 regression tests added (and later 1 for Bedrock) |
| 8 | (open) "relaxed" notes can mention filters that were never set | |
| 9 | (open) `--max-answers` is split evenly per type and rounded, so the real number can differ (16 asked → 18 answers with 18 types) | |
| 10 | (open) a partly extracted Open-i archive is never repaired | |
| 11 | (open) the "abnormal" rule is written in two places | |
| 12 | (open) duplicated not-found message; unused `clean_dir` property | |

The new tests were confirmed to fail on the old code (via `git stash`) and pass on the new.

---

## 22. Tests

`pytest` — 15 tests, all passing. They build a small dataset with the hash embedder (no downloads) and run the real pipeline.

| Test | Checks |
|---|---|
| `test_raw_data_is_dirty_and_clean_data_passes` | raw data fails error checks; clean data passes them all |
| `test_chunks_have_context_headers` | every chunk starts with its patient/study header |
| `test_incremental_rebuild_reuses_vectors` | a re-build embeds nothing new, leaves exactly one vectors file, and the index loads |
| `test_parser_and_retrieval` | patient, modality, date filters; right study retrieved |
| `test_computed_facts_count_exactly` | counts match the data exactly |
| `test_eval_metrics` | Hit, MRR, Recall, nDCG formulas |
| `test_query_expansion_maps_lay_terms` | "slipped disc" → disc herniation, etc. |
| `test_answer_scoring` | number, negative, abstain and fact checks |
| `test_hard_testset` | hard questions are built with gold documents |
| `test_document_cleaning_and_linking` | headings rebuilt, duplicates merged, every document linked |
| `test_chunking_splits_long_text` | long text split within the word budget |
| `test_count_is_zero_when_patient_lacks_the_modality` | audit finding 1 |
| `test_short_modality_codes_only_in_capitals` | audit finding 2 |
| `test_clean_keeps_patients_without_mrn` | audit finding 3 |
| `test_bedrock_streams_converse_text` | Bedrock provider with a fake client |

---

## 23. Problems hit and how they were solved

| Problem | Cause | Solution |
|---|---|---|
| General reranker made hard questions worse | web-trained model does not understand medical wording | switched to MedCPT; fuse its order with the hybrid order |
| Counts depended on what retrieval returned | LLM counted the 8 sources | exact computed facts from the full metadata |
| Counts by finding were off | AI label is wrong 14% of the time | count by the radiologist's impression (regex per finding) |
| Name questions mixed two patients | shared names in the data | return all matching patients; warning + per-patient facts; `ambiguous_name` test type |
| Lay terms missed reports | "slipped disc" vs "disc extrusion" | medical query expansion with report wording |
| MTSamples headings glued to text | CSV export artefact `HEADING:,` | rebuild headings before chunking |
| Duplicate transcriptions | same text under several specialties | merge, keep all specialties |
| Gemini judge 429/503 | free-tier quota is per model per day; `gemini-3.8-flash` was exhausted | judge model overridable per run (`JUDGE_MODEL=gemini-3-flash-preview`), retry with back-off |
| Documents eval stalled for hours | laptop went to sleep during the long "no filters" baseline | stopped it, requested keep-awake, re-ran without baselines |
| Audit script could not import the package | run from outside the project | `PYTHONPATH=.` |
| Answers slow on CPU (40-450 s) | 3-4B models on CPU | measured; options in section 24 |
| Qwen3 printed its reasoning | thinking mode | `think: false` and `<think>` blocks removed from the stream |

---

## 24. Deployment: AWS Bedrock

On a CPU laptop local models take 40 to 450 seconds per answer. For real-time use by several people the plan is a managed model service rather than a GPU server: **AWS Bedrock**.

- Provider `bedrock` added (`BedrockLLM`) using the Converse streaming API, so one code path works for every Bedrock model (Amazon Nova, Claude, Llama, Mistral).
- Configure: `LLM_PROVIDER=bedrock`, `LLM_MODEL=us.amazon.nova-lite-v1:0` (or another id from the Bedrock console), `AWS_REGION`, optional `AWS_PROFILE`.
- Credentials come from the standard AWS chain (`aws configure`, `AWS_PROFILE`, or an IAM role on EC2/ECS), never from `.env`.
- Throttling (`ThrottlingException`) is retried with back-off.
- Expected speed: a few seconds per answer instead of a minute. Retrieval adds about 2-3 s (MedCPT on CPU); for lower latency, the MiniLM reranker (0.7 s) loses little quality.
- To evaluate on Bedrock: `python -m medrag eval --llms "bedrock:us.amazon.nova-lite-v1:0" --max-answers 16`.
- **Status: paused by the user.** Live testing needs Bedrock model access enabled in the AWS console and `aws configure`.
- Real patient data on AWS needs a signed HIPAA Business Associate Agreement and HIPAA-eligible services.

---

## 25. Security and data handling

- API keys only in the git-ignored `.env`; never printed or committed (git history checked: the Gemini key was never committed).
- AWS credentials from the AWS chain only.
- Outside data files (`mtsamples.csv`, Open-i archive) are not committed.
- The Open-i archive is extracted with Python's safe `data` filter; XML parsed with `defusedxml`.
- SQL uses parameters for values; table and schema names come from fixed lists in the code.
- All patient data is synthetic. Open-i and MTSamples are public, de-identified texts; their link to patients is synthetic.
- The system prompt limits answers to the records and forbids treatment advice beyond what reports recommend. It is decision support, not diagnosis.

---

## 26. Known limits and next steps

**Limits**
- Cohort counts ("how many patients had X in 2023") and finding counts fail with every local model even though the computed facts contain the number: the small models do not reliably pick it. Fix in code: answer pure count questions directly from the computed facts.
- Finding a document by content alone is weak (Open-i 0.56 MRR) because many reports are nearly identical.
- 16-18 answers per model is a small sample; confidence intervals are wide.
- The data is synthetic; real reports are more varied.
- Answers are slow on CPU.
- Audit findings 8-12 are open.

**Next steps**
1. Return exact counts directly for count questions (no LLM needed).
2. Larger answer evaluation (for example 4 per type, 64 answers) with the judge, for tighter intervals.
3. Bedrock live test once AWS access is set up.
4. Consider `qwen3:1.7b` as the local default.
5. Fix audit findings 8-12.

---

## 27. Command reference

```bash
# one go: generate -> check -> clean -> index
python -m medrag pipeline

# step by step
python -m medrag generate --patients 20000 --mean-studies 5 --dirty-rate 0.03 --seed 7
python -m medrag import-legacy            # optional: the original oads_db tables
python -m medrag check --stage raw
python -m medrag clean
python -m medrag check --stage clean --strict
python -m medrag documents                # Open-i + MTSamples
python -m medrag index --max-words 220

# ask
python -m medrag ask "What was the most recent imaging study for Asha Rao?"
python -m medrag ask --llm bedrock:us.amazon.nova-lite-v1:0
streamlit run app.py

# keep the index in sync
python -m medrag watch --interval 30

# evaluate
python -m medrag eval --baselines --blends --rerankers "ncbi/MedCPT-Cross-Encoder,cross-encoder/ms-marco-MiniLM-L-6-v2"
python -m medrag eval --llms "ollama:qwen2.5:3b,ollama:qwen3:1.7b" --max-answers 16
JUDGE_MODEL=gemini-3-flash-preview python -m medrag eval --judge --max-answers 16
python -m medrag eval --difficulty documents --skip-generation

# tests
pytest
```

On this machine Python is run as `PYTHONIOENCODING=utf-8 .venv/Scripts/python`.

---

## 28. Git history

| Commit | Change |
|---|---|
| `ff6778d` | MedRAG: rebuilt medical RAG pipeline |
| `3e77e55` | Outside documents, query expansion, MedCPT reranker, harder evaluation |
| `d76d4c9` | Audit fixes: exact counts, modality parsing, MRN merge, sync, index safety |
| `560109c` | MTSamples transcriptions and judged answer results |
| `14a5045` | README: documents retrieval results with MTSamples |
| `fef51fb` | AWS Bedrock LLM provider |
| `fa71d71` | README: five local LLMs compared |
| `2317364` | README: corrected not-found ranges |
