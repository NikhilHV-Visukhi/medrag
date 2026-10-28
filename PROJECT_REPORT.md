# MedRAG: full project report

Everything done in this project, end to end: data collection, quality checks, cleaning, outside documents, chunking, embedding, indexing, retrieval, answering, evaluation (all question types and metrics), every experiment and result, problems hit along the way, fixes, deployment options and known limits. **Part I** (sections 1-28) describes what was done and the results; **Part II** (sections 29-59) is the technical deep dive: for every stage, the method, where it is in the code, how it works with sample code, and why it was chosen.

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

**Part II: technical deep dive** (method, where, how and why for every stage, with code): sections 29-59, [go to Part II](#part-ii-technical-deep-dive-method-where-how-why)

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
| `fc3fae2` | Full project report; model-comparison question count corrected |

---

# Part II: Technical deep dive (method, where, how, why)

Part I says *what* was done. This part goes stage by stage through the code: **which method** is used, **where** it lives (file and function), **how** it works (real code from the repository, trimmed for reading), and **why** it was chosen over the alternatives.

## Part II contents

29. [Configuration](#29-configuration)
30. [Data generation](#30-data-generation)
31. [Dirty data injection](#31-dirty-data-injection)
32. [Data quality checks](#32-data-quality-checks)
33. [Data cleaning and manipulation](#33-data-cleaning-and-manipulation)
34. [Storage: parquet and PostgreSQL](#34-storage-parquet-and-postgresql)
35. [Outside documents: loading, cleaning, linking](#35-outside-documents-loading-cleaning-linking)
36. [Chunking](#36-chunking)
37. [Embeddings](#37-embeddings)
38. [Vector database: FAISS](#38-vector-database-faiss)
39. [Keyword index: BM25](#39-keyword-index-bm25)
40. [Index build: incremental and versioned](#40-index-build-incremental-and-versioned)
41. [Live sync: watch](#41-live-sync-watch)
42. [Query parsing](#42-query-parsing)
43. [Query expansion](#43-query-expansion)
44. [Metadata filtering with relaxation](#44-metadata-filtering-with-relaxation)
45. [Hybrid search and Reciprocal Rank Fusion](#45-hybrid-search-and-reciprocal-rank-fusion)
46. [Reranking: MedCPT cross-encoder](#46-reranking-medcpt-cross-encoder)
47. [Pinning profile and latest studies](#47-pinning-profile-and-latest-studies)
48. [Computed facts](#48-computed-facts)
49. [Prompt and guardrails](#49-prompt-and-guardrails)
50. [LLM providers, streaming and retries](#50-llm-providers-streaming-and-retries)
51. [Evaluation: building the test set](#51-evaluation-building-the-test-set)
52. [Evaluation: retrieval metrics](#52-evaluation-retrieval-metrics)
53. [Evaluation: answer checks](#53-evaluation-answer-checks)
54. [Evaluation: LLM judge](#54-evaluation-llm-judge)
55. [Evaluation: confidence intervals and report](#55-evaluation-confidence-intervals-and-report)
56. [User interface](#56-user-interface)
57. [Tests](#57-tests)
58. [One question, end to end](#58-one-question-end-to-end)
59. [Design decisions at a glance](#59-design-decisions-at-a-glance)

---

## 29. Configuration

**Method:** `pydantic-settings` `BaseSettings` with a cached singleton.
**Where:** `medrag/config.py` → `Settings`, `get_settings()`, `index_path()`.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    data_backend: str = "postgres"            # postgres | parquet
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "ncbi/MedCPT-Cross-Encoder"
    llm_provider: str = "ollama"
    top_k: int = 8
    candidates: int = 60
    rerank_pool: int = 30
    rerank_blend: str = "fuse"

    def index_path(self, provider=None, model=None) -> Path:
        slug = f"{provider or self.embed_provider}__{model or self.embed_model}".replace("/", "_").replace(":", "_")
        return self.index_dir / slug          # one index folder per embedding model

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Why:**
- **Typed values:** a wrong value in `.env` (`TOP_K=abc`) fails at start-up instead of deep inside retrieval.
- **Order of precedence:** environment variables override `.env`, so a single run can change the judge without editing files (`JUDGE_MODEL=gemini-3-flash-preview python -m medrag eval ...`).
- **`lru_cache`:** the file is read once.
- **`index_path`:** gives every embedding model its own folder, so several indexes can be compared side by side.

---

## 30. Data generation

**Method:** seeded NumPy random generator + Faker name pools + clinical templates from a catalog.
**Where:** `medrag/data/generate.py` → `generate()`, `_name_pool()`, `_fill()`, `_report()`; vocabulary in `medrag/data/catalog.py` (`FINDINGS`, `TECHNIQUES`, `MODALITY_WEIGHTS`, `MODALITY_BODY_PARTS`, `FINDING_MODALITIES`, `PROGRESSION`).

```python
rng = np.random.default_rng(seed)                                   # one seeded generator: reproducible
n_studies = max(1, int(rng.gamma(1.2, mean_studies / 1.2)))         # heavy-tailed history per patient
modality = rng.choice(modalities, p=mod_p)                          # XR 35, CT 25, US 18, MRI 15, MG 4, PET 3

prior = last_abnormal.get((modality, body_part))
if prior and rng.random() < 0.7:                                    # 70%: follow-up of a known abnormality
    finding = prior[1]
    trend = rng.choice(["stable", "improved", "worse"], p=[0.5, 0.3, 0.2])
    comparison = C.PROGRESSION[trend].format(prior=prior[0])        # "Interval progression compared with ... 2023-01-04"
elif options and rng.random() < 0.45 + min(age, 90) / 400:          # chance of disease rises with age
    finding = options[int(rng.integers(len(options)))]

# AI model: right 86% of the time, lower confidence when wrong
correct = rng.random() < 0.86
conf = float(np.clip(rng.normal(0.88 if correct else 0.62, 0.08), 0.05, 0.999))
```

Report template (`_report`). The same random seed fills findings and impression, so the size and side agree:

```python
seed = int(rng.integers(1 << 30))
findings   = _fill(f_t, np.random.default_rng(seed))   # "A 14 mm solid nodule in the left upper lobe..."
impression = _fill(i_t, np.random.default_rng(seed))   # "Indeterminate 14 mm pulmonary nodule."
text = (f"TECHNIQUE: {technique}\nCOMPARISON: {comparison or 'None available.'}\n"
        f"FINDINGS: {findings}\nIMPRESSION: {impression}\nRECOMMENDATION: {rec}")
```

**Why:**
- **No real data:** there was no real hospital data, and real data needs a HIPAA BAA. Synthetic data has a **known right answer** for every question, which makes the evaluation exact.
- **Gamma distribution:** gives a realistic long tail. Most patients have a few studies and some have dozens, which tests "summarise a long history".
- **Follow-up chains:** create coherent histories and the trend questions ("improved, worsened or stayed the same?").
- **Realistic findings:** `FINDING_MODALITIES` stops impossible combinations (an ACL tear on X-ray, a pulmonary embolism on ultrasound).
- **Imperfect AI model:** makes it possible to show that counting by the AI label is wrong, and why the radiologist's impression is used instead (section 48).
- **Shared names:** come from the Faker pool (about 400 × 400 names for 20,000 patients) and drive the `ambiguous_name` tests.
- **Reproducible:** a fixed "now" (2026-10-01) and seed 7.

---

## 31. Dirty data injection

**Method:** random masks at a configurable rate (`--dirty-rate 0.03`) that corrupt values the way real exports do.
**Where:** `medrag/data/generate.py` → `make_dirty()`.

```python
def pick(df, frac=1.0):
    return rng.random(len(df)) < rate * frac                       # each row corrupted with probability rate*frac

p.loc[m, "full_name"] = ["  " + n.upper() + " " if i % 2 else n.lower() for i, n in enumerate(p.loc[m, "full_name"])]
p.loc[m, "gender"] = rng.choice(["M", "F", "Male", "FEMALE", " female", "unknown"], size=int(m.sum()))
dup = p[pick(p, 0.5)].copy()
dup["patient_id"] = np.arange(len(p) + 1, len(p) + 1 + len(dup))  # same person, new id, same MRN
s.loc[m, "modality"] = [rng.choice(variants[c]) for c in canon]   # "Cat Scan", "CXR", "USG": still the TRUE modality
s.loc[pick(s, 0.2), "patient_id"] = 10_000_000                     # orphan
t["studies"] = pd.concat([s, s[pick(s, 0.5)]], ignore_index=True) # exact duplicate rows
r.loc[m, "report_text"] = "<p>" + r.loc[m, "report_text"].str.replace("\n", "<br>  ") + "</p>"
a.loc[m, "confidence"] = (a.loc[m, "confidence"] * 100).round(1)  # percent instead of fraction
```

**Why:**
- **Real work for the cleaning step:** the cleaning and quality steps need realistic problems to fix.
- **Testable cleaning:** misspelt modalities keep the true value, so a correct cleaning step must recover it exactly. `test_raw_data_is_dirty_and_clean_data_passes` checks this.
- **Tunable rate:** `--dirty-rate 0` gives clean data.

---

## 32. Data quality checks

**Method:** vectorised pandas boolean masks; each check produces `{table, check, count, pct, severity}`.
**Where:** `medrag/data/quality.py` → `check()`, `summary()`, `to_markdown()`, `run()`. CLI: `python -m medrag check --stage raw|clean [--strict]`.

```python
def add(table, name, mask_or_count, severity="warn"):
    count = int(mask_or_count.sum()) if hasattr(mask_or_count, "sum") else int(mask_or_count)
    issues.append({"table": table, "check": name, "count": count, "pct": _pct(count, total),
                   "severity": severity if count else "ok"})

add("patients", "duplicate MRN (same person, several ids)", p["mrn"].duplicated(keep=False))
add("studies",  "orphan (patient does not exist)", ~s["patient_id"].isin(known_pids), "error")
add("reports",  "contains HTML markup", text.str.contains(r"<[a-zA-Z/][^>]*>", regex=True))
add("ai_analysis", "confidence outside 0-1", (conf < 0) | (conf > 1), "error")
```

`--strict` turns the report into a pipeline gate:

```python
if a.strict and any(i["severity"] == "error" for i in issues):
    sys.exit(1)
```

**Why:**
- **Speed:** vectorised masks check about 90k rows in milliseconds. A row-by-row loop would be about 100 times slower.
- **Same checks before and after:** running them on raw and on clean data proves the cleaning worked, with numbers. Raw had 13 error-level checks failing; clean has 0.
- **Severity levels:** separate "must fix" from "normalise" from "information".
- **Two output formats:** Markdown for people, JSON for machines (the UI's data quality tab).
- **No library:** Great Expectations or pandera would add a framework for 27 checks that fit in about 50 lines.

---

## 33. Data cleaning and manipulation

**Method:** ordered, logged pandas transformations; synonym maps for normalisation; group-by de-duplication with an id remap; rank-and-keep for duplicate reports.
**Where:** `medrag/data/clean.py` → `clean()`, `_norm()`, `_strip_html()`; synonym maps in `medrag/data/catalog.py`.

### 33.1 Normalisation through synonym maps

```python
def _norm(series, synonyms, default):
    return series.fillna("").astype(str).str.strip().str.lower().map(synonyms).fillna(default)

p["gender"]   = _norm(p["gender"], C.GENDER_SYNONYMS, "unknown")      # "M", " female", "FEMALE" -> male/female
s["priority"] = _norm(s["priority"], C.PRIORITY_SYNONYMS, "routine")  # "HIGH" -> urgent, "Critical" -> stat
mod = s["modality"].fillna("").str.strip().str.lower().map(C.MODALITY_SYNONYMS)
mod = mod.fillna(s["modality"].where(s["modality"].isin(C.MODALITIES)))  # already-canonical codes kept
```

**Why:**
- **Explicit:** a dictionary is auditable and deterministic. Every mapping is visible in `catalog.py`.
- **Shared with the parser:** the same maps are used to parse questions (section 42), so "cat scan" means CT both in the data and in the question.
- **Not fuzzy:** fuzzy matching could map "MR" to "MRN".

### 33.2 Duplicate patients: merge by MRN, remap foreign keys

```python
p = p.sort_values("patient_id")
has_mrn = p["mrn"].fillna("").astype(str).str.strip() != ""
d = p[has_mrn]
keep_id = d.groupby("mrn")["patient_id"].transform("min")                       # keep the lowest id
remap = dict(zip(d.loc[keep_id != d["patient_id"], "patient_id"], keep_id[keep_id != d["patient_id"]]))
d = d.groupby("mrn", as_index=False).agg({c: "first" for c in d.columns if c != "mrn"} | {"patient_id": "min"})
p = pd.concat([d, p[~has_mrn]], ignore_index=True)                              # no MRN: cannot match, keep
s["patient_id"] = s["patient_id"].replace(remap)                                # studies follow the merge
t_cond["patient_id"] = t_cond["patient_id"].replace(remap)                      # so do conditions
```

**Why:**
- **MRN is the identity:** the MRN is the hospital's identifier for a person. Merging by name plus date of birth would wrongly merge two different people with the same name.
- **`agg("first")` after sorting:** fills gaps in the kept record from its duplicates (for example a missing date of birth).
- **Remapping foreign keys:** keeps every study and condition attached to the surviving patient, so no history is lost (632 studies were re-pointed).
- **Patients without an MRN:** kept separately. Audit finding 3 found they were being dropped.

### 33.3 One report per study: rank and keep the best

```python
rank = r["status"].map({"final": 0, "amended": 1, "preliminary": 2})
r = r.assign(_rank=rank).sort_values(["study_id", "_rank", "signed_at"], ascending=[True, True, False])
r = r.drop_duplicates("study_id").drop(columns="_rank")
```

**Why:** a clinician wants the signed final report. Among several of equal status, the most recent one wins. Sort-then-`drop_duplicates` is the idiomatic, vectorised "top-1 per group".

### 33.4 Text and value repairs

```python
def _strip_html(text):
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)        # line breaks kept: they separate report sections
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)                                 # &amp; -> &
    return "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip())

pct = (conf > 1) & (conf <= 100)
conf = conf.where(~pct, conf / 100)                            # 88.0 -> 0.88
invalid = (conf < 0) | (conf > 1)
a["confidence"] = conf.where(~invalid)                         # impossible values cleared, row kept
bad_dob = p["date_of_birth"] > NOW
p.loc[bad_dob, "date_of_birth"] = pd.NaT                       # clear the value, keep the patient
```

**Why:**
- **`<br>` becomes a newline, not a space:** otherwise the section headings would end up mid-line and the chunker could not split on them.
- **Repair before removing:** a percentage confidence is obviously recoverable (88 → 0.88); only impossible values are cleared.
- **Keep the record when one field is bad:** a bad date of birth clears that field. Removing the patient would lose their whole history.

### 33.5 Logging every rule

```python
def note(msg, n):
    if n:
        log.append(f"{msg}: {n:,}")

note("studies: removed orphans (unknown patient)", n - len(s))
```

**Why:** each rule reports how many rows it touched (`reports/cleaning_log.md`). This makes cleaning auditable ("why did 558 studies disappear?") and catches regressions: a rule that suddenly touches 10 times more rows is visible.

---

## 34. Storage: parquet and PostgreSQL

**Method:** one table schema (`COLUMNS`) with two backends; PostgreSQL bulk load with `COPY`.
**Where:** `medrag/data/store.py` → `write_table()`, `read_table()`, `read_all()`, `coerce_types()`, `import_legacy_oads()`.

```python
if s.data_backend == "parquet":
    df.to_parquet(out / f"{name}.parquet", index=False)
    return

cur.execute(f"CREATE TABLE {schema}.{name} ({cols})")
with cur.copy(f"COPY {schema}.{name} ({', '.join(df.columns)}) FROM STDIN WITH (FORMAT csv)") as copy:
    for start in range(0, len(df), 50_000):                       # stream in 50k-row batches
        buf = io.StringIO()
        df.iloc[start:start + 50_000].to_csv(buf, index=False, header=False)
        copy.write(buf.getvalue())
if stage == "clean":
    cur.execute(f"ALTER TABLE {schema}.{name} ADD PRIMARY KEY ({key})")
    cur.execute(f"CREATE INDEX ON {schema}.{name} (patient_id)")
```

**Why:**
- **`COPY` is fast:** it is 10 to 100 times faster than row inserts (90k studies load in seconds), and batching keeps memory flat.
- **Loose types in raw:** raw tables use text and loose types so dirty values ("HIGH", "88.0", "") survive loading. Constraints and indexes are only added to the clean schema.
- **Parquet backend:** lets the whole project run offline with no database. It is columnar and compressed, and pandas reads it directly.
- **Nullable integer ids** (`Int64`): avoid `12.0` in CSV output.
- **Safe SQL:** schema and table names come from fixed lists in the code, never from user input.

---

## 35. Outside documents: loading, cleaning, linking

### 35.1 Loading

**Where:** `medrag/documents/loaders.py` → `load_openi()`, `load_mtsamples()`, `load_all()`.

```python
with tarfile.open(source) as t:
    t.extractall(xml_dir, filter="data")                    # blocks ../ paths and links inside the archive
root = ET.parse(f).getroot()                                # defusedxml: blocks XML bombs / external entities
parts = {el.get("Label"): (el.text or "").strip() for el in root.iter("AbstractText")}
text = "\n".join(f"{s}: {parts[s]}" for s in SECTIONS if parts.get(s))   # "FINDINGS: ...\nIMPRESSION: ..."
labels = [m.text for m in root.iter("major") if m.text]     # MeSH terms
```

**Why:**
- **Safety:** downloaded archives and XML are untrusted input. `filter="data"` and `defusedxml` stop path traversal and XML attacks.
- **Same format as the synthetic reports:** the text is rebuilt as `SECTION: text` lines, so the same chunker works on it.

### 35.2 Cleaning

**Where:** `medrag/documents/prepare.py` → `clean()`, `check()`, `_norm()`.

```python
HEADING_CSV = re.compile(r"\s*([A-Z][A-Z0-9 /&(),'-]{2,60}):,\s*")
d["text"] = d["text"].map(lambda t: HEADING_CSV.sub(lambda m: f"\n{m.group(1)}: ", t))   # "PLAN:,  x" -> "\nPLAN: x"
d["text"] = d["text"].str.replace(r"\bX{2,}\b", "[redacted]", regex=True)                 # Open-i "XXXX"
d = d[d["text"].str.split().str.len() >= 15]                                              # empty / very short

# duplicate transcriptions: same normalised text, only within MTSamples
d = d.assign(_n=d["text"].map(_norm).where(d["source"] == "mtsamples", d.index.astype(str)))
also = d.groupby(["source", "_n"])["specialty"].agg(lambda s: "; ".join(dict.fromkeys(s)))
d = d.drop_duplicates(["source", "_n"]).copy()
d["specialty"] = [also[(s, k)] for s, k in zip(d["source"], d["_n"])]
```

**Why:**
- **Heading repair:** puts each heading back on its own line, which section chunking needs (4,118 transcriptions).
- **`[redacted]`:** stops the LLM treating "XXXX" as a word or a name.
- **Dedupe key** (`_norm`: lowercase, letters and digits only): catches copies that differ only in spacing or punctuation.
- **Keeping all specialties:** the merged copy remembers every specialty it was filed under, so a cardiology question still finds a note also filed under surgery.
- **Open-i never deduplicated:** identical one-line reports there belong to different patients.

### 35.3 Linking documents to patients

**Where:** `prepare.link_to_patients()`.

```python
m, f = len(MALE_RE.findall(t)), len(FEMALE_RE.findall(t))
g = "male" if m > f else "female" if f > m else None                  # majority of gendered words
cand = np.ones(len(p), bool) if g is None else gender == g
if (a := AGE_RE.search(t)) and 0 < int(a.group(1)) < 100:              # "45-year-old"
    age = (np.datetime64(when) - dob).astype("timedelta64[D]").astype(int) / 365.25
    near = cand & (np.abs(age - int(a.group(1))) <= 2)                 # within 2 years at the document date
    if near.any():
        cand = near
out_pid.append(int(pids[rng.choice(np.flatnonzero(cand))]))
```

**Why:**
- **Synthetic link:** real documents have no link to synthetic patients. Matching gender and age makes the pairing clinically plausible (a "68-year-old gentleman" note goes to a man of about 68).
- **Vectorised:** the age filter runs over all 20,000 patients at once, so linking 6,211 documents takes seconds.
- **Reproducible:** seeded (13).

---

## 36. Chunking

**Methods:** structure-based (section) splitting → sentence fallback (recursive) → greedy fixed-size packing with a one-sentence overlap → contextual header → three chunk types → content-hash ids and metadata.
**Where:** `medrag/ingest/chunking.py` → `split_sections()`, `chunk_text()`, `chunk_document()`; `medrag/ingest/documents.py` → `build_chunks()`, `document_chunks()`, `_summary_line()`, `_hash()`.

No library splitter (LangChain, LlamaIndex, spaCy) is used: two regexes and a packing loop, about 50 lines.

| Technique | Method | Why |
|---|---|---|
| Section splitting | `SECTION_RE = r"^([A-Z][A-Z0-9 /&(),'-]{2,60}):"` with `re.M` | A report section is one complete clinical thought; a finding stays with its impression |
| Sentence fallback | `SENTENCE_RE = r"(?<=[.!?])\s+(?=[A-Z0-9])"`, only for sections over the budget | Smallest unit that still makes sense; keeps "3.5 cm" and "e.g. the" intact |
| Recursive | Section first, sentence only if needed | Keep the largest coherent unit that fits |
| Packing + overlap | Greedy, 220 words, carry the last unit if under a third of the budget | Fits the 512-token limit of bge-small and MedCPT; overlap keeps boundary text answerable |
| Contextual header | `chunk_document(header, body)`, the header on every chunk | Name, MRN, date and modality become searchable; the LLM knows whose record it is |
| Chunk types | Study / patient profile (300 words) / document | One unit per kind of question |
| Metadata + hash id | `chunk_id = sha1(text)[:20]` + filter fields | Filtering before search; incremental re-embedding |

```python
SECTION_RE  = re.compile(r"^([A-Z][A-Z0-9 /&(),'-]{2,60}):", re.M)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def split_sections(text):
    starts = [m.start() for m in SECTION_RE.finditer(text)]
    if not starts:
        return [text.strip()] if text.strip() else []
    if starts[0] != 0:
        starts.insert(0, 0)
    bounds = starts + [len(text)]
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]

def chunk_text(body, max_words=220, overlap_sentences=1):
    units = []
    for sec in split_sections(body):                         # level 1: sections
        if _words(sec) <= max_words:
            units.append(sec)
        else:
            units.extend(SENTENCE_RE.split(sec))             # level 2: sentences
    chunks, cur = [], []
    for u in units:                                          # greedy packing
        if cur and _words(" ".join(cur + [u])) > max_words:
            chunks.append("\n".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences and _words(cur[-1]) < max_words // 3 else []
        cur.append(u)
    if cur:
        chunks.append("\n".join(cur))
    return chunks

def chunk_document(header, body, max_words=220):
    budget = max(60, max_words - _words(header))            # the header counts against the budget
    return [f"{header}\n{c}" for c in chunk_text(body, budget)]
```

The three chunk types (`build_chunks`):

```python
# study: report + AI prediction
header = (f"Patient: {name} | MRN: {mrn} | Patient ID: {pid} | {gender}, age {age_at_study}\n"
          f"Study: {modality} {body_part} | Date: {date} | Accession: {acc} | Priority: {prio} | Report status: {status}\n"
          f"Indication: {indication}")
chunk_document(header, report_text + f"\nAI MODEL ({model}): predicted '{label}' (confidence {conf:.2f}).", 220)

# patient profile: a summary built from data, 300-word budget
body = ("HISTORY: Chronic conditions: ...\nHISTORY: 12 imaging studies from ...: X-ray x5, CT x4 ...\n"
        "HISTORY: 2 uploaded documents: ...\nFINDINGS: Most recent studies first. " + timeline_of_40_impressions)
chunk_document(profile_header, body, 220 + 80)

# document: Open-i / MTSamples, header with title, specialty, source and the MTSamples one-line summary
chunk_document(doc_header, text, 220)          # Open-i also tagged modality=XR, body_part=chest
```

Each chunk row:

```python
{"chunk_id": sha1(text)[:20], "doc_id": "study:123", "doc_type": "study", "part": i,
 "patient_id", "study_id", "mrn", "patient_name", "gender", "modality", "body_part", "priority",
 "study_date", "status", "ai_label", "ai_confidence", "impression", "text", "source", "title", "specialty"}
```

Result: 121,335 chunks (89,014 study, 19,923 profile, 3,880 Open-i, 8,518 MTSamples). Long transcriptions average 3.7 chunks; reports fit in one.

**Not used, and why:**
- **Fixed character or token windows:** they cut sections in the middle.
- **Semantic chunking by embedding similarity:** the text is already structured, so it would only add cost.
- **LLM-based chunking:** too slow for 121k chunks on a CPU.
- **Parent–child chunks:** most reports are already a single chunk.

The 220-word size was chosen from the model limit, not tuned (see section 26).

---

## 37. Embeddings

**Method:** a provider interface; L2-normalised float32 vectors; model-specific query/document prefixes; batched encoding.
**Where:** `medrag/providers/embeddings.py` → `Embedder`, `SentenceTransformerEmbedder`, `OllamaEmbedder`, `OpenAIEmbedder`, `GeminiEmbedder`, `HashEmbedder`, `get_embedder()`, `PREFIXES`.

```python
PREFIXES = {"bge": ("Represent this sentence for searching relevant passages: ", ""),
            "e5": ("query: ", "passage: "),
            "nomic-embed-text": ("search_query: ", "search_document: ")}

def _normalise(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

class Embedder:
    def embed_documents(self, texts):
        return _normalise(self._embed([self.d_prefix + t for t in texts]))
    def embed_query(self, text):
        return _normalise(self._embed([self.q_prefix + text]))[0]

class SentenceTransformerEmbedder(Embedder):
    def _embed(self, texts):
        return self.st.encode(texts, batch_size=get_settings().embed_batch_size, show_progress_bar=False)
```

**Why:**
- **Normalised vectors:** the inner product then equals cosine similarity, so FAISS `IndexFlatIP` and a NumPy `@` give the same, comparable scores.
- **Query/document prefixes:** bge, e5 and nomic were trained with an instruction on one side. Leaving it out measurably lowers retrieval quality.
- **One interface:** switching to OpenAI, Gemini or Ollama embeddings is a `.env` change, and each model gets its own index folder.
- **`HashEmbedder`:** needs no download, so the tests and smoke runs work offline and are deterministic.
- **Why `bge-small-en-v1.5`:**
  - 384 dimensions, about 33M parameters;
  - strong for its size on retrieval benchmarks;
  - fast enough on CPU: 121k chunks in about 3 hours for the first build, about 3.5 chunks per second afterwards;
  - small vectors: 121k × 384 × 4 bytes = 186 MB.
- **Alternatives:**
  - larger models (bge-large, e5-large) are 3 to 10 times slower on CPU;
  - medical bi-encoders (MedCPT query/article encoders) could be tried, but the medical gain was put in the reranker instead (section 46).

---

## 38. Vector database: FAISS

**Method:** FAISS `IndexFlatIP` (exact inner-product search) under 150,000 chunks, `IndexHNSWFlat` above that; **filtered search done exactly in NumPy** over the memory-mapped vectors.
**Where:** `medrag/index/store.py` → `build()` (index creation), `Index.__init__()` (loading), `Index.dense_search()`; `HNSW_THRESHOLD = 150_000`.

```python
if len(chunks) > HNSW_THRESHOLD:
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)   # M = 32 links per node
    index.hnsw.efConstruction = 80
else:
    index = faiss.IndexFlatIP(dim)                                     # exact: brute force, 100% recall
index.add(vecs)
faiss.write_index(index, str(f["dense"]))

# loading
self.vectors = np.load(f["vectors"], mmap_mode="r")                    # memory-mapped, not read into RAM
self.dense = faiss.read_index(str(f["dense"]))
if hasattr(self.dense, "hnsw"):
    self.dense.hnsw.efSearch = 128

def dense_search(self, qvec, k, rows=None):
    if rows is not None:                                               # a metadata filter is active
        if len(rows) == 0:
            return []
        scores = np.asarray(self.vectors[rows]) @ qvec                 # exact search inside the subset
        top = np.argsort(-scores)[:k]
        return [(int(rows[i]), float(scores[i])) for i in top]
    scores, ids = self.dense.search(qvec[None, :].astype(np.float32), k)   # whole index: FAISS
    return [(int(i), float(sc)) for i, sc in zip(ids[0], scores[0]) if i >= 0]
```

**Why FAISS, not a vector database server:**

| Option | Why not chosen here |
|---|---|
| Chroma, Qdrant, Weaviate, Milvus | An extra server or process to run and secure; the project must run offline on a laptop; 121k vectors fit easily in memory |
| Pinecone and other cloud vector DBs | Sends patient text to a third party; costs money; needs a network |
| pgvector | A good fit when PostgreSQL is used, but the parquet mode must work with no database |
| **FAISS** | A library, not a server; the fastest exact and approximate search on CPU; saved as one file; the industry standard |

**Why exact search (`IndexFlatIP`):**
- **Fast enough:** at 121k × 384, brute force takes about 10–20 ms per query on CPU, and recall is 100%.
- **Only when needed:** HNSW is approximate (it may miss some neighbours) and is only switched on above 150k chunks, where brute force starts to cost.

**Why filtered search in NumPy, not FAISS:**
- **Most questions are filtered:** once a question names a patient or modality, the search space shrinks to dozens or a few thousand rows.
- **FAISS without filters would miss results:** searching all 121k vectors and filtering afterwards ("post-filtering") can return zero matching rows in the top 60.
- **A dot product over the subset is exact and fast:** about 1 ms for 1,000 rows.
- **Memory-mapped vectors** (`mmap_mode="r"`): the OS loads only the pages that are touched, so start-up is quick and several processes share the same memory.

---

## 39. Keyword index: BM25

**Method:** `bm25s` (sparse BM25, k1 = 1.2, b = 0.75) with English stop words and the Snowball stemmer (`PyStemmer`); filtered scoring by array indexing.
**Where:** `medrag/index/store.py` → `tokenize()`, `build()` (indexing), `Index.sparse_search()`.

```python
STEMMER = Stemmer.Stemmer("english")

def tokenize(texts):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=False)

bm25 = bm25s.BM25(k1=1.2, b=0.75)
bm25.index(tokenize(texts), show_progress=False)
bm25.save(str(f["bm25"]))

def sparse_search(self, query, k, rows=None):
    toks = tokenize([query])
    inv = {v: key for key, v in toks.vocab.items()}
    words = [inv[t] for t in toks.ids[0] if inv[t] in self._vocab]      # drop words the corpus never saw
    if not words:
        return []
    scores = self.bm25.get_scores(words)                                # one score per chunk
    if rows is not None:
        sub = scores[rows]
        top = np.argsort(-sub)[:k]
        return [(int(rows[i]), float(sub[i])) for i in top if sub[i] > 0]
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]         # O(n) top-k, then sort only k
    top = top[np.argsort(-scores[top])]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
```

**Why BM25 alongside embeddings:**
- **Exact tokens:** medical text is full of identifiers and precise terms (`MRN0001234`, `ACC000012345`, `LI-RADS 5`, `BI-RADS 4`, "cholelithiasis"). Embeddings blur these; BM25 matches them exactly.
- **Measured:** BM25 alone had the **best** documents MRR (0.837), while dense alone had 0.621 (`doc_semantic` 0.27). Dense alone was better on paraphrases. Hybrid covers both (section 45).
- **Stemming:** "fractures" and "fractured" match "fracture". Stop words remove noise words.
- **k1 1.2 and b 0.75:** the standard defaults, which work well for documents of mixed length.

**Why `bm25s`:**
- It uses sparse SciPy matrices and is 100 to 500 times faster than `rank_bm25`.
- It saves to disk and needs no server (unlike Elasticsearch or OpenSearch).

**`argpartition`:** finds the top k in O(n) instead of sorting all 121k scores.

---

## 40. Index build: incremental and versioned

**Method:** content-hash reuse of vectors; versioned file names; atomic manifest swap with `os.replace`; deferred clean-up.
**Where:** `medrag/index/store.py` → `build()`, `_files()`, `_remove_old_builds()`, `Index.__init__()`.

```python
# 1. reuse the vectors of unchanged chunks
prev = _files(path, json.loads((path / "manifest.json").read_text()).get("version"))
old_ids = pd.read_parquet(prev["chunks"], columns=["chunk_id"])["chunk_id"]
old = dict(zip(old_ids, np.load(prev["vectors"])))
todo = [i for i, cid in enumerate(chunks["chunk_id"]) if cid not in old]          # embed only these
for start in tqdm(range(0, len(todo), batch)):                                     # batches of 256
    idx = todo[start:start + batch]
    vecs[idx] = embedder.embed_documents([texts[i] for i in idx])

# 2. write a new versioned build
version = datetime.now().strftime("%Y%m%d%H%M%S%f")
def _files(path, version):
    sfx = f"-{version}" if version else ""                                         # no version: an old unversioned index
    return {k: path / f"{k}{sfx}{ext}" for k, ext in
            (("chunks", ".parquet"), ("vectors", ".npy"), ("dense", ".faiss"), ("bm25", ""))}

# 3. switch over atomically, then clean up
tmp = path / "manifest.json.tmp"
tmp.write_text(json.dumps(manifest, indent=2))
os.replace(tmp, path / "manifest.json")      # atomic rename: readers see the old build or the new one
_remove_old_builds(path, version)            # OSError ignored: a file still open (Windows) is removed next build
```

**Why:**
- **Incremental:** embedding is the slow step (hours on CPU). Hashing each chunk's text means only new or changed chunks are embedded. Adding MTSamples embedded 10,749 chunks in 50 minutes instead of all 121,335 in about 3 hours.
- **Versioned files:** the running app memory-maps `vectors.npy`. Overwriting it in place could crash the app or give it half-written data.
- **Atomic manifest swap:** `os.replace` is atomic on Windows and POSIX, so a reader opens either the complete old build or the complete new one, never a mix (audit finding 6).
- **Readable manifest:** it records model, dimension, embedded vs reused counts and build time. The app shows it and caches on `built_at`, so a new build is picked up automatically.

---

## 41. Live sync: watch

**Method:** polling a cheap change signature; re-run clean + incremental index when it changes.
**Where:** `medrag/__main__.py` → `_raw_signature()`, `cmd_watch()`.

```python
def _raw_signature():
    if s.data_backend == "parquet":
        sig = sorted((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in s.raw_dir.glob("*.parquet"))
    else:
        cur.execute("SELECT relname, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables "
                    "WHERE schemaname = %s ORDER BY relname", (store.SCHEMAS["raw"],))
        sig = cur.fetchall()
    return hashlib.md5(json.dumps(sig, default=str).encode()).hexdigest()

while True:
    sig = _raw_signature()
    if sig != last:
        cmd_clean(a)
        cmd_index(a)
        last = sig
    time.sleep(a.interval)                                     # default 30 s
```

**Why:**
- **Cheap signature:** it reads statistics, not data, so polling every 30 seconds costs nothing.
- **Catches updates:** PostgreSQL's insert, update and delete counters catch an update that leaves the row count unchanged; the old row-count method missed it (audit finding 4).
- **Polling instead of triggers:** polling needs no database changes (triggers, logical replication, LISTEN/NOTIFY) and works the same for parquet.
- **Cost of a sync:** re-indexing is incremental, so a sync after a few new reports embeds only those.

---

## 42. Query parsing

**Method:** regular expressions + dictionaries + fuzzy name matching (`rapidfuzz`) into a `Filters` dataclass.
**Where:** `medrag/retrieval/query_parser.py` → `Filters`, `QueryParser.__init__()`, `_patients()`, `parse()`.

```python
# exact identifiers first
for mrn in re.findall(r"\b(MRN\d{4,}|LEG\d{4,})\b", q, re.I): ...
for acc in re.findall(r"\b(ACC\d{4,})\b", q, re.I): ...

# names: every 3- then 2-word window, exact match first, then fuzzy
for n in (3, 2):
    for i in range(len(words) - n + 1):
        cand = " ".join(words[i:i + n]).lower()
        if cand in self.by_name:
            best = (100.0, cand)
            break
        hit = process.extractOne(cand, self.names, scorer=fuzz.ratio, score_cutoff=90)   # typo-tolerant
ids.extend(self.by_name[best[1]])          # a shared name returns ALL matching patients

# short modality codes count only in capitals
if syn in ("us", "mr", "mg", "pet") and not re.search(rf"\b{syn.upper()}\b", q):
    continue

# dates
if m := re.search(r"\b(?:last|past|previous)\s+(\d+|a|one|two|three|six)\s+(day|week|month|year)s?\b", ql): ...
elif m := re.search(r"\bbetween\s+(20\d\d)\s+and\s+(20\d\d)\b", ql): ...

f.aggregate = bool(re.search(r"\b(how many|count|number of|total|percentage|proportion|...)\b", ql))
f.latest    = bool(re.search(r"\b(latest|most recent|newest|last (study|scan|report|imaging|visit))\b", ql))
```

**Why:**
- **Deterministic:** the parser is fast (under 1 ms besides fuzzy matching), testable, and adds no LLM call before retrieval. Using an LLM to extract filters would add seconds and could invent values.
- **Fuzzy matching with `fuzz.ratio ≥ 90`:** tolerates one or two typos in a name, which the `hard_typo_name` questions test: MRR 1.00. It uses C++ code, so scanning 20,000 names is fast.
- **Shared names return everyone:** the answer can then list each patient separately instead of silently picking one.
- **Unknown names:** an `unresolved_patient` (a capitalised name that matches nobody) short-circuits to "not found" (section 49).
- **Capitals rule for short codes:** stops "tell us" being read as ultrasound (audit finding 2).

---

## 43. Query expansion

**Method:** dictionary-based expansion. Lay phrases are mapped to a canonical finding label plus the radiologist's wording, and appended to the query; the longest match wins, with no overlaps.
**Where:** `medrag/retrieval/expansion.py` → `SYNONYMS`, `REPORT_TERMS`, `IMPRESSION_PATTERNS`, `detect_findings()`, `expand()`, `impression_pattern()`.

```python
SYNONYMS = {"disc herniation": ["slipped disc", "herniated disc", "disc prolapse", ...],
            "intracranial haemorrhage": ["brain bleed", "bleeding in the brain", ...], ...}   # 32 findings
REPORT_TERMS = {"disc herniation": "disc extrusion nerve root compression",
                "gallstones": "cholelithiasis", ...}                                       # how reports phrase it

_TERMS = sorted(((syn, label) for label, syns in SYNONYMS.items() for syn in syns + [label]),
                key=lambda x: -len(x[0]))                                                  # longest first

def detect_findings(query):
    for phrase, label in _TERMS:
        for m in re.finditer(rf"(?<![a-z]){re.escape(phrase.lower())}(?![a-z])", ql):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue                                                                   # no overlapping matches
            ...

def expand(query):
    for phrase, label in detect_findings(query):
        terms = [label] + ([REPORT_TERMS[label]] if label in REPORT_TERMS else [])
        new = [t for t in terms if t.lower() not in query.lower()]
        extra += new
        notes.append(f"'{phrase}' means {label} (reports may say: {REPORT_TERMS.get(label)})")
    return (f"{query} ({'; '.join(extra)})" if extra else query), notes
```

Example: `"Has X ever had a slipped disc?"` → `"Has X ever had a slipped disc? (disc herniation; disc extrusion nerve root compression)"`.

**Why:**
- **The vocabulary gap:** patients and clinicians say "slipped disc", reports say "disc extrusion". BM25 cannot match words that are not there, and small embedding models only partly bridge it.
- **Measured gain:** about +0.1 MRR on hard questions (0.793 → 0.905).
- **Dictionary, not an LLM rewrite** (HyDE, query rewriting): it is instant, deterministic and cannot invent clinical terms.
- **Honest measurement:** the test set includes phrasings not in the dictionary. Recall there is still 92.6%, thanks to dense search.
- **Longest match, no overlaps:** "lung mass" is not also matched as "mass".
- **Notes to the LLM:** the LLM is told the mapping, so it can explain "slipped disc = disc extrusion" in the answer.
- **Reused for counting:** `IMPRESSION_PATTERNS` gives exact cohort counts by the radiologist's wording (section 48).

---

## 44. Metadata filtering with relaxation

**Method:** pandas boolean mask from the parsed filters; if nothing matches, drop filters one at a time in a fixed order of importance.
**Where:** `medrag/retrieval/retriever.py` → `FILTER_ORDER`, `Retriever._mask()`, `Retriever.filter_rows()`.

```python
FILTER_ORDER = ["abnormal", "documents", "priority", "dates", "body_part", "modality", "patient"]   # first relaxed first

def _mask(self, f, use):
    m = pd.Series(True, index=self.meta.index)
    if "patient" in use and f.patient_ids:
        m &= self.meta["patient_id"].isin(f.patient_ids)
    if "modality" in use and f.modality:
        m &= self.meta["modality"] == f.modality
    if "dates" in use and f.date_from is not None:
        m &= self.meta["study_date"] >= f.date_from
    if "documents" in use and f.documents:
        m &= self.meta["doc_type"].isin(["document", "patient"])
    ...
    return m

def filter_rows(self, f, notes):
    use = set(FILTER_ORDER)
    while use:
        m = self._mask(f, use)
        if m.any():
            return None if m.all() else np.flatnonzero(m.to_numpy())
        dropped = FILTER_ORDER[len(FILTER_ORDER) - len(use)]
        use.discard(dropped)
        notes.append(f"No records match every filter; relaxed the '{dropped}' filter.")
```

**Why:**
- **Pre-filtering:** filtering *before* search is the single largest gain measured. MRR is about 0.5 without filters and 0.9–1.0 with them, because a patient question searches that patient's 5–50 chunks instead of 121,335.
- **Relaxation instead of returning nothing:** "Asha Rao's MRI" when she has no MRI still shows her other studies, with a note saying so.
- **The order** keeps the most important constraint, the patient, until last.
- **Counts never relax:** they always use every filter (section 48).
- **Fast:** a vectorised mask over 121k rows takes milliseconds.

---

## 45. Hybrid search and Reciprocal Rank Fusion

**Method:** run dense (FAISS/NumPy) and sparse (BM25) search on the same filtered rows, 60 candidates each, and fuse them with Reciprocal Rank Fusion, k = 60.
**Where:** `medrag/retrieval/retriever.py` → `rrf()`, `Retriever.search()`.

```python
def rrf(rankings, k=60):
    fused = {}
    for name, ranking in rankings.items():
        for rank, (row, _) in enumerate(ranking):
            h = fused.setdefault(row, Hit(row, 0.0))
            h.score += 1.0 / (k + rank + 1)                 # depends only on rank, not on the raw score
            h.sources.append(name)
    return sorted(fused.values(), key=lambda h: -h.score)

rankings["dense"] = self.index.dense_search(self.embedder.embed_query(q), n_cand, rows)
rankings["bm25"]  = self.index.sparse_search(q, n_cand, rows)
hits = rrf(rankings)
```

**Why:**
- **Complementary strengths:** dense finds meaning (paraphrases); BM25 finds exact tokens (IDs, rare terms).
- **Measured:** on documents, dense alone scored 0.621 and BM25 alone 0.837. On hard questions, BM25 alone scored 0.865 and dense alone 0.913. Neither single method is safe everywhere; hybrid plus rerank is the only configuration that is strong on all three sets (1.000 / 0.905 / 0.808).
- **Why RRF instead of a weighted score sum:** cosine scores (0 to 1) and BM25 scores (0 to about 30) are on different scales that change per query. RRF uses only ranks, so there is no normalisation and no weight to tune.
- **Why k = 60:** the standard value from the original RRF paper; it stops the top-ranked item of one list from dominating.
- **Agreement shows:** a chunk found by both methods gets two contributions and rises; `via = "dense+bm25"` shows this in the UI.

---

## 46. Reranking: MedCPT cross-encoder

**Method:** a cross-encoder scores `(query, chunk)` pairs for the top 30 fused candidates; its order is fused again with the hybrid order by RRF (`fuse`), or replaces it (`replace`).
**Where:** `medrag/retrieval/retriever.py` → `Reranker`, `Retriever.reranker` (lazy property), the rerank block in `search()`.

```python
class Reranker:
    _models = {}                                                  # one copy per process, shared by all retrievers
    def __init__(self, model):
        if model not in self._models:
            self._models[model] = CrossEncoder(model, max_length=512)
        self.ce = self._models[model]
    def score(self, query, texts):
        return np.asarray(self.ce.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False))

pool = hits[: max(s.rerank_pool, k)]                              # 30 candidates
scores = self.reranker.score(q, self.meta["text"].iloc[[h.row for h in pool]].tolist())
if self.rerank_blend == "fuse":
    by_rerank = [(pool[i].row, float(scores[i])) for i in np.argsort(-scores)]
    fused = rrf({"hybrid": [(h.row, h.score) for h in pool], "rerank": by_rerank})
else:                                                             # replace
    hits = sorted(pool, key=lambda h: -h.score)                   # after setting h.score = reranker score
```

**Why a cross-encoder:**
- **Reads query and chunk together:** a bi-encoder embeds each text separately. A cross-encoder reads both at once, so it sees exactly which words of the question match which words of the chunk. This is more accurate, but too slow for 121k chunks, hence only the top 30.

**Why MedCPT** (`ncbi/MedCPT-Cross-Encoder`):
- **Medical training:** it is trained on PubMed search logs.
- **Measured on hard questions:**

  | Reranker | Hard MRR |
  |---|---:|
  | `bge-reranker-base` (general, web-trained) | 0.845 |
  | No reranker | 0.876 |
  | MiniLM | 0.874–0.885 |
  | **MedCPT** | **0.905–0.915** |

  The general reranker made results *worse* and was slower.

**Why `fuse`:**
- **Safer:** the reranker was trained on a different domain (PubMed abstracts, not reports), so fusing keeps what the first stage got right. `replace` was slightly better on hard questions (0.915) but worse on documents (0.793 vs 0.808).
- **Within the noise:** the top three configurations are inside each other's confidence intervals.

**Other details:**
- **Pool of 30:** a CPU latency trade-off of about 2–3 s per question. MiniLM takes 0.7 s if speed matters more.
- **Lazy loading and class-level cache:** the model loads once and only if used. `RERANK_MODEL=none` switches it off.

---

## 47. Pinning profile and latest studies

**Method:** for patient questions, add the patient's profile chunk; for "latest" questions, also add their 3 most recent studies. Then fill the remaining slots from search.
**Where:** `medrag/retrieval/retriever.py` → end of `search()`.

```python
if f.patient_ids and use_filters:
    prof = self.meta.index[(self.meta["doc_type"] == "patient") & self.meta["patient_id"].isin(f.patient_ids)
                           & (self.meta["part"] == 0)]
    pinned += [Hit(int(r), float("inf"), ["profile"]) for r in prof]
    if f.latest and rows is not None:
        studies = self.meta.iloc[rows]
        studies = studies[studies["doc_type"] == "study"].sort_values("study_date", ascending=False)
        pinned += [Hit(int(r), float("inf"), ["latest"]) for r in studies.index[:3]]
hits = pinned + [h for h in hits if h.row not in seen][: max(k - len(pinned), k // 2)]
```

**Why:**
- **"Latest" is not a similarity question:** "what was the most recent scan" is answered by sorting dates, not by text similarity. Pinning guarantees the right study is there: `latest` MRR is 1.00.
- **The profile gives the whole history:** conditions, counts, timeline. A clinician would always want it in front of them.
- **At least half the slots stay for search,** so pinning never crowds out the evidence.
- **Scoring is fair:** evaluation removes the pinned profile before scoring, unless the profile is the gold document (section 52), so pinning cannot inflate retrieval metrics.

---

## 48. Computed facts

**Method:** exact pandas aggregations over the **full filtered metadata** (not the 8 retrieved chunks), passed to the LLM as text.
**Where:** `medrag/rag/answer.py` → `computed_facts()`; `Retriever.finding_mask()`.

```python
scope = meta[retriever._mask(f, set(FILTER_ORDER))]                  # ALL filters, never relaxed
st = scope[(scope["doc_type"] == "study") & (scope["part"] == 0)]     # part 0: each study counted once
if f.findings:
    st = st[retriever.finding_mask(f.findings).loc[st.index]]        # radiologist's impression, not the AI label
lines = [f"Filters applied: {f.describe()}",
         f"Matching studies: {len(st):,} across {st['patient_id'].nunique():,} patients",
         "By modality: ...", "By priority: ...", f"Studies with abnormal findings: {len(abnormal):,} (...%)"]
if len(f.patient_ids) > 1:
    lines.append(f"WARNING: {len(f.patient_ids)} different patients match this name. "
                 "Answer separately for each patient (give MRN):")

def finding_mask(self, labels):
    pattern = "|".join(f"(?:{impression_pattern(lbl)})" for lbl in labels)   # e.g. "disc extrusion"
    return self.meta["impression"].fillna("").str.contains(pattern, case=False, regex=True)
```

**Why:**
- **LLMs cannot count what they cannot see:** "How many stat MRIs in 2023?" might match 600 studies, but the LLM only gets 8. Computed facts give it the exact number from all of them.
- **All filters always:** the count uses every filter even when search relaxed one. "How many MRIs does X have" for a patient with no MRI is 0, not the count of other studies (audit finding 1, tested).
- **Radiologist's impression, not the AI label:** the AI model is wrong 14% of the time, so counting by impression is ground truth.
- **`part == 0`:** a report split into several chunks is still one study.
- **Ambiguous names:** the per-patient breakdown lets the LLM answer "two patients named X: MRN... has 5 studies, MRN... has 3".
- **Patient timeline:** up to 60 rows, so "has X ever had..." can be answered over the whole history, not just 8 chunks.

---

## 49. Prompt and guardrails

**Method:** a fixed system prompt with rules; a structured user message (question, notes, computed facts, numbered sources); short-circuit for unknown patients.
**Where:** `medrag/rag/answer.py` → `SYSTEM`, `build_prompt()`, `RAG.prepare()`, `RAG.stream()`, `RAG.ask()`.

```python
SYSTEM = """You are a clinical assistant answering questions about a hospital's radiology records.
Rules:
- COMPUTED FACTS are exact counts from the full database ... answer directly from them ...
- Otherwise use only the SOURCES. Never invent patients, dates, findings or numbers.
- Cite sources inline as [S1], [S2] after each claim taken from them.
- Only if neither the COMPUTED FACTS nor the SOURCES contain the answer, say
  "The records do not contain this information."
- Be concise and clinical ... dates as YYYY-MM-DD.
- This is decision support, not a diagnosis; do not give treatment advice beyond what reports recommend."""

def build_prompt(question, r, facts):
    sources = "\n\n".join(f"[S{i}] ({row.doc_id})\n{row.text}" for i, row in enumerate(r.hits.itertuples(), 1))
    parts = [f"QUESTION: {question}"]
    if r.notes: parts.append("NOTES: " + " ".join(r.notes))
    if facts:   parts.append(f"COMPUTED FACTS:\n{facts}")
    parts.append(f"SOURCES:\n{sources or '(none)'}")
    return "\n\n".join(parts)

if r.filters.unresolved_patient:                        # no LLM call at all
    text = f"No patient matching '{r.filters.unresolved_patient}' was found in the records."
```

**Why:**
- **Grounding rules:** "use only the SOURCES", a fixed "not found" sentence, and citations are standard RAG guardrails against hallucination.
- **Checkable "not found":** the fixed sentence makes abstaining detectable by `ABSTAIN_RE` in the evaluation.
- **Numbered sources with `doc_id`:** let the user check each claim. The UI shows the same numbers.
- **Unknown patients short-circuit:** the LLM never sees an empty context it could fill with a made-up patient, and it is faster.
- **Low temperature (0.1):** answers are close to deterministic.

---

## 50. LLM providers, streaming and retries

**Method:** an abstract `LLM` with `stream()`; `complete()` joins the stream and retries on rate-limit errors with linear back-off; a registry keyed by provider name.
**Where:** `medrag/providers/llms.py` → `LLM`, `OllamaLLM`, `AnthropicLLM`, `OpenAILLM`, `GroqLLM`, `OpenRouterLLM`, `GeminiLLM`, `BedrockLLM`, `PROVIDERS`, `get_llm()`, `get_judge()`.

```python
def complete(self, system, user, max_tokens=1500, retries=4):
    for attempt in range(retries + 1):
        try:
            return "".join(self.stream(system, user, max_tokens))
        except Exception as e:
            msg = str(e)
            if attempt == retries or not any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "rate limit", "Rate limit", "ThrottlingException")):
                raise
            time.sleep(15 * (attempt + 1))                                   # 15, 30, 45, 60 s

class OllamaLLM(LLM):
    def stream(self, system, user, max_tokens=1500):
        body = {"model": self.model, "stream": True, "think": False, "messages": [...],
                "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": max_tokens}}
        with httpx.stream("POST", f"{url}/api/chat", json=body, timeout=600) as r:
            for line in r.iter_lines(): ...                                   # also strips any <think>...</think>

class BedrockLLM(LLM):
    def stream(self, system, user, max_tokens=1500):
        resp = self.client.converse_stream(modelId=self.model, system=[{"text": system}],
                                           messages=[{"role": "user", "content": [{"text": user}]}],
                                           inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1})
        for ev in resp["stream"]:
            if text := ev.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                yield text

def get_llm(spec=None):          # "ollama:qwen2.5:3b" -> split on the FIRST colon only
    provider, model = spec.split(":", 1) if spec else (s.llm_provider, s.llm_model)
    return PROVIDERS[provider](model)
```

**Why:**
- **Streaming everywhere:** the UI shows text as it arrives. On CPU, a 60-second answer is usable when you watch it appear.
- **One interface:** comparing local and cloud models is a `--llms` list, with the same prompts and the same scoring.
- **Retries only on rate limits:** free tiers (Gemini, Groq) return 429 often. Retrying anything else would hide real bugs.
- **`num_ctx 8192`:** fits the system prompt, computed facts (up to a 60-row timeline) and 8 sources.
- **`think: False` and stripping `<think>`:** Qwen3 would otherwise print its reasoning into the answer.
- **Bedrock through the Converse API:** one code path for every Bedrock model family, with credentials from the AWS chain (never `.env`).
- **`split(":", 1)`:** model names contain colons (`qwen2.5:3b`, `us.amazon.nova-lite-v1:0`).

---

## 51. Evaluation: building the test set

**Method:** generate questions from the index metadata with known gold documents and answers; seeded sampling; three difficulty tiers; deliberate traps (shared names, typos, unknown patients, lay wording not in the dictionary).
**Where:** `medrag/eval/testset.py` → `build()` (easy), `build_hard()`, `build_documents()`, `keywords()`, `review_sheet()`, `TEST_PHRASES`.

```python
# only unique names for lookups, so the gold answer is unambiguous
names = meta[meta["doc_type"] == "patient"].drop_duplicates("patient_id")["patient_name"]
unique_name = set(names[~names.duplicated(keep=False)])

# semantic: day + modality + finding must identify exactly one study
abnormal = abnormal.assign(_key=day + abnormal["modality"] + abnormal["ai_label"].fillna(""))
unique_case = abnormal[~abnormal["_key"].duplicated(keep=False)]

# count questions: the answer is computed from the data
n = int(((studies["study_date"].dt.year == y) & (studies["modality"] == mod) & (studies["priority"] == pr)).sum())
items.append({"type": "count_cohort", "question": f"How many {pr} priority {mod} studies were performed in {y}?",
              "gold_docs": [], "number": n, "answer": str(n)})

# hard: lay phrase, and whether the expansion dictionary knows it
def phrase(label):
    p = rnd.choice(TEST_PHRASES[label])                                  # "water on the lung"
    return p, any(lbl == label for _, lbl in detect_findings(p))         # in_dictionary flag

# hard_negative: pick a finding the patient NEVER had
if (g["finding"] == label).any():
    continue

# hard_typo_name: drop one inner letter of the surname
typo = f"{first} {last[:i] + last[i + 1:]}"
```

**Why:**
- **Built from the data:** every question has an exact right answer and gold document, at any scale and without hand-labelling. 313 questions are rebuilt in seconds.
- **Seeded:** the same questions come back every run, so runs and models compare fairly.
- **Traps:** shared names, typos, unknown patients and negatives test failure modes that a plain "lookup" set never would.
- **`in_dictionary` flag:** shows how much of the hard-question success is the expansion dictionary and how much is the search itself (100% vs 92.6% recall).
- **Content-only document questions:** `doc_semantic` and `doc_transcription_semantic` have no name or date in the question, so they test pure retrieval.
- **Human review sheet:** a person can verify a sample of the expected answers (`golden_review.csv`), guarding against bugs in the generator itself.

---

## 52. Evaluation: retrieval metrics

**Method:** Hit@k, MRR, Recall@k and nDCG@k at the **document** level, over de-duplicated rankings, with pinned profiles removed.
**Where:** `medrag/eval/run.py` → `retrieval_metrics()`, `eval_retrieval()`, `Config`.

```python
def retrieval_metrics(ranked_docs, gold, k):
    ranked = list(dict.fromkeys(ranked_docs))[:k]                  # chunks -> unique documents, order kept
    rel = [1 if d in gold else 0 for d in ranked]
    first = next((i for i, r in enumerate(rel) if r), None)
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return {"hit": float(first is not None), "mrr": 1 / (first + 1) if first is not None else 0.0,
            "recall": sum(rel) / len(gold), "ndcg": dcg / idcg if idcg else 0.0}

if not any(g.startswith("patient:") for g in it["gold_docs"]):
    ranked = ranked[ranked["via"] != "profile"]                     # don't let pinning inflate the score
```

**Why each metric:**
- **Hit@8:** does the LLM see the right evidence at all? This sets the ceiling for answer quality.
- **MRR:** how high is the first right document? Small LLMs pay most attention to the first sources.
- **Recall@8:** for cohort questions with 1–3 right documents, did we find all of them?
- **nDCG@8:** rewards putting all right documents near the top. It is the standard ranking metric.
- **Document level:** one report split into 3 chunks should count once, so chunks are de-duplicated to documents.

**Configurations:**
- **`Config`:** bundles a retriever, a mode and flags, so ablations run on exactly the same questions: no filters, no expansion, dense, BM25, hybrid, rerankers, fuse vs replace.

---

## 53. Evaluation: answer checks

**Method:** deterministic regex and substring checks matched to each question type, collapsed to one `auto_correct` pass/fail.
**Where:** `medrag/eval/run.py` → `score_answer()`, `_fact_hit()`, `ABSTAIN_RE`, `NEGATIVE_RE`, `eval_generation()`.

```python
ABSTAIN_RE  = re.compile(r"(no patient matching|not (be )?found|do(es)? not contain|no record|not in the records|no information|unable to find)", re.I)
NEGATIVE_RE = re.compile(r"\b(no|not|never|none|did not|has not|hasn't|without)\b", re.I)

def score_answer(it, text):
    low = text.lower()
    if it.get("facts"):
        row["fact_recall"] = sum(_fact_hit(f, low) for f in it["facts"]) / len(it["facts"])
    if "number" in it:
        row["number_correct"] = float(bool(re.search(rf"(?<![\d,.])({n:,}|{n})(?![\d,]|\.\d)", text)))
    if it.get("expect_no"):
        row["negative_correct"] = float(bool(NEGATIVE_RE.search(text)) and not low.lstrip().startswith("yes"))
    row["abstain_correct"] = float(bool(ABSTAIN_RE.search(text)) == bool(it.get("abstain")))
    # one pass/fail per question, from the check that fits it:
    # number -> number_correct; negative -> negative_correct; not-found -> abstain_correct; else all facts present

def _fact_hit(fact, text):
    return any(alt.strip().lower() in text for alt in fact.split("|"))   # "ct|computed tomography"
```

**Why:**
- **Free, instant, reproducible:** running a judge model on every answer costs money and quota, and judges disagree with themselves between runs.
- **Exact number regex:** accepts `1,234` or `1234` but not `11234` or `12.3`. Lookarounds stop partial matches.
- **`abstain_correct` cuts both ways:** it penalises saying "not found" when the answer exists, and inventing when it does not.
- **Alternatives with `|`:** CT or "computed tomography" both count.
- **Errors are recorded, not fatal:** one provider failing doesn't lose the run.

---

## 54. Evaluation: LLM judge

**Method:** a separate, stronger model grades each answer on four 1–5 scales and returns JSON; agreement with the deterministic checks is reported.
**Where:** `medrag/eval/run.py` → `JUDGE_PROMPT`, `JUDGE_KEYS`, `_judge()`, the agreement block in `report()`; `medrag/providers/llms.py` → `get_judge()`.

```python
JUDGE_KEYS = ["correctness", "faithfulness", "answer_relevance", "context_relevance"]
prompt = JUDGE_PROMPT.format(q=it["question"], ref=it["answer"], ctx=ctx[:14000], ans=ans.text)
raw = judge.complete("You are a strict, fair evaluator. Output JSON only.", prompt, 600)
j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))            # tolerate text around the JSON

agree = ((both["judge_correctness"] >= 4) == (both["auto_correct"] == 1)).mean()   # 88% in the judged run
```

**Why:**
- **Catches what exact checks cannot:** a correct answer in different words, or a claim not supported by the sources (faithfulness).
- **The four scales cover the RAG triad:** context relevance (retrieval), faithfulness (grounding), answer relevance and correctness (generation).
- **A different model from the one answering:** a model grading itself is biased toward its own answers. Here qwen2.5:3b answered and Gemini 3 Flash judged.
- **Reference answer and sources included:** the judge grades against ground truth, not its own knowledge.
- **Agreement metric:** each grader checks the other. The 2 disagreements out of 16 were on the looser checks (trends, "never had"), where the judge was probably right.
- **`ctx[:14000]`:** bounds judge cost.
- **Robust:** the JSON is extracted with a regex, and a judge failure keeps the other metrics.

---

## 55. Evaluation: confidence intervals and report

**Method:** percentile bootstrap (1,000 resamples, seeded) for every mean; Markdown and CSV report per run; stratified sampling for answers.
**Where:** `medrag/eval/run.py` → `ci()`, `report()`; CLI `cmd_eval()` in `medrag/__main__.py`.

```python
def ci(values, n_boot=1000, seed=0):
    v = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(v) == 0:
        return "-"
    if len(v) == 1 or v.min() == v.max():
        return f"{v.mean():.3f}"
    means = np.random.default_rng(seed).choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return f"{v.mean():.3f} [{np.percentile(means, 2.5):.2f}-{np.percentile(means, 97.5):.2f}]"

# stratified answer sample: the same number of answers from every question type
per = max(1, a.max_answers // len(by_type))
gen_items = [it for group in by_type.values() for it in group[:per]]
```

**Why:**
- **Bootstrap intervals:** they show when a difference is noise. On 80 document questions, 0.808 vs 0.797 is inside the intervals, so the report says "equivalent" instead of declaring a winner.
- **Why bootstrap:** it works for any metric (binary hit, MRR, 1–5 scores) without assuming a normal distribution.
- **Vectorised:** one `(1000 × n)` resample matrix and one `.mean(axis=1)`.
- **Stratified answer sampling:** with an expensive LLM, the sample covers every question type rather than mostly the easy ones. (Rounding means 16 requested gives 18 answers with 18 types: audit finding 9.)

**Outputs:**
- **`retrieval.csv` and `generation.csv`:** keep every row, so any table can be recomputed or explored later.
- **`summary.md`:** the readable report.

---

## 56. User interface

**Method:** Streamlit with cached resources; streaming output; four tabs.
**Where:** `app.py`; `.streamlit/config.toml` (headless, no usage statistics).

```python
@st.cache_resource(show_spinner="Loading index...")
def load_retriever(path: str, built_at: str) -> Retriever:       # built_at in the key: a new build reloads
    return Retriever(Index(Path(path)))

rag = RAG(retriever, get_llm(llm_spec))
r, facts, stream = rag.stream(question, mode)
st.caption(f"Filters: {r.filters.describe()}" + (f" · {' '.join(r.notes)}" if r.notes else ""))
text = st.write_stream(stream)                                   # token-by-token output
```

**Why:**
- **Streamlit:** a full data app in one Python file, with no frontend build. It is right for an internal clinical tool and a demo.
- **`cache_resource` keyed on `built_at`:** the 442 MB index and models load once per process and reload automatically after a rebuild.
- **Transparency:** filters, expansion notes, sources and computed facts are visible, so a clinician can see *why* the answer says what it says.
- **Settings in the sidebar:** index, LLM, mode and k are switchable live, which makes manual comparison easy.

---

## 57. Tests

**Method:** pytest with a session fixture that builds a small dataset and index using the hash embedder.
**Where:** `tests/conftest.py`, `tests/test_pipeline.py` (15 tests).

```python
def test_count_is_zero_when_patient_lacks_the_modality(env, built):
    # a patient with no MRI: the relaxed search still shows their studies, but the count must be 0
    ...

class FakeClient:                                    # used by test_bedrock_streams_converse_text
    def converse_stream(self, **kw):
        return {"stream": [{"contentBlockDelta": {"delta": {"text": "Hel"}}},
                           {"contentBlockDelta": {"delta": {"text": "lo"}}}]}
```

**Why:**
- **Hash embedder:** no model download, so tests run offline in seconds and give the same result every time.
- **Real pipeline end to end:** generate, dirty, clean, chunk, index, search, answer scoring. This catches integration bugs that unit tests miss.
- **Regression tests for each audit fix:** they were confirmed to fail on the old code and pass on the new.
- **Fake Bedrock client:** tests the stream parsing with no AWS account.

---

## 58. One question, end to end

Question: **"Has Asha Rao ever had a slipped disc?"**

| Step | Method | What happens |
|---|---|---|
| 1. Parse | `QueryParser.parse` | `patient_ids=[<her id>]` (exact name match); no modality or date; not an aggregate |
| 2. Expand | `expand` | "slipped disc" → `disc herniation; disc extrusion nerve root compression`; note for the LLM |
| 3. Filter | `filter_rows` | Only her rows: studies, profile and documents (tens of rows instead of 121,335) |
| 4. Dense | `embed_query` + `dense_search` | NumPy dot product over her vectors, ranked |
| 5. BM25 | `sparse_search` | "disc", "extrusion", "nerve", "root" scored over the same rows |
| 6. Fuse | `rrf` | One ranking; chunks found by both rise |
| 7. Rerank | MedCPT + `rrf` (fuse) | A spine MRI that says "disc extrusion", if she has one, moves to the top |
| 8. Pin | Profile | Her profile chunk added as [S1] |
| 9. Facts | `computed_facts` | Study count by modality, abnormal count and her full timeline of impressions (up to 60) |
| 10. Prompt | `build_prompt` | Question + note + facts + [S1]–[S8] |
| 11. Answer | `OllamaLLM.stream` | "No: none of her N studies report a disc herniation [S1]." or "Yes: on <date> an MRI lumbar spine showed an L4-L5 disc extrusion [S2]." |
| 12. Evaluate | `score_answer` | `hard_negative` → `negative_correct`; `hard_paraphrase_patient` → date and modality facts → `auto_correct` |

---

## 59. Design decisions at a glance

| Decision | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Data | Synthetic + dirty injection + real public documents | Real patient data | Known right answers; no HIPAA risk; realistic mess |
| Cleaning | Explicit pandas rules with logging | Great Expectations, pandera, LLM cleaning | Small, auditable, deterministic, fast |
| Storage | Parquet or PostgreSQL behind one API | One backend only | Runs offline; same code in production |
| Chunking | Section → sentence → 220-word packing + overlap + contextual header | Fixed tokens, semantic, LLM chunking | Follows report structure; fits model limits; standalone chunks |
| Chunk types | Study, patient profile, document | One generic chunk | Different questions need different units |
| Embedding | bge-small-en-v1.5 (384-d), normalised | bge-large, e5, OpenAI, medical bi-encoders | Good quality per CPU second; small vectors |
| Vector store | FAISS flat (HNSW above 150k) + exact NumPy search inside filters | Chroma, Qdrant, Pinecone, pgvector | No server; offline; exact; filter-correct |
| Keyword search | bm25s + stemming + stop words | Elasticsearch, rank_bm25 | Fast, no server, exact tokens and IDs |
| Fusion | RRF, k = 60 | Weighted score sum | Scale-free, no tuning |
| Reranker | MedCPT cross-encoder, fused with the hybrid order | bge-reranker-base, MiniLM, none, replace | Best measured; medical training; fusion is safest |
| Filters | Rule-based parser + relaxation | LLM filter extraction | Instant, deterministic, testable |
| Lay terms | Dictionary expansion | HyDE, LLM query rewrite | Instant, cannot invent terms; +0.1 MRR |
| Counts | Computed facts from all metadata | Let the LLM count sources | Exact; the LLM sees only 8 sources |
| Index updates | Content-hash incremental + versioned atomic swap | Full rebuild, in-place overwrite | 50 min instead of about 3 h; safe while the app runs |
| LLM | Provider interface; local Ollama default; Bedrock for production | One vendor | Compare models; no lock-in; works offline |
| Evaluation | Generated test set, 4 retrieval metrics, deterministic checks + LLM judge, bootstrap CIs | Hand-labelled set, judge only | Scale, exactness, cost, honest uncertainty |
