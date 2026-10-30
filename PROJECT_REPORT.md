# MedRAG: full project report 
 
Everything done in this project, end to end: data collection, quality checks, cleaning, outside documents, chunking, embedding, indexing, retrieval, answering, evaluation (all question types and metrics), every experiment and result, problems hit along the way, fixes, deployment options and known limits. **Part I** (sections 1-28) describes what was done and the results; **Part II** (sections 29-59) is the technical deep dive: for every stage, the method, where it is in the code, how it works with sample code, and why it was chosen; **Part III** (sections 60-72) covers the practical and non-technical side. 
 
Repository: https://github.com/nikhilhv414/medrag (branch `main`). Report date: 2026-10-28, benchmarks and the larger answer test in sections 20.8-20.11 added 2026-10-30. 
 
## Executive summary 
 
**What it is.** MedRAG is an assistant that answers questions about patients' imaging records in plain English, for example "What did Asha Rao's last CT show?" or "How many urgent MRIs were done in 2023?". It finds the relevant records first, then has a language model write a short answer that cites them. Counts come straight from the database, so they are exact. If the records don't contain the answer, it says so instead of guessing. 
 
**What it was built and tested on.** 
- 20,000 synthetic patients with 89,868 imaging studies and reports, generated with realistic errors and then cleaned. 
- 6,211 real, public, de-identified documents: Open-i chest X-ray reports and MTSamples transcriptions. 
- In total, 121,335 searchable pieces of text. 
 
**How well it works.** 
- **Finding the right record:** it is among the top 8 sources for 100% of simple questions, 99% of hard ones (lay wording, misspelt names) and 88% of document questions. 
- **Design choices tested:** six embedding models, six chunking strategies and 16 retrieval settings were compared. None clearly beat the current setup, so it was kept (sections 20.8-20.10).
- **Larger answer test (54 questions per model):** Gemini 3.5 Flash-Lite passes 85%, qwen3:1.7b 80% and qwen2.5:3b 78%. On a laptop GPU the local models answer in about 4 s (median), close to the cloud (section 20.11).
- **Answers:** the best models, local (qwen3) and cloud (Gemini 3 Flash), pass 89% (16 of 18) of exact answer checks. Gemini 3 Flash answers in about 3 seconds, while local models on the laptop CPU take 45 seconds or more. The two remaining failures are pipeline issues, the same for every model (section 20.7). 
 
**Speed and cost.** 
- **Speed:** search takes 1-8 seconds on a laptop. Answers take about a minute with a local model; a hosted model (AWS Bedrock, already supported) brings that to a few seconds. 
- **Cost:** everything used is free and open source, and no paid service was needed. 
 
**Status.** 
- **Ready:** the pipeline works end to end, with 18 tests, a 313-question evaluation and a web app. 
- **Not ready:** it is not yet ready for real patient data. That needs compliance work, a login and validation on real reports (section 69). 
 
**How the report is organised.** 
- **Part I** (sections 1-28): what was done and the results. 
- **Part II** (sections 29-59): how each stage works in the code, and why. 
- **Part III** (sections 60-72): users, setup, hardware, performance, cost, risks, lessons learned, a glossary and an FAQ. 
 
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
 
**Part III: practical and non-technical details** (users, timeline, hardware, setup, models, performance, scaling, reproducibility, cost, risks and licences, lessons, glossary, FAQ): sections 60-72, [go to Part III](#part-iii-practical-and-non-technical-details) 
 
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
 DATA PREPARE SEARCH + ANSWER 
 ---- ------- --------------- 
 synthetic generator ─┐ 
 legacy oads_db ├─> raw stage ─> quality check ─> clean ─> clean stage ─┐ 
 ┘ (report) (log) │ 
 Open-i XML reports ──┐ ├─> chunks (+ context header) 
 MTSamples CSV ───────┴─> load ─> check ─> clean ─> link to patient ──────────┘ │ 
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
| `medrag/data/quality.py` | 28 data quality checks, Markdown + JSON report | 
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
| `tests/test_pipeline.py` | 16 end-to-end and unit tests | 
 
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
| `JUDGE_PROVIDER` / `JUDGE_MODEL` | nvidia / `nvidia/nemotron-3.5-super-vl-120b-a12b` (until 2026-10-30: gemini, set per run) | judge for evaluation | 
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
| reports | empty report text; HTML markup (`<p>`, `<br>`); status as `approved`, `draft`, `FINAL`, `signed`, `pending`; second duplicate report for the same study; hidden Unicode pasted into the text (zero-width spaces, a no-break space, a right-to-left override) | 
| ai_analysis | confidence as a percentage (88.0 instead of 0.88); confidence -1; missing confidence | 
 
Also, half of all clean modality values are written as the long name ("Ultrasound", "X-ray") instead of the code, as happens when systems are merged. 
 
Raw output: 20,316 patients, 35,417 conditions, 92,944 studies, 91,528 AI analyses, 92,935 reports. 
 
#### Generation and dirty injection, step by step 
 
`python -m medrag generate` (`cmd_generate` → `generate()` → `make_dirty()` → `store.write_all("raw", ...)`): 
 
| # | Step | Code | Records | 
|---|---|---|---:| 
| 1 | Build name pools (3 Faker locales, about 400 first and 400 last names each) | `_name_pool` | 20,000 patient names + 60 radiologists + 300 referring physicians | 
| 2 | Create patients with MRN, gender, date of birth, blood group, city | `generate` | 20,000 patients | 
| 3 | Add chronic conditions (count rises with age) | `generate` | 35,417 conditions | 
| 4 | Create each patient's study history (gamma-distributed count, follow-ups of earlier findings) | `generate` | 91,528 studies | 
| 5 | Write one five-section report per study | `_report` | 91,528 reports | 
| 6 | Add one AI prediction per study (86% correct) | `generate` | 91,528 AI analyses | 
| 7 | Write half the modality values as long names ("X-ray", "Ultrasound") | `generate` | about 45,800 studies | 
| 8 | Inject dirty records at a 3% rate | `make_dirty` | see below | 
| 9 | Write the raw stage | `store.write_all("raw", ...)` | 5 tables, 342,094 − 8,954 documents = 333,140 records | 
 
What `make_dirty` added or changed (measured by the raw quality check): 
 
| Problem | Records | 
|---|---:| 
| Duplicate patient registrations (same MRN, new id) | +311 patients | 
| Test / dummy patients | +5 patients | 
| Names with wrong case or extra spaces | 656 | 
| Non-standard gender values | 694 | 
| Missing date of birth | 196 | 
| Future date of birth | 66 | 
| Exact duplicate study rows | +1,416 studies | 
| Orphan studies (patient does not exist) | 567 | 
| Missing study date | 609 | 
| Future study date | 524 | 
| Misspelt modality (plus the long names from step 7) | 29,846 non-standard in total | 
| Non-standard priority | 2,826 | 
| Duplicate reports for the same study | +1,407 reports | 
| Empty report text | 858 | 
| Reports wrapped in HTML | 2,687 | 
| Reports with hidden Unicode characters | 2,756 | 
| Non-standard report status | 2,847 | 
| Confidence as a percentage or -1 | 3,468 outside 0-1 | 
| Missing confidence | 855 | 
 
### 4.3 Legacy database (`python -m medrag import-legacy`) 
 
The original `oads_db` PostgreSQL database (`oads.patients`, `oads.studies`, `oads.images`, `oads.analysis`, `oads.reports`; SQL dump in `data/source/oads_db.sql`) can be mapped onto the same raw layout: MRNs become `LEG0000001`, accessions `LEG-ACC-<id>`, modality is taken from the first image of the study. It then goes through the same check, clean and index steps. The main runs used the synthetic data because the legacy data is small. 
 
### 4.4 Outside documents (real text) 
 
- **Open-i** (Indiana University chest X-ray reports, US National Library of Medicine): `NLMCXR_reports.tgz` in `data/source/openi/`, 3,955 de-identified reports in XML with MeSH labels. Licence CC BY-NC-ND 4.0. 
- **MTSamples** medical transcriptions: `mtsamples.csv` (4,999 rows, about 40 specialties) downloaded with the user's approval from the socd06/medical-nlp GitHub copy of the Kaggle dataset, saved in `data/documents/`. 
 
Neither file is committed to git (the licence of MTSamples is unclear); `.gitignore` excludes `data/documents/*.csv`. 
 
### 4.5 Data inventory: every table, column and record count 
 
**Source files on disk** 
 
| File | Where | Size | Records | 
|---|---|---:|---:| 
| `NLMCXR_reports.tgz` (Open-i archive) | `data/source/openi/` | 1.1 MB | 3,955 XML reports | 
| Extracted Open-i XML | `data/source/openi/xml/ecgen-radiology/` | 32 MB | 3,955 files | 
| `mtsamples.csv` | `data/documents/` | 17 MB | 4,999 rows | 
| `oads_db.sql` (legacy database dump, optional) | `data/source/` | | not used in the main runs | 
| Raw stage (6 parquet tables) | `data/raw/` | 19 MB | see below | 
| Clean stage (6 parquet tables) | `data/clean/` | 14 MB | see below | 
| Index (bge-small) | `indexes/sentence-transformers__BAAI_bge-small-en-v1.5/` | 442 MB | 121,335 chunks | 
 
**Tables, columns and rows (raw → clean)** 
 
| Table | Columns | Raw rows | Clean rows | Removed / merged | Key | Links to | 
|---|---|---:|---:|---:|---|---| 
| `patients` | patient_id, mrn, full_name, gender, date_of_birth, blood_group, city (7) | 20,316 | 20,000 | 316 (5 test records, 311 duplicate registrations merged) | patient_id | | 
| `conditions` | patient_id, condition, diagnosed_on (3) | 35,417 | 35,417 | 0 (re-pointed to merged patients) | (patient_id, condition) | patients | 
| `studies` | study_id, patient_id, accession, study_date, modality, body_part, indication, priority, referring_physician (9) | 92,944 | 89,868 | 3,076 (1,416 duplicates, 558 orphans, 1,102 missing or future dates) | study_id | patients | 
| `reports` | report_id, study_id, radiologist, report_text, status, signed_at (6) | 92,935 | 89,014 | 3,921 (884 empty, 1,369 duplicate reports, the rest belonged to removed studies) | report_id | studies | 
| `ai_analysis` | analysis_id, study_id, model_name, predicted_label, confidence (5) | 91,528 | 89,868 | 1,660 (analyses of removed studies) | analysis_id | studies | 
| `documents` | document_id, patient_id, source, source_id, title, specialty, doc_date, text, labels, description (10) | 8,954 | 6,211 | 2,743 (158 empty or very short, 2,585 duplicate transcriptions merged) | document_id | patients | 
| **Total** | **40 columns** | **342,094** | **330,378** | | | | 
 
How the tables relate: 
 
``` 
patients (20,000) ──< conditions (35,417) 
 │ 
 ├──< studies (89,868) ──< reports (89,014) one report per study after cleaning 
 │ └──< ai_analysis (89,868) one AI prediction per study 
 │ 
 └──< documents (6,211) Open-i 3,878 + MTSamples 2,333, linked to patients 
``` 
 
**What is in the clean data** 
 
| Measure | Value | 
|---|---| 
| Patients | 20,000 (female 10,134, male 9,587, unknown 279) | 
| Patients with at least one study | 19,897 (103 have none left after cleaning) | 
| Studies per patient | mean 4.5, median 3, max 42 | 
| Chronic conditions | 35,417, mean 2.2 per patient who has any | 
| Studies by modality | X-ray 33,375 · CT 22,438 · Ultrasound 16,314 · MRI 13,328 · PET-CT 2,675 · Mammography 1,738 | 
| Studies by body part | chest 27,518 · abdomen 19,196 · knee 9,285 · spine 9,204 · head 5,554 · brain 5,416 · thyroid 4,176 · hand 3,272 · whole body 2,675 · pelvis 1,834 · breast 1,738 | 
| Studies by priority | routine 61,956 · urgent 17,821 · stat 10,091 | 
| Study dates | 2020-10-02 to 2026-09-29 | 
| Reports by status | final 77,601 · preliminary 7,223 · amended 4,190 | 
| Report length | mean 37 words (27 to 55) | 
| Reports with an abnormal finding | 63.5% | 
| AI predictions without a confidence (cleared as invalid or missing) | 1,640 | 
| Documents | 6,211, attached to 5,377 patients, dated 2020-10-01 to 2026-09-29 | 
| Open-i report length | mean 49 words (15 to 238) | 
| MTSamples transcription length | mean 466 words, median 396 (15 to 3,029) | 
| MTSamples specialties | 39 main specialties; largest: Surgery 969, Radiology 246, General Medicine 156, Urology 154, SOAP / Chart / Progress Notes 143, Neurology 67 | 
 
**From records to chunks** 
 
| Records | Count | Chunks | Chunks per record | 
|---|---:|---:|---:| 
| Reports (study chunks) | 89,014 | 89,014 | 1.0 | 
| Patients with studies (profile chunks) | 19,897 | 19,923 | 1.0 | 
| Open-i reports | 3,878 | 3,880 | 1.0 | 
| MTSamples transcriptions | 2,333 | 8,518 | 3.7 | 
| **Total** | **115,122** | **121,335** | | 
 
--- 
 
## 5. Data quality checks 
 
`python -m medrag check --stage raw|clean` runs 28 checks and writes `reports/quality_<stage>.md` and `.json`. `--strict` exits with code 1 if any error-level check fails, so it can gate a pipeline. 
 
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
| reports | hidden Unicode characters (zero-width, bidi, tag) | 2,756 | 2.97 | warn | 
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
2. **Remove hidden Unicode**: zero-width characters, bidi overrides and tag characters are deleted and exotic spaces (no-break, thin, ideographic) become normal spaces. They are invisible to a reader but reach BM25, the embedder and the LLM, where they can split words or hide instructions. Load-bearing invisibles (emoji joiners, Arabic/Indic joiners) are kept. 
3. Remove empty reports. 
4. Keep only reports of studies that survived cleaning. 
5. Normalise status (`approved`, `signed`, `finalized` → final; `draft`, `prelim`, `pending` → preliminary; `addendum`, `corrected` → amended). 
6. **One report per study**: prefer final, then amended, then preliminary; within that the latest signed. 
 
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
| reports: removed hidden Unicode characters | 2,756 | 
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
 
### 6.4 Cleaning step by step, rows in and out 
 
`python -m medrag clean` (`cmd_clean` → `store.read_all("raw")` → `clean()` → `store.write_all("clean", ...)` → `quality.run("clean", ...)`). The rules run in this order; the right-hand column is the row count after each step. 
 
**patients** 
 
| # | Step | Changed | Rows after | 
|---|---|---:|---:| 
| | Read raw | | 20,316 | 
| 1 | Remove test / dummy records | −5 | 20,311 | 
| 2 | Remove records with no name | 0 | 20,311 | 
| 3 | Normalise name case and spaces | 654 changed | 20,311 | 
| 4 | Normalise gender | 694 changed | 20,311 | 
| 5 | Clear future date of birth | 66 cleared | 20,311 | 
| 6 | Merge duplicate registrations by MRN (remap kept) | −311 | **20,000** | 
 
**conditions** 
 
| # | Step | Changed | Rows after | 
|---|---|---:|---:| 
| | Read raw | | 35,417 | 
| 1 | Re-point to merged patients, drop duplicate (patient, condition) pairs | 0 removed | **35,417** | 
 
**studies** 
 
| # | Step | Changed | Rows after | 
|---|---|---:|---:| 
| | Read raw | | 92,944 | 
| 1 | Drop exact duplicate rows and duplicate ids | −1,416 | 91,528 | 
| 2 | Re-point to merged patients | 632 changed | 91,528 | 
| 3 | Remove orphans | −558 | 90,970 | 
| 4 | Normalise modality; remove unknown | 29,221 changed, 0 removed | 90,970 | 
| 5 | Normalise priority | 2,761 changed | 90,970 | 
| 6 | Remove missing or future dates | −1,102 | **89,868** | 
 
**reports** 
 
| # | Step | Changed | Rows after | 
|---|---|---:|---:| 
| | Read raw | | 92,935 | 
| 1 | Strip HTML | 2,687 changed | 92,935 | 
| 2 | Remove hidden Unicode | 2,756 changed | 92,935 | 
| 3 | Remove empty reports | −884 | 92,051 | 
| 4 | Keep only reports of surviving studies | −1,668 | 90,383 | 
| 5 | Normalise status | 2,768 changed | 90,383 | 
| 6 | One report per study (final > amended > preliminary, then latest) | −1,369 | **89,014** | 
 
**ai_analysis** 
 
| # | Step | Changed | Rows after | 
|---|---|---:|---:| 
| | Read raw | | 91,528 | 
| 1 | Keep only analyses of surviving studies | −1,660 | 89,868 | 
| 2 | Convert percentage confidence to 0-1 | 2,602 changed | 89,868 | 
| 3 | Clear confidence still outside 0-1 | 800 cleared | **89,868** | 
 
**After cleaning** 
 
| Check | Result | 
|---|---| 
| Studies with no report (kept for counts) | 854 (89,868 studies − 89,014 reports) | 
| Error-level quality checks failing | 9 before → **0** after | 
| Total records | 333,140 raw → 324,167 clean (without documents) | 
| Files written | `data/clean/*.parquet`, `reports/cleaning_log.md`, `reports/quality_clean.md`, `reports/quality_clean.json` | 
 
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
| hidden Unicode characters (zero-width, bidi, tag) | 0 | 0 | 
| de-identification tokens (XXXX) | 3,423 | 0 | 
| Open-i report without IMPRESSION | 34 | 6 | 
| missing patient link | 8,954 | 0 | 
 
### 7.3 Cleaning 
 
1. **Remove hidden Unicode** (same rule as for reports). The current Open-i and MTSamples files have none, but any new source goes through it. 
2. **Rebuild section headings** in transcriptions. MTSamples text has CSV artefacts like `HISTORY OF PRESENT ILLNESS:, The patient...`; these are turned back into `\nHISTORY OF PRESENT ILLNESS: The patient...` so the chunker can split on sections (4,118 documents). 
3. **Mark de-identified words**: Open-i replaces names and dates with `XXXX`; these become `[redacted]` so the LLM does not treat them as a word (3,423 documents). 
4. Tidy whitespace per line. 
5. **Remove empty and very short** documents, under 15 words (158). 
6. **Merge duplicate transcriptions**: MTSamples files the same transcription under several specialties (for example both "Surgery" and "Orthopedic"). One copy is kept and the other specialties are remembered in the specialty field (2,585 merged). Only MTSamples: identical Open-i reports ("No acute cardiopulmonary disease.") belong to different patients and are kept. 
 
Result: 3,878 Open-i reports and 2,333 unique MTSamples transcriptions. 
 
### 7.4 Linking to patients 
 
The document text is real, but it has no link to the synthetic patients. To make per-patient questions possible, each document is attached to a plausible synthetic patient: 
 
- **Gender** from the text: counts of male words (he, his, him, Mr, male, man, gentleman, boy) vs female words (she, her, Mrs, Ms, female, woman, lady, girl). 
- **Age** from phrases like "45-year-old". 
- A random date in the data window (2020-10-01 to 2026-09-30). 
- The patient is picked at random among those of the same gender and within 2 years of the stated age at that date. If none match, gender only; if no gender either, any patient. 
 
Linked by age + gender 1,058, by gender only 1,879, at random 3,274 (mostly Open-i reports, which are de-identified). Seeded (13), so it is reproducible. This link is synthetic and is documented as such. 
 
### 7.5 Ingestion step by step, with counts 
 
`python -m medrag documents` (`cmd_documents` in `medrag/__main__.py`) runs these steps in order: 
 
| # | Step | Code | Open-i | MTSamples | Total | 
|---|---|---|---:|---:|---:| 
| 1 | Download / place the files | Open-i archive in `data/source/openi/`, CSV in `data/documents/` | 3,955 reports | 4,999 rows | 8,954 | 
| 2 | Extract the archive once (safe `data` filter) | `loaders.load_openi` | 3,955 XML files | | | 
| 3 | Parse into one layout: source, source_id, title, specialty, text, labels, description | `loaders.load_openi`, `loaders.load_mtsamples`, `loaders.load_all` | 3,955 | 4,999 | 8,954 | 
| 4 | Write as-is to the **raw** stage (`documents` table, no patient yet) | `store.write_table("raw", "documents", ...)` | | | 8,954 | 
| 5 | Quality check "before" | `prepare.check` | | | 7 checks | 
| 6 | Rebuild section headings (`HEADING:,` → new line) | `prepare.clean` | | 4,118 fixed | 4,118 | 
| 7 | Mark de-identified words `XXXX` → `[redacted]` | `prepare.clean` | 3,411 fixed | 12 fixed | 3,423 | 
| 8 | Tidy whitespace per line | `prepare.clean` | all | all | | 
| 9 | Remove empty or very short (< 15 words) | `prepare.clean` | −77 | −81 | −158 | 
| 10 | Merge duplicate transcriptions (keep one, remember every specialty) | `prepare.clean` | (not applied) | −2,585 | −2,585 | 
| 11 | Link each document to a patient (gender, age, date) | `prepare.link_to_patients` | 3,878 | 2,333 | 6,211 | 
| 12 | Write to the **clean** stage (`documents` table with patient_id and doc_date) | `store.write_table("clean", "documents", ...)` | 3,878 | 2,333 | 6,211 | 
| 13 | Quality check "after", report `reports/quality_documents.md` | `prepare.check` | | | 7 checks | 
 
Then `python -m medrag index` adds them to the index (it reads every clean table, including `documents`): 
 
| # | Step | Code | Result | 
|---|---|---|---| 
| 14 | Chunk each document with a header naming the patient, title, specialty, date and source (MTSamples also gets its one-line summary) | `ingest.documents.document_chunks` | 3,880 Open-i + 8,518 MTSamples chunks | 
| 15 | Tag Open-i chunks as X-ray / chest, and store each document's conclusion line as its impression | `document_chunks`, `_summary_line` | "chest X-ray" filters include them | 
| 16 | Add the patient's documents to their **profile chunk** ("HISTORY: 2 uploaded documents: ...") | `build_chunks` | patient summaries mention them | 
| 17 | Embed only the new chunks; reuse every existing vector | `index.store.build` | 10,749 embedded when MTSamples was added (50 min); 7,428 when Open-i was added (12 min) | 
| 18 | Rebuild BM25 and FAISS over all 121,335 chunks; switch the manifest atomically | `index.store.build` | the app picks up the new build | 
 
### 7.6 How documents join the rest of the data 
 
After ingestion, documents are used everywhere the radiology records are used: 
 
| Where | How documents are used | 
|---|---| 
| Search | Document chunks sit in the same FAISS and BM25 index as the studies, with `doc_type = "document"` | 
| Filters | Words like "note", "transcription", "discharge summary" or a specialty name ("cardiology") set `documents=True`, which limits search to documents and profiles | 
| Patient questions | The patient filter includes their documents, because every document has a `patient_id` | 
| Patient profile | Lists the patient's 15 most recent documents with date, title and specialty | 
| Computed facts | For a patient question, lists up to 20 of their documents with date, title, specialty, id and conclusion | 
| User interface | The Patients tab shows "Uploaded documents" for each patient, with source and specialty | 
| Evaluation | Four document question types, 80 questions (section 18) | 
 
To add more documents later: put the new files in the same folders and run `python -m medrag documents` and then `python -m medrag index`. Only the new chunks are embedded. 
 
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
 
### 9.5 Chunking step by step 
 
`python -m medrag index` → `cmd_index` → `build_chunks(store.read_all("clean"), max_words=220)`: 
 
| # | Step | Code | In | Out | 
|---|---|---|---:|---:| 
| 1 | Read every clean table | `store.read_all("clean")` | 6 tables | 330,378 records | 
| 2 | Join studies + reports + latest AI prediction + patient | `build_chunks` (merges) | 89,868 studies | 89,868 rows | 
| 3 | Skip studies with no report | `build_chunks` | 89,868 | 89,014 | 
| 4 | Study chunks: header + report + AI line, packed to 220 words | `chunk_document` | 89,014 reports | 89,014 chunks | 
| 5 | Group studies by patient, newest first; list conditions and documents | `build_chunks` | 19,897 patients | 19,897 profiles | 
| 6 | Profile chunks: header + history + up to 40 impressions, 300 words | `chunk_document` | 19,897 profiles | 19,923 chunks (19,871 profiles fit in 1 chunk, 26 needed 2) | 
| 7 | Document chunks: header + cleaned text, 220 words | `document_chunks` | 6,211 documents | 12,398 chunks (Open-i 3,880, MTSamples 8,518) | 
| 8 | Hash every chunk's text (id), drop exact duplicates | `_hash`, `drop_duplicates("chunk_id")` | 121,335 | 121,335 (none were duplicates) | 
 
Chunk size in the result: mean 88 words (study 79, profile 86, document 154), smallest 42, largest 313 (an over-long single sentence). Chunk metadata file: 24 MB. 
 
Example, one study chunk exactly as indexed: 
 
``` 
Patient: Aarini Hayre | MRN: MRN0000011 | Patient ID: 11 | male, age 51 
Study: CT chest | Date: 2020-11-15 | Accession: ACC000000040 | Priority: urgent | Report status: preliminary 
Indication: Follow-up of lung nodule 
TECHNIQUE: Contrast-enhanced CT of the chest with axial 1.25 mm reconstructions. 
COMPARISON: None available. 
FINDINGS: Moderate left pleural effusion with adjacent compressive atelectasis. 
IMPRESSION: Moderate left pleural effusion. 
RECOMMENDATION: Consider diagnostic thoracentesis if clinically indicated. 
AI MODEL (DeepCT-Seg v4.2): predicted 'pleural effusion' (confidence 0.86). 
``` 
 
(Copied from the index. The patient is synthetic.) 
 
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
 
### 11.1 Embedding and indexing step by step (latest build) 
 
`cmd_index` → `index.store.build(chunks, embedder, path)`: 
 
| # | Step | Code | Result | 
|---|---|---|---| 
| 1 | Read the previous build's chunk ids and vectors | `build` (via `_files` + manifest version) | previous build (before MTSamples) | 
| 2 | Compare ids: which chunks are new or changed | `todo = [... if cid not in old]` | 10,749 to embed (new MTSamples chunks, plus profiles whose document list changed), 110,586 reused | 
| 3 | Embed the new chunks with bge-small in batches of 256 | `embedder.embed_documents` | 42 batches, 10,749 vectors, about 50 min on CPU (3.6 chunks/s) | 
| 4 | Save all vectors | `np.save` | `vectors-<version>.npy`: 121,335 × 384 float32, 186 MB | 
| 5 | Save chunk text and metadata | `to_parquet` | `chunks-<version>.parquet`: 24 MB | 
| 6 | Build the FAISS index (121,335 < 150,000, so exact) | `faiss.IndexFlatIP` | `dense-<version>.faiss`: 186 MB, 121,335 vectors | 
| 7 | Tokenise (stop words, stemming) and build BM25 | `tokenize`, `bm25s.BM25.index` | `bm25-<version>/`: vocabulary of 145,557 terms, 64 MB | 
| 8 | Write the manifest to a temp file and swap it in atomically | `os.replace` | `manifest.json` points at the new version | 
| 9 | Delete the previous build's files | `_remove_old_builds` | one build left on disk, 442 MB in total | 
 
Earlier builds: first full build about 107k chunks (2 h 50 min, everything embedded); adding Open-i 7,428 new chunks (12 min). 
 
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
2. **Filter**: restrict the search to rows matching all filters. If nothing matches, drop the least important filter and try again, in this order: abnormal, body part, documents, priority, dates, modality, patient. Only filters the question actually set are reported. Each relaxation is noted ("No records match every filter; relaxed the 'modality' filter."). 
3. **Expand** the query (section 13). 
4. **Dense search**: top 60 by cosine similarity. Inside a filter this is an exact search over the subset's vectors. 
5. **BM25 search**: top 60 over the same subset. 
6. **Reciprocal Rank Fusion** (RRF, k = 60): score = Σ 1 / (60 + rank) over both lists. 
7. **Rerank**: the top 30 fused candidates are re-scored by the MedCPT cross-encoder (max length 512). In `fuse` mode (default) the reranker's order is fused again with the hybrid order by RRF, so a reranker cannot throw away what the first stage got right. In `replace` mode the reranker's order is used alone. 
8. **Keep top k** (8). 
9. **Pin**: for a patient question, that patient's profile chunk is always included; for a "latest" question, their 3 most recent studies too. At least half of the k slots stay for search results. 
 
Each hit records how it was found (`dense+bm25`, `profile`, `latest`), shown in the UI. 
 
### 14.1 Retrieval step by step: two real questions 
 
Measured on the current index (121,335 chunks). 
 
**Question A: "Has Elaine Chanda ever had a slipped disc?"** (a patient question with a lay term) 
 
| # | Step | Code | Result | 
|---|---|---|---| 
| 1 | Parse | `QueryParser.parse` | patient Elaine Chanda (id 14277), finding = disc herniation | 
| 2 | Filter | `filter_rows` | 121,335 → **15** rows (her studies, profile and documents) | 
| 3 | Expand | `expand` | "... slipped disc? (disc herniation; disc extrusion nerve root compression)" | 
| 4 | Dense search | `dense_search` | 15 candidates (all of her rows; fewer than 60) | 
| 5 | BM25 search | `sparse_search` | 15 candidates | 
| 6 | Fuse | `rrf` | 15 | 
| 7 | Rerank top 30 with MedCPT, fuse again | `Reranker.score`, `rrf` | 15 re-ordered | 
| 8 | Pin profile, keep top 8 | `search` | 8 sources: 1 profile + 7 found by both dense and BM25 | 
| 9 | Computed facts | `computed_facts` | "Matching studies: 0 across 0 patients" (she never had one) | 
| 10 | Prompt | `build_prompt` | about 770 words to the LLM | 
 
**Question B: "How many stat priority MRI studies were performed in 2023?"** (a cohort count) 
 
| # | Step | Code | Result | 
|---|---|---|---| 
| 1 | Parse | `QueryParser.parse` | modality = MRI, priority = stat, dates 2023-01-01 to 2023-12-31, aggregate | 
| 2 | Filter | `filter_rows` | 121,335 → **281** rows | 
| 3 | Expand | `expand` | no medical terms, query unchanged | 
| 4 | Dense search | `dense_search` | top 60 of 281 | 
| 5 | BM25 search | `sparse_search` | top 60 of 281 | 
| 6 | Fuse | `rrf` | **109** unique candidates (11 found by both) | 
| 7 | Rerank top 30, fuse again | `Reranker.score`, `rrf` | 30 re-ordered | 
| 8 | Keep top 8 | `search` | 8 sources: 5 found by both, 2 dense only, 1 BM25 only | 
| 9 | Computed facts | `computed_facts` | "Matching studies: 281 across 273 patients", date range, by modality, priority, abnormal count, top findings | 
| 10 | Prompt | `build_prompt` | about 850 words to the LLM | 
 
Search time per question on CPU: 1.7 s (easy), 2.6 s (hard) and 8 s (documents) on average in the evaluation. The MedCPT reranker is most of it. 
 
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
 
### 15.4 Answering step by step 
 
`RAG.ask(question)` (or `RAG.stream` in the app): 
 
| # | Step | Code | Size / time | 
|---|---|---|---| 
| 1 | Retrieve | `Retriever.search` (section 14.1) | 8 sources, 1.7-8 s | 
| 2 | Unknown patient? Answer "not found" without the LLM | `RAG.ask` | instant | 
| 3 | Compute exact facts over all matching rows | `computed_facts` | 2-7 lines for cohort questions; up to about 90 lines for one patient (60-row timeline + 20 documents) | 
| 4 | Build the prompt: question, expansion notes, facts, [S1]-[S8] | `build_prompt` | about 770-850 words (about 1,100 tokens), within the 8,192-token context | 
| 5 | Generate with the system rules, temperature 0.1, up to 1,500 tokens | `LLM.stream` / `complete` | 44-450 s on CPU depending on the model (section 20.6) | 
| 6 | Return answer + sources + facts + timings | `Answer` | shown in the UI, or scored in the evaluation | 
 
--- 
 
## 16. LLM providers 
 
`medrag/providers/llms.py`. One interface: `stream(system, user)` yields text; `complete()` joins it and retries on rate limits and temporary overload (HTTP 429, RESOURCE_EXHAUSTED, "rate limit", ThrottlingException, 503 UNAVAILABLE) with waits of 15, 30, 45, 60 seconds. 
 
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
 
### 18.1a Question counts in the current test set 
 
Built with `--per-type 20`, seed 11 (`reports/eval_20261028_1350/testset.json`): 
 
| Difficulty | Type | Questions | Has a gold document (used for retrieval scoring) | 
|---|---|---:|:-:| 
| easy | lookup_by_name | 20 | yes | 
| easy | lookup_by_mrn | 20 | yes | 
| easy | latest | 20 | yes | 
| easy | count_patient | 20 | yes (the profile) | 
| easy | semantic | 20 | yes | 
| easy | ambiguous_name | 10 | no | 
| easy | count_cohort | 10 | no | 
| easy | not_found | 5 | no | 
| hard | hard_paraphrase_patient | 20 | yes | 
| hard | hard_semantic_cohort | 20 | yes | 
| hard | hard_trend | 20 | yes | 
| hard | hard_typo_name | 18 | yes (2 surnames were too short to misspell) | 
| hard | hard_negative | 20 | no | 
| hard | hard_count_finding | 10 | no | 
| documents | doc_patient_lookup | 20 | yes | 
| documents | doc_semantic | 20 | yes | 
| documents | doc_transcription_lookup | 20 | yes | 
| documents | doc_transcription_semantic | 20 | yes | 
| | **Total** | **313** (125 easy, 108 hard, 80 documents) | **258** | 
 
### 18.1b Evaluation run step by step 
 
`python -m medrag eval --llms "ollama:phi4-mini,ollama:llama3.2:3b" --max-answers 16 --per-type 20` (`cmd_eval`): 
 
| # | Step | Code | Result | 
|---|---|---|---| 
| 1 | Load the index and build the retriever configurations | `cmd_eval`, `run.Config` | 1 configuration (more with `--baselines`, `--rerankers`, `--blends`) | 
| 2 | Build the test set from the index | `testset.build` | 313 questions | 
| 3 | Save the questions and a human review sheet | `testset.save`, `testset.review_sheet` | `testset.json`, `golden_review.csv` (30 rows) | 
| 4 | Run retrieval for every question with a gold document | `eval_retrieval` | 258 searches per configuration | 
| 5 | Score each search | `retrieval_metrics` | Hit@8, MRR, Recall@8, nDCG@8, latency | 
| 6 | Pick the answer sample, the same number from every type | `cmd_eval` | 18 questions (one per type) | 
| 7 | Answer them with each LLM | `eval_generation` → `RAG.ask` | 18 answers per model | 
| 8 | Score each answer with the exact checks | `score_answer` | fact_recall, number, negative, abstain, citations, auto_correct | 
| 9 | Optional: judge each answer | `_judge` (with `--judge`) | 4 scores of 1-5 + reason | 
| 10 | Means with bootstrap 95% intervals, tables by difficulty and type | `ci`, `report` | `summary.md` | 
| 11 | Save every row | `report` | `retrieval.csv` (258 rows per configuration), `generation.csv` (18 rows per model) | 
 
Run times on CPU: retrieval for 258 questions about 17 minutes (sum of the measured search times); answers about 1 minute each for 3-4B models, 7-9 minutes for qwen3:4b. The `--baselines` run adds 5 more configurations, so retrieval takes about 6 times as long. 
 
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
| `fact_recall` | share of expected key facts present in the answer (a fact may list alternatives, `ct\|computed tomography`) | 
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
| `eval_20261028_1847` | Gemini 3 Flash and Gemini 3.1 Flash-Lite on the same 18 questions | local vs cloud comparison; found the cohort-count test bug | 
| `eval_20261028_2227` | retrieval only, first filter-order fix (all study filters before documents) | documents up, but `lookup_by_mrn` Hit@8 fell to 0.55; rejected | 
| `eval_20261028_2246` | retrieval only, final filter-order fix (body part before documents) | current retrieval table (20.4) |
| `bench_embed_20261029_0050`, `_2258` | 6 embedding models on a 15,749-chunk subset | keep bge-small (20.8) |
| `bench_chunk_20261029_2325` | 6 chunking strategies on the same subset | keep section-220 (20.9) |
| `bench_tune_20261029_2344` | 16 retrieval variants on the full index | `dense_weight=0.5` the best variant (20.10) |
| `dense_weight_check.csv` (2026-10-30) | `dense_weight` 1.0 vs 0.5 on 240 fresh document questions | 0.5 confirmed (+0.016 [0.01-0.03]) and made the default |
| `ctx_chunk.jsonl`, `ctx_parent.jsonl` (2026-10-30) | chunk vs parent context, 60 document answers, Gemini 3.5 Flash-Lite | parent 93% vs chunk 83% correct (20.9) |
| `reports/gen_large*.jsonl` (2026-10-29/30) | 54 questions per model, local models on GPU, Gemini and NVIDIA | deterministic checks and LLM judge (NVIDIA Nemotron 3.5 Super): all 432 answers of 8 models judged (20.11) | 
 
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
| hybrid + MiniLM reranker (replace) | 1.000 | 0.872 | 0.851 | 0.6 s | 
| hybrid, no reranker | 1.000 | 0.875 | 0.762 | 0.1 s | 
| dense only | 0.995 | 0.913 | 0.621 | 0.1 s | 
| BM25 only | 1.000 | 0.865 | 0.837 | 0.05 s | 
| hybrid + rerank, no query expansion | 0.995 | 0.793 | 0.797 | 0.6 s | 
| hybrid + rerank, no filters | 0.495 | 0.509 | 0.507 | 4.1 s | 
 
Questions: 100 easy, 78 hard, 40 documents (Open-i only; MTSamples was added later). Latency is the mean over all 218. 
 
Earlier (`eval_20261027_1958`, hard questions): general reranker `bge-reranker-base` 0.845, hybrid with no reranker 0.876, MiniLM 0.874, no expansion 0.817. 
 
Findings: 
- **Metadata filtering is the biggest win**: it roughly doubles MRR (0.5 → 0.9-1.0). 
- **Query expansion** adds about 0.11 MRR on hard questions (0.905 with it, 0.793 without). On paraphrases whose wording is not in the dictionary, the default still finds 92.6% of gold documents (Recall@8), and plain hybrid without the reranker 98.1%, vs 100% for wording in the dictionary. 
- **Reranker choice matters**: a general web-trained reranker made results worse; the medical MedCPT helps; fusing its order with the hybrid order is the safest. MiniLM in replace mode is best on documents (0.851) but weakest of the rerankers on hard questions (0.872). 
- **Dense alone** is weak on keyword-style document questions (`doc_semantic` 0.27); **BM25 alone** is weak on paraphrases. Hybrid covers both. 
- Differences of a few hundredths are within the confidence intervals; the top three rerank configurations are effectively equivalent. 
 
nDCG@8 for the default: easy 1.000, hard 0.920, documents 0.831. 
 
### 20.4 Retrieval: default configuration on the full current test set (`eval_20261028_2246`) 
 
After the filter-order fix (section 44). Values before the fix (`eval_20261028_1044`) in brackets where they differ. 
 
| Difficulty | Questions | Hit@8 | MRR | Recall@8 | nDCG@8 | Latency | 
|---|---:|---|---|---|---|---:| 
| Easy | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.5 s | 
| Hard | 78 | 0.987 [0.96-1.00] | 0.905 [0.85-0.95] | 0.983 [0.95-1.00] | 0.920 [0.88-0.96] | 2.4 s | 
| Documents | 80 | **0.925** [0.86-0.97] (0.875) | **0.859** [0.79-0.93] (0.809) | **0.925** [0.86-0.97] (0.875) | **0.875** [0.81-0.94] (0.825) | 8.3 s | 
 
MRR by type: count_patient 1.00, lookup by name/MRN 1.00, latest 1.00, semantic 1.00, typo name 1.00, hard paraphrase 0.938, hard trend 0.875, hard semantic cohort 0.815, doc patient lookup 0.975, doc transcription lookup 1.00, doc transcription semantic **0.90** (0.70 before the fix), doc semantic (Open-i by content) 0.562. 
 
The 258 questions here are those with a gold document; the other 55 (counts, "not found", negatives) are checked only in the answer evaluation. 
 
Finding a document by content alone is the hardest case: many Open-i reports use near-identical wording ("no acute cardiopulmonary disease"). 
 
### 20.5 Answers judged by an LLM (`eval_20261028_0006`) 
 
qwen2.5:3b answering, gemini-3-flash-preview judging, 16 questions (8 easy, 6 hard, 2 documents; one per type, before the two MTSamples types existed): 
 
| | Easy | Hard | Documents | 
|---|---:|---:|---:| 
| Pass the exact check (auto_correct) | 63% (75% after the test fix in 20.7) | 83% | 50% | 
| Judge correctness (1-5) | 3.6 | 3.2 | 3.0 | 
| Judge faithfulness (1-5) | 4.1 | 3.7 | 3.0 | 
| Judge answer relevance (1-5) | 4.9 | 5.0 | 5.0 | 
| Judge context relevance (1-5) | 4.5 | 4.5 | 5.0 | 
| Says "not found" correctly | 100% | 83% | 100% | 
| Cites sources | 13% | 17% | 100% | 
| Seconds per answer (CPU) | 76 | 97 | 82 | 
 
- Scored correctly: lookups by name and MRN, latest study, per-patient counts, not found, misspelt names, paraphrases, document lookups. 
- Failed: ambiguous names (shared by two patients), semantic search across many records, finding counts, document content search. The cohort count was also marked as failed, but the model's answer (101) was right; the test expected 99 because of a test-set bug fixed later (section 20.7). 
- Judge vs exact checks agree on **88%** of answers; both disagreements are on the looser checks ("never had", trends), where the judge is probably right. 
 
### 20.6 Five local models on the same 18 questions (`eval_20261028_1044`, `eval_20261028_1350`) 
 
18 questions, one per type (8 easy, 6 hard, 4 documents); exact checks only (no judge), CPU; seconds are mean generation time (search adds about 5 s): 
 
| Model | Easy | Hard | Documents | Correct "not found" | Cites sources | Seconds per answer | 
|---|---:|---:|---:|---:|---:|---:| 
| **qwen3:1.7b** | 88% | 83% | 75% | 25-63% | 50-63% | 43 | 
| qwen3:4b | 88% | 83% | 75% | 75-88% | 75-100% | 446 | 
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
- **Every model failed** the cohort count, the finding count and transcription content search. Section 20.7 shows that the cohort-count failure was a test-set bug (every model was right), and explains the other two. 
- The default was kept at qwen2.5:3b by the user's choice; `LLM_MODEL=qwen3:1.7b` in `.env` switches. 
- With one question per type, one answer moves a row by 12 points (easy), 17 (hard) or 25 (documents): treat differences as a direction, not proof. (qwen2.5:3b scored 83% on hard in the judged run and 67% here because of one question.) 
 
### 20.7 Local models vs Gemini (7 models, corrected scoring) 
 
Gemini 3 Flash and Gemini 3.1 Flash-Lite (Google AI Studio free tier) answered the same 18 questions as the five local models, with the same retrieval, prompt and checks (`reports/eval_20261028_1847`). 
 
**A test-set bug found by this comparison.** 
- **What was wrong:** the cohort-count question ("How many urgent priority PET-CT studies were performed in 2023?") expected **99**, but the true number in the data is **101**. The question builder counted only patients with a unique name. 
- **Effect:** all 7 models answered 101, so all 7 were marked wrong on a question they got right. 
- **Fix:** `testset.build` now counts every patient, and a regression test was added. 
- **Re-scoring:** all 126 saved answers were re-scored against the fixed test set, without calling any model again (`reports/model_comparison_7.csv`). The table in 20.6 shows the scores from before this fix. 
 
| Model | Runs on | Easy (8) | Hard (6) | Documents (4) | **Total (18)** | Says "not found" only when right | Cites sources | Generation seconds per answer (mean / median) | 
|---|---|---:|---:|---:|---:|---:|---:|---:| 
| **Gemini 3 Flash** | Google cloud | 100% | 83% | 75% | **89% (16)** | 89% | 89% | **6.8 / 3.1** | 
| Gemini 3.1 Flash-Lite | Google cloud | 100% | 83% | 75% | **89% (16)** | 78% | 67% | 15.3 / 11.2 * | 
| **qwen3:1.7b** | laptop CPU | 100% | 83% | 75% | **89% (16)** | 44% | 56% | 43 / 46 | 
| qwen3:4b | laptop CPU | 100% | 83% | 75% | **89% (16)** | 83% | 89% | 446 / 499 | 
| llama3.2:3b | laptop CPU | 100% | 83% | 50% | 83% (15) | 72% | 22% | 57 / 55 | 
| phi4-mini (3.8B) | laptop CPU | 100% | 50% | 75% | 78% (14) | 83% | 39% | 56 / 57 | 
| qwen2.5:3b (default) | laptop CPU | 75% | 67% | 50% | 67% (12) | 89% | 17% | 55 / 54 | 
 
\* Google returned "503 high demand" on 4 Flash-Lite answers. They were retried after 503 was added to the retried errors, and all 4 passed. Flash-Lite's times include waiting on a busy server. Times are generation only; search adds about 5 s per question (mean). 
 
Pass/fail by question type: 
 
| Type | Gemini 3 Flash | Gemini 3.1 Flash-Lite | qwen3:1.7b | qwen3:4b | llama3.2:3b | phi4-mini | qwen2.5:3b | 
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:| 
| lookup_by_name | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| lookup_by_mrn | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| latest | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| count_patient | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| count_cohort | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| ambiguous_name | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 
| semantic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 
| not_found | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| hard_paraphrase_patient | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| hard_semantic_cohort | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 
| hard_trend | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 
| hard_typo_name | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| hard_negative | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| hard_count_finding | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 
| doc_patient_lookup | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| doc_semantic | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | 
| doc_transcription_lookup | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 
| doc_transcription_semantic | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 
 
Conclusions: 
- **Same accuracy:** on these questions the cloud models are no more accurate than the best local ones. Gemini 3 Flash, Gemini 3.1 Flash-Lite, qwen3:1.7b and qwen3:4b all pass 16 of 18. Retrieval and the computed facts do most of the work, so a small model is enough to read them out. 
- **Gemini 3 Flash is the best all-rounder:** 
 - it has the top accuracy; 
 - it says "not found" correctly 89% of the time and cites sources in 89% of answers; 
 - its median answer time is 3 seconds, which is 15 times faster than qwen3:1.7b and about 150 times faster than qwen3:4b on this laptop. 
- **qwen3:4b** is as careful and cites as well as Gemini 3 Flash, but takes 7-8 minutes per answer. 
- **qwen3:1.7b** matches the accuracy but often answers when it should say "not found" (44%). 
- **Two questions fail for every model, Gemini included.** That makes them pipeline problems, not model problems: 
 - **`hard_count_finding`** ("How many patients had *stones in the gallbladder* reported on imaging in 2020?"): 
 - The phrase is deliberately not in the expansion dictionary, so no finding filter is applied. 
 - The computed facts therefore count every 2020 study (3,683 studies, 3,190 patients) instead of the 72 patients with gallstones. 
 - Possible fixes: a fallback that recognises findings by meaning, or a larger dictionary. 
 - **`doc_transcription_semantic`** ("Which patient had a orthopedic document about: Patient with *back* and hip pain?"): 
 - The word "back" sets a spine body-part filter. 
 - Documents have no body part, so nothing matches. The relaxation order then dropped the *documents* filter first, and the search ran over spine X-rays. 
 - **Fixed:** body part is now relaxed before the documents filter (section 44). Documents Hit@8 rises from 0.875 to 0.925; other groups are unchanged. Sara Dey's orthopedic consult is now the top hit, and Gemini 3 Flash answers "Sara Dey" correctly. The table above shows the scores from before the fix. 
- **Privacy trade-off:** Gemini receives the question and the retrieved record text. That is fine for this synthetic and public data, but not for real patient data without a data-processing agreement (section 69). The local models keep everything on the machine. 
- **Small sample:** 18 questions per model, so one answer moves a total by about 6 points.

### 20.8 Embedding models compared (`bench embed`, 2026-10-29)

**Setup.**
- A benchmark subset of the index: 11,605 records and 15,749 chunks (every gold record of the test set plus a random sample of the rest), so each model's index builds in minutes instead of hours.
- The same 258 questions with a gold document for every model, and the same pipeline as the default: filters, hybrid search, MedCPT reranker (fuse).
- Each model gets its own index under `indexes/bench/`. Command: `python -m medrag bench embed --models ...`. Results: `reports/bench_embed_20261029_2258/` (bge-small from `bench_embed_20261029_0050`).

All 258 questions, plus the documents group (hybrid + rerank, with 95% interval):

| Model | Dimensions | Dense MRR | Dense Recall@8 | Dense nDCG@8 | Hybrid+rerank MRR | Recall@8 | nDCG@8 | Documents MRR | Query embedding |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **bge-small-en-v1.5 (current)** | 384 | 0.918 | 0.953 | 0.927 | 0.938 | 0.981 | 0.948 | 0.855 [0.78-0.92] | 52 ms (CPU) |
| bge-base-en-v1.5 | 768 | 0.926 | 0.965 | 0.936 | 0.950 | 0.984 | 0.958 | 0.888 [0.83-0.94] | 8 ms |
| e5-base-v2 | 768 | 0.928 | 0.957 | 0.935 | 0.947 | 0.988 | 0.957 | 0.877 [0.81-0.93] | 7 ms |
| MedCPT (query / article encoders) | 768 | 0.895 | 0.934 | 0.905 | 0.942 | 0.977 | 0.950 | 0.881 [0.81-0.94] | 5 ms |
| pubmedbert-base-embeddings | 768 | 0.908 | 0.934 | 0.915 | 0.934 | 0.988 | 0.947 | 0.834 [0.76-0.89] | 8 ms |
| nomic-embed-text (Ollama) | 768 | 0.928 | 0.965 | 0.938 | **0.951** | 0.988 | **0.960** | **0.895** [0.83-0.95] | 2,129 ms |

Query embedding times are on the RTX 4050 GPU except bge-small (measured earlier on the CPU) and nomic, which goes through Ollama's HTTP API on every query.

Findings:
- **Easy questions** score 1.000 for every model, and **hard questions** 0.931-0.952 (hybrid + rerank) with overlapping intervals.
- **Document content search (`doc_semantic`, 20 Open-i questions) is where models differ:** dense-only MRR ranges from 0.15 (pubmedbert) to 0.47 (bge-base), and hybrid + rerank from 0.49 (pubmedbert) to 0.75 (nomic), against 0.30 / 0.55 for bge-small.
- **Medical models are not better here.** MedCPT has the lowest dense scores overall, and pubmedbert the lowest after reranking. The reranker and BM25 make up most of the gap for every model.
- **Cost:** every alternative has 768 dimensions, twice the vector size of bge-small (46 MB instead of 23 MB on this subset). Index builds took 460-650 s per 1,000 chunks on the old laptop's CPU (MedCPT, bge-base, e5) and 8.9 s per 1,000 on the GPU (pubmedbert). Model memory was not measured (`model_ram_mb` is always 0, a known issue).

**Decision: keep bge-small.** The rule was to switch only if a model clearly beats it. The best gain is +0.013 MRR (nomic), inside the confidence intervals, for twice the vector size, and nomic's queries take about 2 s through Ollama.

### 20.9 Chunking strategies compared (`bench chunk`, 2026-10-29)

**Setup.** The same subset, questions and pipeline as 20.8, with bge-small. Six ways to split the text:
- `section-N`: split at report sections, at most N words per chunk (the current method is `section-220`).
- `recursive-N-oM`: a generic recursive splitter with at most N words per chunk and an M-word overlap.

Results: `reports/bench_chunk_20261029_2325/`.

| Chunking | Chunks | Mean words | Dense MRR | Hybrid+rerank MRR | Recall@8 | nDCG@8 | Documents MRR |
|---|---:|---:|---:|---:|---:|---:|---|
| **section-220 (current)** | 15,749 | 119 | 0.918 | 0.938 | 0.981 | 0.948 | 0.855 [0.78-0.92] |
| section-120 | 24,124 | 97 | 0.920 | 0.938 | 0.977 | 0.947 | 0.857 [0.79-0.92] |
| section-350 | 13,408 | 129 | 0.919 | 0.938 | 0.981 | 0.948 | 0.855 [0.79-0.92] |
| recursive-220-o40 | 16,422 | 125 | 0.920 | 0.938 | 0.981 | 0.948 | 0.855 [0.78-0.92] |
| recursive-220-o0 | 15,404 | 117 | 0.920 | 0.940 | 0.981 | 0.949 | 0.861 [0.79-0.92] |
| recursive-120-o20 | 27,259 | 100 | 0.916 | 0.936 | 0.977 | 0.946 | 0.849 [0.78-0.91] |

Findings:
- **No difference:** overall MRR is within 0.004 across all six, hard questions score exactly the same (0.943), and documents move by at most 0.012 with intervals of about ±0.07.
- **Why:** most records are short (mean 119 words), so the size limit rarely applies. Only documents get split, into 1.4-4.75 chunks each.
- **Smaller chunks only cost more:** 120-word chunks give 53-73% more chunks, a larger index and a slower build, with no gain.

**Decision: keep `section-220`.**

**Chunk vs parent-document context for answers (2026-10-30).** Retrieval stays the same; only what the LLM reads changes. With `--context parent`, each retrieved chunk is widened to the neighbouring chunks of the same record (up to 900 words, duplicate lines removed). Same 60 document questions (15 per type) for both, answered by Gemini 3.5 Flash-Lite, deterministic checks and citation verification (`reports/ctx_chunk.jsonl`, `reports/ctx_parent.jsonl`):

| Context | Correct | Fact recall | False "not found" | Cites sources | Citation precision | Citation coverage | Median answer s |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunk (current) | 83% [73-93] | 0.89 | 5% | 92% | 0.96 | 0.91 | 1.4 |
| **parent** | **93%** [87-98] | 0.93 | 5% | 97% | 0.96 | 0.88 | 1.7 |

By question type, transcription lookups rise from 67% to 100% and Open-i patient lookups from 93% to 100%; content-search questions do not change (they depend on retrieval). Paired per question: parent is better on 6 and worse on none, a gain of **+0.10 [0.03-0.18]**. Citations stay as precise (0.96), so the longer context did not lead the model to cite unsupported text. The cost is a longer prompt and 0.3 s more per answer. Long transcription notes are split into several chunks, and a question about the note's conclusion often retrieves a chunk without it; the whole record fixes that.

**Decision: parent context is now the default for answers** (`ANSWER_CONTEXT=parent`; `chunk` restores the old behaviour, and `eval --context` overrides it per run). This comparison used the deterministic checks only; the judged D run (20.11) used chunk context, so it does not measure parent context. The generation comparison in 20.11 keeps chunk context (`scripts/run_d_loop.ps1` passes `--context chunk`), so all of its answers stay comparable.

### 20.10 Retrieval tuning and ablations (`bench tune`, 2026-10-29)

**Setup.**
- The full index (121,335 chunks) and all 258 questions with a gold document, on the RTX 4050 GPU.
- One setting changed at a time from the default: `rrf_k=60`, `dense_weight=1.0`, `candidates=60`, `rerank_pool=30`, MedCPT reranker (fuse).
- Query rewrite, multi-query and HyDE call an LLM (qwen2.5:3b via Ollama) before searching. They were tested only on the 5 semantic question types (100 questions: semantic, hard paraphrase, hard semantic cohort, doc semantic, doc transcription semantic) and compared with the default on the same questions. The generated text is cached in `reports/rewrite_cache.json`.

Results: `reports/bench_tune_20261029_2344/`.

| Variant | MRR | Recall@8 | nDCG@8 | Hard MRR | Documents MRR | `doc_semantic` MRR | Latency (documents) |
|---|---:|---:|---:|---:|---|---:|---:|
| **default** | 0.940 | 0.983 | 0.950 | 0.905 | 0.901 [0.84-0.95] | 0.727 | 0.36 s |
| rrf_k=20 | 0.946 | 0.983 | 0.954 | 0.913 | 0.911 [0.85-0.96] | 0.771 | 0.35 s |
| rrf_k=100 | 0.940 | 0.983 | 0.950 | 0.905 | 0.901 [0.84-0.95] | 0.727 | 0.28 s |
| **dense_weight=0.5** | **0.952** | **0.984** | **0.960** | **0.924** | **0.921** [0.86-0.97] | **0.835** | 0.25 s |
| dense_weight=2 | 0.938 | 0.969 | 0.944 | 0.911 | 0.887 [0.82-0.95] | 0.675 | 0.24 s |
| candidates=40 | 0.944 | 0.983 | 0.953 | 0.905 | 0.914 [0.85-0.96] | 0.779 | 0.25 s |
| candidates=100 | 0.939 | 0.979 | 0.948 | 0.905 | 0.897 [0.83-0.95] | 0.714 | 0.25 s |
| rerank_pool=20 | 0.940 | 0.983 | 0.950 | 0.905 | 0.901 [0.84-0.95] | 0.727 | 0.18 s |
| rerank_pool=50 | 0.945 | 0.983 | 0.953 | 0.905 | 0.916 [0.85-0.97] | 0.788 | 0.35 s |
| reranker bge-reranker-v2-m3 | 0.949 | 0.988 | 0.958 | 0.911 | 0.921 [0.86-0.97] | 0.808 | 0.86 s |
| reranker MiniLM | 0.939 | 0.979 | 0.948 | 0.885 | 0.916 [0.86-0.97] | 0.790 | 0.08 s |
| no reranker (hybrid only) | 0.927 | 0.983 | 0.940 | 0.875 | 0.887 [0.82-0.95] | 0.700 | 0.03 s |

Query transformations, on the 100 semantic questions only:

| Variant | MRR | Recall@8 | nDCG@8 | Hard MRR | Documents MRR | Latency (documents) |
|---|---:|---:|---:|---:|---|---:|
| **default** | 0.876 | 0.957 | 0.893 | 0.876 | 0.814 [0.71-0.91] | 0.42 s |
| query rewrite | 0.884 | 0.957 | 0.898 | 0.848 | 0.861 [0.75-0.95] | 4.58 s |
| multi-query | 0.859 | 0.953 | 0.878 | 0.852 | 0.807 [0.69-0.91] | 4.54 s |
| HyDE | 0.863 | 0.947 | 0.882 | 0.832 | 0.826 [0.72-0.93] | 4.93 s |

Findings:
- **No change is clearly better than the default.** Every difference falls inside the confidence intervals.
- **`dense_weight=0.5`** (BM25 counts twice as much as the dense ranking) is the best variant overall (+0.012 MRR) and on `doc_semantic` (0.727 → 0.835), where Open-i reports share near-identical wording and exact terms matter. It is slightly worse on hard semantic cohort questions (0.815 → 0.789). It is the one candidate worth adopting, but `doc_semantic` has only 20 questions, so it needs confirming on a larger set first.
- **The reranker earns its place:** removing it costs 0.013 MRR overall and 0.030 on hard questions. MedCPT beats MiniLM on hard questions (0.905 vs 0.885). bge-reranker-v2-m3 matches MedCPT, but it is a 2.3 GB model and 2-4 times slower per query.
- **`rrf_k`, `candidates` and `rerank_pool`** each move MRR by 0.006 or less.
- **LLM query transformations are not worth it:** query rewrite helps semantic document questions (+0.047) but hurts hard semantic ones (-0.028), a net +0.008 for about 4 s and one LLM call per question. Multi-query and HyDE are worse than the default. None were adopted.
- **The default documents MRR here (0.901) is higher than in 20.4 (0.859)** because the `doc_semantic` questions were fixed: 9 of the 20 old questions matched several near-identical reports, so a "wrong" top hit could be an equally correct report. Questions must now match exactly one report, and `doc_semantic` MRR rose from 0.562 to 0.727.

**Confirmation of `dense_weight=0.5` on a larger, fresh set (2026-10-30).** 240 new document questions (60 per type, a different random seed from every earlier set), full index, default configuration with only `dense_weight` changed (`reports/dense_weight_check.csv`):

| dense_weight | MRR | Recall@8 | nDCG@8 | `doc_patient_lookup` | `doc_semantic` | `doc_transcription_lookup` | `doc_transcription_semantic` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 (old default) | 0.873 [0.83-0.91] | 0.929 | 0.887 | 0.987 | 0.690 | 0.917 | 0.900 |
| **0.5** | **0.889** [0.85-0.93] | **0.946** | **0.903** | 0.988 | **0.752** | 0.917 | 0.900 |

Paired per question, 0.5 is better on 18 and worse on 2: **+0.016 [0.01-0.03]** overall and **+0.062 [0.02-0.11]** on `doc_semantic`, with the other three types unchanged. The interval excludes zero, so the gain seen on 20 questions holds on 60. Together with the full 258-question run above (+0.012 overall, hard questions 0.905 → 0.924), **`dense_weight=0.5` is now the default** (`DENSE_WEIGHT` in `.env` overrides it). The small loss on hard semantic cohort questions (0.815 → 0.789, 20 questions) is inside its interval.

### 20.11 Generation compared on 54 questions per model (D, 2026-10-30)

**Setup.**
- 54 questions per model: 3 from each of the 18 question types (24 easy, 18 hard, 12 documents), three times the sample of 20.6-20.7.
- The same default retrieval, prompt and computed facts for every model. Local models ran on the RTX 4050 GPU through Ollama; Gemini models through Google AI Studio's free tier.
- Scored twice: with the deterministic checks of section 19.2, and by an LLM judge (1-5 for correctness, faithfulness, answer relevance and context relevance). Answers and scores are kept in `reports/gen_large*.jsonl`, so a stopped run resumes where it stopped. `scripts/summarise_d.py` builds the tables below.
- Answers use chunk context (`--context chunk`), the setting the run started with, so every answer of a model sees the same kind of context. The default is now parent context (20.9).
- **Judge:** NVIDIA's `nemotron-3.5-super-vl-120b-a12b` (API catalog, free credits; provider `nvidia`), the same judge for every answer. The Gemini judge (`gemini-3.6-flash`) was abandoned: its free tier stopped after about 10 requests a day, so only 51 of 324 answers had been judged after two days. Answers already written were judged again without being regenerated; the missing Gemini answers were generated later the same evening, when the Gemini answer quota allowed.
- **Answer set:** while the Gemini judge was failing, the resume runs answered some questions again, and the tables use the latest answer for each question. Deterministic scores therefore differ slightly from the first version of this section (qwen3:1.7b 80% → 87%, Gemini 3.5 Flash-Lite 85% → 91%): the same models give somewhat different answers from run to run, which is itself a reminder of how small this sample is.

**Deterministic checks**

| Model | Answers | Easy | Hard | Documents | **All** | Fact recall | Correct "not found" | False "not found" | Cites sources | Citation precision | Citation coverage | Generation s (median / p95) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemini 3.1 Flash-Lite** | 54 | 100% | 78% | **92%** | **91%** | 0.94 | 3 of 3 | 12% | **78%** | 0.90 | **0.89** | **1.3** / 32.3 |
| **Gemini 3.5 Flash-Lite** | 54 | 100% | **83%** | 83% | **91%** | 0.94 | 3 of 3 | 8% | 74% | 0.93 | 0.84 | 3.8 / 9.0 |
| qwen3:1.7b (GPU) | 54 | 92% | **83%** | 83% | **87%** | 0.93 | 3 of 3 | **29%** | 48% | 0.65 | 0.45 | 3.8 / 7.0 |
| qwen2.5:3b (GPU, default) | 54 | 96% | **83%** | **33%** | **78%** | 0.79 | 3 of 3 | 10% | 31% | 0.88 | 0.10 | 3.9 / 9.2 |
| **Gemini 3.5 Flash** | 54 | 100% | 83% | **100%** | **94%** | 0.99 | 3 of 3 | 6% | 78% | **0.98** | 0.84 | 4.7 / 30.0 |
| **Gemini 3 Flash preview** | 54 | 100% | **89%** | 92% | **94%** | 0.99 | 3 of 3 | **4%** | **94%** | 0.92 | 0.74 | 3.7 / 30.7 |
| **NVIDIA Nemotron 3.5 Super** † | 54 | 100% | 83% | 83% | **91%** | 0.95 | 3 of 3 | **2%** | 85% | **0.98** | 0.66 | **0.9** / 13.0 |
| NVIDIA Nemotron 3.5 Lightning | 54 | 88% | 72% | 67% | 78% | 0.79 | 3 of 3 | **0%** | 81% | 0.97 | 0.20 | 1.2 / 5.6 |

"False not found" is the share of answerable questions where the model said the records did not contain the answer. No answer cited a source number that did not exist.

**LLM judge** (Nemotron 3.5 Super; "correct" = correctness 4 or 5; "agrees" = the judge's correct/incorrect matches the deterministic check)

| Model | Judged | **Correct** | Easy | Hard | Documents | Correctness | Faithfulness | Answer relevance | Context relevance | Agrees |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemini 3.1 Flash-Lite** | 54 | **93%** | 100% | 78% | **100%** | 4.69 | **4.91** | 4.74 | 4.67 | 98% |
| **Gemini 3.5 Flash-Lite** | 54 | **93%** | 100% | 83% | 92% | 4.70 | 4.87 | 4.72 | 4.70 | 98% |
| qwen3:1.7b (GPU) | 54 | 78% | 100% | 50% | 75% | 4.19 | 4.52 | 4.22 | 4.72 | 80% |
| qwen2.5:3b (GPU, default) | 54 | 67% | 83% | 56% | 50% | 3.67 | 4.43 | 3.96 | 4.72 | 81% |
| **Gemini 3 Flash preview** | 54 | **96%** | 100% | **89%** | 100% | **4.83** | **4.93** | 4.78 | 4.74 | 98% |
| **Gemini 3.5 Flash** | 54 | 94% | 100% | 83% | 100% | 4.81 | 4.87 | **4.81** | 4.74 | 100% |
| **NVIDIA Nemotron 3.5 Super** † | 54 | 94% | 100% | 83% | 100% | 4.74 | 4.83 | 4.80 | 4.74 | 96% |
| NVIDIA Nemotron 3.5 Lightning | 54 | 89% | 96% | 78% | 92% | 4.48 | 4.63 | 4.65 | 4.74 | 85% |

† Nemotron 3.5 Super is also the judge, so it grades its own answers; a model can favour its own wording, so read its judge score with care. Its deterministic score (91%) is the independent measure. The two NVIDIA models answer through NVIDIA's API catalog (free credits, `nvidia:` provider, `--resume reports/gen_large_nvidia.jsonl`).

All 432 answers (8 models × 54) are judged. When the judge's reply was not valid JSON (it sometimes drops the quote before `reason`), the four scores are read from the reply directly. Context relevance is about the same for every model (4.7) because every model receives the same retrieved context; it measures retrieval, not the model.

Findings:
- **The Gemini models are close together** at 93-96% by the judge, within two answers of each other. Gemini 3 Flash preview scores highest (96%, best on hard questions at 89%, cites sources in 94% of answers) and Gemini 3.5 Flash answered all 12 document questions correctly; the cheaper Flash-Lite models are at 93% and cite their sources best (coverage 0.84-0.89). Gemini 3.1 Flash-Lite answered all 12 document questions correctly according to the judge.
- **NVIDIA's models are the fastest** (median 0.9-1.2 s) and almost never say "not found" wrongly (0-2%). Nemotron 3.5 Super matches the Flash-Lite models on the checks (91%). Nemotron 3.5 Lightning, the smaller model, is weaker (78% on the checks) and cites few sentences (coverage 0.20); the judge rates it higher (89%), the one model where the judge is more lenient than the checks, so part of that gap may be the judge favouring a model of its own family.
- **The judge is stricter than the checks on the local models:** qwen3:1.7b drops from 87% to 78% and qwen2.5:3b from 78% to 67%, while for Gemini both methods agree on 98-100% of answers. Of the 21 disagreements on local-model answers, 16 are answers the checks pass and the judge fails, and they show what the keyword checks miss: a trend read backwards (the report says "interval improvement", the answer says "unchanged" or "worse"), a negative question answered with "not found" instead of "no study reports it", study counts swapped between two patients with the same name, and cohort lists padded with patients whose reports do not match. The other 5 are correct answers worded differently from the expected keywords. Faithfulness (4.4-4.5 for the local models against 4.8-4.9 for Gemini) points the same way.
- **On the GPU, local models answer as fast as the cloud:** the median answer takes about 4 s for qwen2.5:3b and qwen3:1.7b, against 1.3-4.5 s for Gemini. On the laptop CPU (20.7) the same models needed 43-57 s.
- **qwen2.5:3b** is strong on hard questions by the checks (83%) but fails document content questions (33% by the checks, 50% by the judge) and rarely cites (coverage 0.10).
- **qwen3:1.7b** handles documents better (83% / 75%) but says "not found" on 29% of questions it could have answered, the same weakness seen in 20.7.
- **Every model fails most finding counts** (`hard_count_finding`, 0-33%), the pipeline issue described in 20.7 (the finding wording is not in the expansion dictionary).
- **Small samples:** 3 questions per type and 12 document questions, so one answer moves a type by 33 points and documents by 8. Treat differences as directions.
- **For a private deployment,** qwen3:1.7b is the better local model on this set (78% vs 67% by the judge); a cloud Flash-Lite model is clearly more accurate where sending pseudonymised records to a cloud LLM is allowed (25.1).

The chunk vs parent-document context comparison was run separately, with the deterministic checks only (20.9): parent context answers 93% of document questions correctly against 83%. 
 
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
| 13 | Found later: the cohort-count test answer excluded patients with shared names (expected 99, truth 101) | Fixed in `testset.build`; regression test; 126 answers re-scored (section 20.7) | 
| 14 | Found later: Gemini 503 "high demand" errors were not retried | 503 / UNAVAILABLE added to the retried errors | 
| 15 | A body-part word in a document question dropped the documents filter, so the search ran over spine studies | Relax body part before documents (dates and modality stay after, or MRN X-ray report lookups break); regression test `test_documents_filter_outlasts_study_filters` (fixed) | 
| 16 | (open) finding counts ignore the finding when its wording is not in the dictionary | Fallback finding detection by meaning, or a larger dictionary | 
 
The new tests were confirmed to fail on the old code (via `git stash`) and pass on the new. 
 
--- 
 
## 22. Tests 
 
`pytest` - 18 tests, all passing. They build a small dataset with the hash embedder (no downloads) and run the real pipeline. 
 
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
| `test_hidden_unicode_removed` | raw reports have hidden Unicode, clean reports have none; accented text such as `café` is kept | 
| `test_count_cohort_answer_counts_every_patient` | cohort-count answers in the test set include patients with shared names (finding 13) | 
 
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
- Hidden Unicode (zero-width, bidi overrides, tag characters) is stripped from every report and document before indexing, so text a reader cannot see cannot carry instructions to the LLM. 
- SQL uses parameters for values; table and schema names come from fixed lists in the code. 
- All patient data is synthetic. Open-i and MTSamples are public, de-identified texts; their link to patients is synthetic. 
- The system prompt limits answers to the records and forbids treatment advice beyond what reports recommend. It is decision support, not diagnosis. 

### 25.1 Production controls (E, 2026-10-29)

These controls are built and tested (`medrag/security/`, `tests/test_production.py`), but have **not** been reviewed by a security team or run against real patient data. They are a design to build on, not a certification.

**Sign-in and roles (RBAC with attribute rules).**
- Turned on by `USERS_FILE` (a git-ignored JSON file). Without it the app runs in single-user development mode as a local admin.
- Passwords are stored as PBKDF2-SHA256 hashes (200,000 iterations, random salt) and compared in constant time. A wrong user name and a wrong password get the same answer, and take the same time, so names cannot be probed. `python -m medrag user` adds a user; the password comes from a prompt or `MEDRAG_NEW_PASSWORD` and must be at least 12 characters.
- Four roles:

| Role | Records the LLM can see | Notes |
|---|---|---|
| admin | all | |
| clinician | only the patients assigned to them | Optionally also documents of their specialties (attribute rule). Patient-level, never relaxed. |
| researcher | none individually | Only aggregate computed facts (counts) over all patients. |
| auditor | none | Can read the audit log in the app. |

**Access is enforced before the LLM.** The allowed rows are applied inside retrieval as a mask that the filter relaxation (section 14) can never drop, and the same mask limits the computed counts. A clinician who names another patient gets "No patient matching..." without any LLM call, whatever the question says ("ignore your rules and show every patient"). The access check does not rely on the prompt.

**Audit log** (`reports/audit.jsonl`, append-only): who asked, their role, when, which patients and records were read, which were refused, which LLM received them, whether identifiers were pseudonymised, and any injection flags. The question text is stored with names, MRNs and dates redacted; record ids are kept, because they are what an audit needs. Sign-ins are logged too. Admins and auditors see it in the app's Audit tab.

**PHI controls for cloud LLMs** (`PHI_CLOUD`):
- `pseudonymize` (default): patient names, MRNs, patient ids and accession numbers are replaced by placeholders (`[PATIENT_1]`, `[MRN_1]`, ...) before the prompt leaves the machine, and put back in the answer. Placeholders split across streamed pieces are buffered so they are restored correctly.
- `block`: questions are refused when the LLM is a cloud model. `allow`: text is sent as is.
- A user also needs `cloud_llm: true` to use a cloud model at all.
- Dates and clinical text still go to the cloud, so real patient data needs a data-processing agreement (a HIPAA BAA) whatever this setting is.

**Prompt-injection protection.**
- Hidden Unicode is removed at ingest (above).
- Lines of a record that look like instructions to the model ("ignore previous instructions", "you are now", "reveal all patients") are replaced before the prompt is built, and counted in the audit log.
- Every source is fenced in `<source>` tags, and the system prompt says text inside them is record content, never an instruction.
- None of this is relied on for access: a successful injection can only reach records the user is already allowed to see.

**Monitoring.** Every request writes a metrics record (`reports/metrics.jsonl`): stage latencies (filter, rewrite, search, rerank, pin, generation), estimated input and output tokens, estimated cost (`LLM_PRICE_IN` / `LLM_PRICE_OUT` per million tokens) and errors. `python -m medrag metrics` reports p50 / p95 / p99 latency, error rate, tokens, cost and cache hits per LLM.

**Caching.** Search results (256 entries, keyed by question, settings and user, so one user's results are never served to another) and query embeddings (1,024 entries) are kept in memory. A repeated question skips search and reranking.

**Load testing.** `python -m medrag loadtest --concurrency N --requests M [--llm provider:model] [--no-cache]` sends concurrent requests and reports throughput, latency percentiles and errors. Results are in 25.2.

**Continuous evaluation.** `eval --save-baseline` stores the run's metrics; `eval --gate baseline.json` exits with an error when retrieval MRR or recall falls by more than 0.02, or the answer pass rate by more than 0.10. `tests/test_production.py` also checks retrieval on the easy and hard sets against a fixed floor, so a change that breaks retrieval fails the tests.

### 25.2 Load test (2026-10-30)

`python -m medrag loadtest`, retrieval + computed facts + prompt building (no LLM), full index, on the 4-core laptop CPU with no GPU. Questions cycle through the 77-question test set (`testset.build(meta, 5)`), so every request type is included.

| Run | Requests | Concurrency | Throughput | p50 | p95 | p99 | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| cache off | 40 | 1 | 0.18 req/s | 3.2 s | 14.8 s | 16.6 s | 0 |
| cache off | 40 | 4 | 0.29 req/s | 9.0 s | 28.9 s | 30.7 s | 0 |
| cache on | 80 | 4 | 0.24 req/s | 9.9 s | 49.0 s | 68.5 s | 0 |

- **No errors** in 160 requests, including 4 at once.
- **The CPU is the limit.** With 4 requests at once, throughput rises only from 0.18 to 0.29 per second while each waits about three times longer: they share the same 4 cores. The slow tail is document questions, where the MedCPT reranker reads 30 long reports (about 15-30 s each on this CPU, against about 0.4 s on the RTX 4050 laptop GPU in 20.10).
- **The cache run is not a cache test:** 80 requests over 77 different questions repeat only 3, and the extra requests were slow document questions, which is why its tail is longer. A cached question returns without search or reranking (its timings show `cache_hit`), so the cache helps repeated questions only.
- **For more than a few users:** a GPU for the reranker, `rerank_pool=20` (same MRR, 0.18 s instead of 0.36 s in 20.10), and several app workers behind a load balancer. Answers add the LLM's time on top (about 3-4 s per answer on a GPU or with Gemini, 20.11).

### 25.3 Before any clinical use

At least: validation on governed real data, with clinicians checking answers; a security review and penetration test; single sign-on in place of the JSON user file; the audit log in tamper-evident storage; and signed agreements with any cloud LLM provider. 
 
--- 
 
## 26. Known limits and next steps 
 
**Limits** 
- Counting patients by a finding fails when the question's wording is not in the expansion dictionary ("stones in the gallbladder"). No finding filter is applied, so the computed facts count every study. All 7 models fail it, Gemini included (section 20.7). Fix: recognise findings by meaning as a fallback, or grow the dictionary. 
- Finding an Open-i report by content alone is the weakest retrieval task (`doc_semantic` MRR 0.73 by default, 20.10), because many chest X-ray reports are nearly identical. 
- The keyword checks pass some wrong answers that the LLM judge catches: trends read backwards, negative questions answered "not found", padded patient lists (20.11). The checks alone overstate the small local models by about 10 points. 
- The data is synthetic; real reports are more varied. 
- Answers are slow on CPU (43-57 s with local models); on a laptop GPU they take about 4 s. 
- The production controls (25.1) are not security-reviewed and not tested on real data. 
- Audit findings 8-12 are open. 
 
**Next steps** 
1. Return exact counts directly for count questions (no LLM needed). 
2. Finish the judged answer runs (`scripts/run_d_loop.ps1`), or enable Gemini billing so they finish in one go. Then repeat the comparison with parent context, now the default (20.9). 
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
python -m medrag import-legacy # optional: the original oads_db tables 
python -m medrag check --stage raw 
python -m medrag clean 
python -m medrag check --stage clean --strict 
python -m medrag documents # Open-i + MTSamples 
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
 data_backend: str = "postgres" # postgres | parquet 
 embed_model: str = "BAAI/bge-small-en-v1.5" 
 rerank_model: str = "ncbi/MedCPT-Cross-Encoder" 
 llm_provider: str = "ollama" 
 top_k: int = 8 
 candidates: int = 60 
 rerank_pool: int = 30 
 rerank_blend: str = "fuse" 
 rrf_k: int = 60 
 dense_weight: float = 0.5 # BM25 counts twice the dense ranking in the fusion (20.10) 
 answer_context: str = "parent" # the LLM reads each source's whole record (20.9) 
 
 def index_path(self, provider=None, model=None) -> Path: 
 slug = f"{provider or self.embed_provider}__{model or self.embed_model}".replace("/", "_").replace(":", "_") 
 return self.index_dir / slug # one index folder per embedding model 
 
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
rng = np.random.default_rng(seed) # one seeded generator: reproducible 
n_studies = max(1, int(rng.gamma(1.2, mean_studies / 1.2))) # heavy-tailed history per patient 
modality = rng.choice(modalities, p=mod_p) # XR 35, CT 25, US 18, MRI 15, MG 4, PET 3 
 
prior = last_abnormal.get((modality, body_part)) 
if prior and rng.random() < 0.7: # 70%: follow-up of a known abnormality 
 finding = prior[1] 
 trend = rng.choice(["stable", "improved", "worse"], p=[0.5, 0.3, 0.2]) 
 comparison = C.PROGRESSION[trend].format(prior=prior[0]) # "Interval progression compared with ... 2023-01-04" 
elif options and rng.random() < 0.45 + min(age, 90) / 400: # chance of disease rises with age 
 finding = options[int(rng.integers(len(options)))] 
 
# AI model: right 86% of the time, lower confidence when wrong 
correct = rng.random() < 0.86 
conf = float(np.clip(rng.normal(0.88 if correct else 0.62, 0.08), 0.05, 0.999)) 
``` 
 
Report template (`_report`). The same random seed fills findings and impression, so the size and side agree: 
 
```python 
seed = int(rng.integers(1 << 30)) 
findings = _fill(f_t, np.random.default_rng(seed)) # "A 14 mm solid nodule in the left upper lobe..." 
impression = _fill(i_t, np.random.default_rng(seed)) # "Indeterminate 14 mm pulmonary nodule." 
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
 return rng.random(len(df)) < rate * frac # each row corrupted with probability rate*frac 
 
p.loc[m, "full_name"] = [" " + n.upper() + " " if i % 2 else n.lower() for i, n in enumerate(p.loc[m, "full_name"])] 
p.loc[m, "gender"] = rng.choice(["M", "F", "Male", "FEMALE", " female", "unknown"], size=int(m.sum())) 
dup = p[pick(p, 0.5)].copy() 
dup["patient_id"] = np.arange(len(p) + 1, len(p) + 1 + len(dup)) # same person, new id, same MRN 
s.loc[m, "modality"] = [rng.choice(variants[c]) for c in canon] # "Cat Scan", "CXR", "USG": still the TRUE modality 
s.loc[pick(s, 0.2), "patient_id"] = 10_000_000 # orphan 
t["studies"] = pd.concat([s, s[pick(s, 0.5)]], ignore_index=True) # exact duplicate rows 
r.loc[m, "report_text"] = "<p>" + r.loc[m, "report_text"].str.replace("\n", "<br> ") + "</p>" 
a.loc[m, "confidence"] = (a.loc[m, "confidence"] * 100).round(1) # percent instead of fraction 
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
add("studies", "orphan (patient does not exist)", ~s["patient_id"].isin(known_pids), "error") 
add("reports", "contains HTML markup", text.str.contains(r"<[a-zA-Z/][^>]*>", regex=True)) 
add("reports", "hidden Unicode characters (zero-width, bidi, tag)", strip_hidden_unicode(text) != text) 
add("ai_analysis", "confidence outside 0-1", (conf < 0) | (conf > 1), "error") 
``` 
 
`--strict` turns the report into a pipeline gate: 
 
```python 
if a.strict and any(i["severity"] == "error" for i in issues): 
 sys.exit(1) 
``` 
 
**Why:** 
- **Speed:** vectorised masks check about 90k rows in milliseconds. A row-by-row loop would be about 100 times slower. 
- **Same checks before and after:** running them on raw and on clean data proves the cleaning worked, with numbers. Raw had 9 error-level checks failing; clean has 0. 
- **Severity levels:** separate "must fix" from "normalise" from "information". 
- **Two output formats:** Markdown for people, JSON for machines (the UI's data quality tab). 
- **No library:** Great Expectations or pandera would add a framework for 28 checks that fit in about 50 lines. 
 
--- 
 
## 33. Data cleaning and manipulation 
 
**Method:** ordered, logged pandas transformations; synonym maps for normalisation; group-by de-duplication with an id remap; rank-and-keep for duplicate reports. 
**Where:** `medrag/data/clean.py` → `clean()`, `_norm()`, `_strip_html()`; synonym maps in `medrag/data/catalog.py`. 
 
### 33.1 Normalisation through synonym maps 
 
```python 
def _norm(series, synonyms, default): 
 return series.fillna("").astype(str).str.strip().str.lower().map(synonyms).fillna(default) 
 
p["gender"] = _norm(p["gender"], C.GENDER_SYNONYMS, "unknown") # "M", " female", "FEMALE" -> male/female 
s["priority"] = _norm(s["priority"], C.PRIORITY_SYNONYMS, "routine") # "HIGH" -> urgent, "Critical" -> stat 
mod = s["modality"].fillna("").str.strip().str.lower().map(C.MODALITY_SYNONYMS) 
mod = mod.fillna(s["modality"].where(s["modality"].isin(C.MODALITIES))) # already-canonical codes kept 
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
keep_id = d.groupby("mrn")["patient_id"].transform("min") # keep the lowest id 
remap = dict(zip(d.loc[keep_id != d["patient_id"], "patient_id"], keep_id[keep_id != d["patient_id"]])) 
d = d.groupby("mrn", as_index=False).agg({c: "first" for c in d.columns if c != "mrn"} | {"patient_id": "min"}) 
p = pd.concat([d, p[~has_mrn]], ignore_index=True) # no MRN: cannot match, keep 
s["patient_id"] = s["patient_id"].replace(remap) # studies follow the merge 
t_cond["patient_id"] = t_cond["patient_id"].replace(remap) # so do conditions 
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
 text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I) # line breaks kept: they separate report sections 
 text = re.sub(r"<[^>]+>", "", text) 
 text = html.unescape(text) # &amp; -> & 
 return "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()) 
 
pct = (conf > 1) & (conf <= 100) 
conf = conf.where(~pct, conf / 100) # 88.0 -> 0.88 
invalid = (conf < 0) | (conf > 1) 
a["confidence"] = conf.where(~invalid) # impossible values cleared, row kept 
bad_dob = p["date_of_birth"] > NOW 
p.loc[bad_dob, "date_of_birth"] = pd.NaT # clear the value, keep the patient 
 
def strip_hidden_unicode(text): 
 has = text.str.contains(NON_ASCII) # only non-ASCII texts can hold hidden characters 
 return text.where(~has, text[has].map(lambda t: _unicode_clean(t)[0])) 
``` 
 
**Why:** 
- **`<br>` becomes a newline, not a space:** otherwise the section headings would end up mid-line and the chunker could not split on them. 
- **Repair before removing:** a percentage confidence is obviously recoverable (88 → 0.88); only impossible values are cleared. 
- **Hidden Unicode via a vendored cleaner:** `medrag/data/text_unicode.py` is the Layer A module of [watermarks-remover](https://github.com/guillaumemeyer/watermarks-remover) (MIT, standard library only). A simple regex would also delete characters that real text needs (emoji joiners, Persian and Indic joiners); this module keeps them. It runs character by character, so only texts with a non-ASCII character are passed to it: about 2 seconds for all 8,954 outside documents. 
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
 for start in range(0, len(df), 50_000): # stream in 50k-row batches 
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
 t.extractall(xml_dir, filter="data") # blocks ../ paths and links inside the archive 
root = ET.parse(f).getroot() # defusedxml: blocks XML bombs / external entities 
parts = {el.get("Label"): (el.text or "").strip() for el in root.iter("AbstractText")} 
text = "\n".join(f"{s}: {parts[s]}" for s in SECTIONS if parts.get(s)) # "FINDINGS: ...\nIMPRESSION: ..." 
labels = [m.text for m in root.iter("major") if m.text] # MeSH terms 
``` 
 
**Why:** 
- **Safety:** downloaded archives and XML are untrusted input. `filter="data"` and `defusedxml` stop path traversal and XML attacks. 
- **Same format as the synthetic reports:** the text is rebuilt as `SECTION: text` lines, so the same chunker works on it. 
 
### 35.2 Cleaning 
 
**Where:** `medrag/documents/prepare.py` → `clean()`, `check()`, `_norm()`. 
 
```python 
HEADING_CSV = re.compile(r"\s*([A-Z][A-Z0-9 /&(),'-]{2,60}):,\s*") 
d["text"] = strip_hidden_unicode(d["text"]) # zero-width, bidi, tag chars 
d["text"] = d["text"].map(lambda t: HEADING_CSV.sub(lambda m: f"\n{m.group(1)}: ", t)) # "PLAN:, x" -> "\nPLAN: x" 
d["text"] = d["text"].str.replace(r"\bX{2,}\b", "[redacted]", regex=True) # Open-i "XXXX" 
d = d[d["text"].str.split().str.len() >= 15] # empty / very short 
 
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
g = "male" if m > f else "female" if f > m else None # majority of gendered words 
cand = np.ones(len(p), bool) if g is None else gender == g 
if (a := AGE_RE.search(t)) and 0 < int(a.group(1)) < 100: # "45-year-old" 
 age = (np.datetime64(when) - dob).astype("timedelta64[D]").astype(int) / 365.25 
 near = cand & (np.abs(age - int(a.group(1))) <= 2) # within 2 years at the document date 
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
SECTION_RE = re.compile(r"^([A-Z][A-Z0-9 /&(),'-]{2,60}):", re.M) 
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
 for sec in split_sections(body): # level 1: sections 
 if _words(sec) <= max_words: 
 units.append(sec) 
 else: 
 units.extend(SENTENCE_RE.split(sec)) # level 2: sentences 
 chunks, cur = [], [] 
 for u in units: # greedy packing 
 if cur and _words(" ".join(cur + [u])) > max_words: 
 chunks.append("\n".join(cur)) 
 cur = cur[-overlap_sentences:] if overlap_sentences and _words(cur[-1]) < max_words // 3 else [] 
 cur.append(u) 
 if cur: 
 chunks.append("\n".join(cur)) 
 return chunks 
 
def chunk_document(header, body, max_words=220): 
 budget = max(60, max_words - _words(header)) # the header counts against the budget 
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
chunk_document(doc_header, text, 220) # Open-i also tagged modality=XR, body_part=chest 
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
- **Parent-child chunks:** most reports are already a single chunk. 
 
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
 index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT) # M = 32 links per node 
 index.hnsw.efConstruction = 80 
else: 
 index = faiss.IndexFlatIP(dim) # exact: brute force, 100% recall 
index.add(vecs) 
faiss.write_index(index, str(f["dense"])) 
 
# loading 
self.vectors = np.load(f["vectors"], mmap_mode="r") # memory-mapped, not read into RAM 
self.dense = faiss.read_index(str(f["dense"])) 
if hasattr(self.dense, "hnsw"): 
 self.dense.hnsw.efSearch = 128 
 
def dense_search(self, qvec, k, rows=None): 
 if rows is not None: # a metadata filter is active 
 if len(rows) == 0: 
 return [] 
 scores = np.asarray(self.vectors[rows]) @ qvec # exact search inside the subset 
 top = np.argsort(-scores)[:k] 
 return [(int(rows[i]), float(scores[i])) for i in top] 
 scores, ids = self.dense.search(qvec[None, :].astype(np.float32), k) # whole index: FAISS 
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
- **Fast enough:** at 121k × 384, brute force takes about 10-20 ms per query on CPU, and recall is 100%. 
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
 words = [inv[t] for t in toks.ids[0] if inv[t] in self._vocab] # drop words the corpus never saw 
 if not words: 
 return [] 
 scores = self.bm25.get_scores(words) # one score per chunk 
 if rows is not None: 
 sub = scores[rows] 
 top = np.argsort(-sub)[:k] 
 return [(int(rows[i]), float(sub[i])) for i in top if sub[i] > 0] 
 top = np.argpartition(-scores, min(k, len(scores) - 1))[:k] # O(n) top-k, then sort only k 
 top = top[np.argsort(-scores[top])] 
 return [(int(i), float(scores[i])) for i in top if scores[i] > 0] 
``` 
 
**Why BM25 alongside embeddings:** 
- **Exact tokens:** medical text is full of identifiers and precise terms (`MRN0001234`, `ACC000012345`, `LI-RADS 5`, `BI-RADS 4`, "cholelithiasis"). Embeddings blur these; BM25 matches them exactly. 
- **Measured:** BM25 alone (0.837) beat the default (0.808) on the 40 Open-i documents questions, while dense alone had 0.621 (`doc_semantic` 0.27). Dense alone was better on paraphrases. Hybrid covers both (section 45). 
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
todo = [i for i, cid in enumerate(chunks["chunk_id"]) if cid not in old] # embed only these 
for start in tqdm(range(0, len(todo), batch)): # batches of 256 
 idx = todo[start:start + batch] 
 vecs[idx] = embedder.embed_documents([texts[i] for i in idx]) 
 
# 2. write a new versioned build 
version = datetime.now().strftime("%Y%m%d%H%M%S%f") 
def _files(path, version): 
 sfx = f"-{version}" if version else "" # no version: an old unversioned index 
 return {k: path / f"{k}{sfx}{ext}" for k, ext in 
 (("chunks", ".parquet"), ("vectors", ".npy"), ("dense", ".faiss"), ("bm25", ""))} 
 
# 3. switch over atomically, then clean up 
tmp = path / "manifest.json.tmp" 
tmp.write_text(json.dumps(manifest, indent=2)) 
os.replace(tmp, path / "manifest.json") # atomic rename: readers see the old build or the new one 
_remove_old_builds(path, version) # OSError ignored: a file still open (Windows) is removed next build 
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
 time.sleep(a.interval) # default 30 s 
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
 hit = process.extractOne(cand, self.names, scorer=fuzz.ratio, score_cutoff=90) # typo-tolerant 
ids.extend(self.by_name[best[1]]) # a shared name returns ALL matching patients 
 
# short modality codes count only in capitals 
if syn in ("us", "mr", "mg", "pet") and not re.search(rf"\b{syn.upper()}\b", q): 
 continue 
 
# dates 
if m := re.search(r"\b(?:last|past|previous)\s+(\d+|a|one|two|three|six)\s+(day|week|month|year)s?\b", ql): ... 
elif m := re.search(r"\bbetween\s+(20\d\d)\s+and\s+(20\d\d)\b", ql): ... 
 
f.aggregate = bool(re.search(r"\b(how many|count|number of|total|percentage|proportion|...)\b", ql)) 
f.latest = bool(re.search(r"\b(latest|most recent|newest|last (study|scan|report|imaging|visit))\b", ql)) 
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
 "intracranial haemorrhage": ["brain bleed", "bleeding in the brain", ...], ...} # 32 findings 
REPORT_TERMS = {"disc herniation": "disc extrusion nerve root compression", 
 "gallstones": "cholelithiasis", ...} # how reports phrase it 
 
_TERMS = sorted(((syn, label) for label, syns in SYNONYMS.items() for syn in syns + [label]), 
 key=lambda x: -len(x[0])) # longest first 
 
def detect_findings(query): 
 for phrase, label in _TERMS: 
 for m in re.finditer(rf"(?<![a-z]){re.escape(phrase.lower())}(?![a-z])", ql): 
 if any(a < m.end() and m.start() < b for a, b in taken): 
 continue # no overlapping matches 
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
# Documents have no body part and a lay word often sets one ("back" -> spine), so body_part goes before "documents". 
FILTER_ORDER = ["abnormal", "body_part", "documents", "priority", "dates", "modality", "patient"] # first relaxed first 
 
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
 if not self._mask(f, {dropped}).all(): # only report filters the question set 
 notes.append(f"No records match every filter; relaxed the '{dropped}' filter.") 
``` 
 
**Why:** 
- **Pre-filtering:** filtering *before* search is the single largest gain measured. MRR is about 0.5 without filters and 0.9-1.0 with them, because a patient question searches that patient's 5-50 chunks instead of 121,335. 
- **Relaxation instead of returning nothing:** "Asha Rao's MRI" when she has no MRI still shows her other studies, with a note saying so. 
- **The order** keeps the most important constraint, the patient, until last. Body part is dropped before the documents filter: "orthopedic document about back and hip pain" sets `body_part=spine` from "back", no document has a body part, and the older order dropped *documents* first and searched spine X-rays instead (audit row 15, fixed). Dates and modality stay after documents: dropping them first was tried and broke "Summarise the X-ray report for patient MRN... dated ..." (the phrase "X-ray report" sets the documents filter), cutting `lookup_by_mrn` Hit@8 from 1.00 to 0.55. With the final order, documents questions rise from Hit@8 0.875 / MRR 0.809 to 0.925 / 0.859, and easy and hard questions are unchanged (run eval_20261028_2246). 
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
 h.score += 1.0 / (k + rank + 1) # depends only on rank, not on the raw score 
 h.sources.append(name) 
 return sorted(fused.values(), key=lambda h: -h.score) 
 
rankings["dense"] = self.index.dense_search(self.embedder.embed_query(q), n_cand, rows) 
rankings["bm25"] = self.index.sparse_search(q, n_cand, rows) 
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
 _models = {} # one copy per process, shared by all retrievers 
 def __init__(self, model): 
 if model not in self._models: 
 self._models[model] = CrossEncoder(model, max_length=512) 
 self.ce = self._models[model] 
 def score(self, query, texts): 
 return np.asarray(self.ce.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False)) 
 
pool = hits[: max(s.rerank_pool, k)] # 30 candidates 
scores = self.reranker.score(q, self.meta["text"].iloc[[h.row for h in pool]].tolist()) 
if self.rerank_blend == "fuse": 
 by_rerank = [(pool[i].row, float(scores[i])) for i in np.argsort(-scores)] 
 fused = rrf({"hybrid": [(h.row, h.score) for h in pool], "rerank": by_rerank}) 
else: # replace 
 hits = sorted(pool, key=lambda h: -h.score) # after setting h.score = reranker score 
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
 | MiniLM | 0.872-0.885 | 
 | **MedCPT** | **0.905-0.915** | 
 
 The general reranker made results *worse* and was slower. 
 
**Why `fuse`:** 
- **Safer:** the reranker was trained on a different domain (PubMed abstracts, not reports), so fusing keeps what the first stage got right. `replace` was slightly better on hard questions (0.915) but worse on documents (0.793 vs 0.808). 
- **Within the noise:** the top three configurations are inside each other's confidence intervals. 
 
**Other details:** 
- **Pool of 30:** a CPU latency trade-off of about 2-3 s per question. MiniLM takes 0.7 s if speed matters more. 
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
scope = meta[retriever._mask(f, set(FILTER_ORDER))] # ALL filters, never relaxed 
st = scope[(scope["doc_type"] == "study") & (scope["part"] == 0)] # part 0: each study counted once 
if f.findings: 
 st = st[retriever.finding_mask(f.findings).loc[st.index]] # radiologist's impression, not the AI label 
lines = [f"Filters applied: {f.describe()}", 
 f"Matching studies: {len(st):,} across {st['patient_id'].nunique():,} patients", 
 "By modality: ...", "By priority: ...", f"Studies with abnormal findings: {len(abnormal):,} (...%)"] 
if len(f.patient_ids) > 1: 
 lines.append(f"WARNING: {len(f.patient_ids)} different patients match this name. " 
 "Answer separately for each patient (give MRN):") 
 
def finding_mask(self, labels): 
 pattern = "|".join(f"(?:{impression_pattern(lbl)})" for lbl in labels) # e.g. "disc extrusion" 
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
 if facts: parts.append(f"COMPUTED FACTS:\n{facts}") 
 parts.append(f"SOURCES:\n{sources or '(none)'}") 
 return "\n\n".join(parts) 
 
if r.filters.unresolved_patient: # no LLM call at all 
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
 if attempt == retries or not any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "rate limit", "Rate limit", "ThrottlingException", "503", "UNAVAILABLE")): 
 raise 
 time.sleep(15 * (attempt + 1)) # 15, 30, 45, 60 s 
 
class OllamaLLM(LLM): 
 def stream(self, system, user, max_tokens=1500): 
 body = {"model": self.model, "stream": True, "think": False, "messages": [...], 
 "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": max_tokens}} 
 with httpx.stream("POST", f"{url}/api/chat", json=body, timeout=600) as r: 
 for line in r.iter_lines(): ... # also strips any <think>...</think> 
 
class BedrockLLM(LLM): 
 def stream(self, system, user, max_tokens=1500): 
 resp = self.client.converse_stream(modelId=self.model, system=[{"text": system}], 
 messages=[{"role": "user", "content": [{"text": user}]}], 
 inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1}) 
 for ev in resp["stream"]: 
 if text := ev.get("contentBlockDelta", {}).get("delta", {}).get("text"): 
 yield text 
 
def get_llm(spec=None): # "ollama:qwen2.5:3b" -> split on the FIRST colon only 
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
 p = rnd.choice(TEST_PHRASES[label]) # "water on the lung" 
 return p, any(lbl == label for _, lbl in detect_findings(p)) # in_dictionary flag 
 
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
 ranked = list(dict.fromkeys(ranked_docs))[:k] # chunks -> unique documents, order kept 
 rel = [1 if d in gold else 0 for d in ranked] 
 first = next((i for i, r in enumerate(rel) if r), None) 
 dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel)) 
 idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k))) 
 return {"hit": float(first is not None), "mrr": 1 / (first + 1) if first is not None else 0.0, 
 "recall": sum(rel) / len(gold), "ndcg": dcg / idcg if idcg else 0.0} 
 
if not any(g.startswith("patient:") for g in it["gold_docs"]): 
 ranked = ranked[ranked["via"] != "profile"] # don't let pinning inflate the score 
``` 
 
**Why each metric:** 
- **Hit@8:** does the LLM see the right evidence at all? This sets the ceiling for answer quality. 
- **MRR:** how high is the first right document? Small LLMs pay most attention to the first sources. 
- **Recall@8:** for cohort questions with 1-3 right documents, did we find all of them? 
- **nDCG@8:** rewards putting all right documents near the top. It is the standard ranking metric. 
- **Document level:** one report split into 3 chunks should count once, so chunks are de-duplicated to documents. 
 
**Configurations:** 
- **`Config`:** bundles a retriever, a mode and flags, so ablations run on exactly the same questions: no filters, no expansion, dense, BM25, hybrid, rerankers, fuse vs replace. 
 
--- 
 
## 53. Evaluation: answer checks 
 
**Method:** deterministic regex and substring checks matched to each question type, collapsed to one `auto_correct` pass/fail. 
**Where:** `medrag/eval/run.py` → `score_answer()`, `_fact_hit()`, `ABSTAIN_RE`, `NEGATIVE_RE`, `eval_generation()`. 
 
```python 
ABSTAIN_RE = re.compile(r"(no patient matching|not (be )?found|do(es)? not contain|no record|not in the records|no information|unable to find)", re.I) 
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
 return any(alt.strip().lower() in text for alt in fact.split("|")) # "ct|computed tomography" 
``` 
 
**Why:** 
- **Free, instant, reproducible:** running a judge model on every answer costs money and quota, and judges disagree with themselves between runs. 
- **Exact number regex:** accepts `1,234` or `1234` but not `11234` or `12.3`. Lookarounds stop partial matches. 
- **`abstain_correct` cuts both ways:** it penalises saying "not found" when the answer exists, and inventing when it does not. 
- **Alternatives with `|`:** CT or "computed tomography" both count. 
- **Errors are recorded, not fatal:** one provider failing doesn't lose the run. 
 
--- 
 
## 54. Evaluation: LLM judge 
 
**Method:** a separate, stronger model grades each answer on four 1-5 scales and returns JSON; agreement with the deterministic checks is reported. 
**Where:** `medrag/eval/run.py` → `JUDGE_PROMPT`, `JUDGE_KEYS`, `_judge()`, the agreement block in `report()`; `medrag/providers/llms.py` → `get_judge()`. 
 
```python 
JUDGE_KEYS = ["correctness", "faithfulness", "answer_relevance", "context_relevance"] 
prompt = JUDGE_PROMPT.format(q=it["question"], ref=it["answer"], ctx=ctx[:14000], ans=ans.text) 
raw = judge.complete("You are a strict, fair evaluator. Output JSON only.", prompt, 600) 
j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0)) # tolerate text around the JSON 
 
agree = ((both["judge_correctness"] >= 4) == (both["auto_correct"] == 1)).mean() # 88% in the judged run 
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
- **Bootstrap intervals:** they show when a difference is noise. On 40 document questions, 0.808 vs 0.797 is inside the intervals, so the report says "equivalent" instead of declaring a winner. 
- **Why bootstrap:** it works for any metric (binary hit, MRR, 1-5 scores) without assuming a normal distribution. 
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
def load_retriever(path: str, built_at: str) -> Retriever: # built_at in the key: a new build reloads 
 return Retriever(Index(Path(path))) 
 
rag = RAG(retriever, get_llm(llm_spec)) 
r, facts, stream = rag.stream(question, mode) 
st.caption(f"Filters: {r.filters.describe()}" + (f" · {' '.join(r.notes)}" if r.notes else "")) 
text = st.write_stream(stream) # token-by-token output 
``` 
 
**Why:** 
- **Streamlit:** a full data app in one Python file, with no frontend build. It is right for an internal clinical tool and a demo. 
- **`cache_resource` keyed on `built_at`:** the 442 MB index and models load once per process and reload automatically after a rebuild. 
- **Transparency:** filters, expansion notes, sources and computed facts are visible, so a clinician can see *why* the answer says what it says. 
- **Settings in the sidebar:** index, LLM, mode and k are switchable live, which makes manual comparison easy. 
 
--- 
 
## 57. Tests 
 
**Method:** pytest with a session fixture that builds a small dataset and index using the hash embedder. 
**Where:** `tests/conftest.py`, `tests/test_pipeline.py` (18 tests). 
 
```python 
def test_count_is_zero_when_patient_lacks_the_modality(env, built): 
 # a patient with no MRI: the relaxed search still shows their studies, but the count must be 0 
 ... 
 
class FakeClient: # used by test_bedrock_streams_converse_text 
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
| 10. Prompt | `build_prompt` | Question + note + facts + [S1]-[S8] | 
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
 
--- 
 
# Part III: Practical and non-technical details 
 
Everything a reader needs besides the pipeline itself: who it is for and how to use it, the project timeline, the machine and software it ran on, how to set it up from scratch, the models, measured speed and memory, scaling, reproducibility, cost, risks and compliance, lessons learned, a glossary and an FAQ. 
 
## Part III contents 
 
60. [Who it is for and how to use it](#60-who-it-is-for-and-how-to-use-it) 
61. [Project timeline](#61-project-timeline) 
62. [Hardware and software](#62-hardware-and-software) 
63. [Setup from scratch](#63-setup-from-scratch) 
64. [Models used](#64-models-used) 
65. [Speed, memory and disk (measured)](#65-speed-memory-and-disk-measured) 
66. [Scaling: more data and more users](#66-scaling-more-data-and-more-users) 
67. [Reproducibility](#67-reproducibility) 
68. [Cost](#68-cost) 
69. [Risks, ethics, licences and compliance](#69-risks-ethics-licences-and-compliance) 
70. [Lessons learned](#70-lessons-learned) 
71. [Glossary](#71-glossary) 
72. [FAQ](#72-faq) 
 
--- 
 
## 60. Who it is for and how to use it 
 
**Users:** radiologists, referring clinicians, clinical coordinators and data or quality teams who need quick answers from imaging records without writing database queries. 
 
**What it is good at** 
 
| Kind of question | Example | Typical result | 
|---|---|---| 
| One study | "What did the CT chest on 2024-03-02 show for Aarini Hayre?" | Exact impression, with the report cited | 
| By MRN | "Summarise the MRI report for MRN0000011 dated 2023-07-11." | Same, found by MRN | 
| Latest | "What was the most recent imaging study for Aarini Hayre?" | Date, modality and impression of the newest study | 
| Patient history | "Summarise Aarini Hayre's imaging history." | Counts by modality, timeline, documents | 
| Has the patient ever had... | "Has Aarini Hayre ever had a kidney stone?" | Yes with the date, or no | 
| Lay words | "Did she have a slipped disc / brain bleed / fatty liver?" | Mapped to the medical finding automatically | 
| Misspelt names | "What did the X-ray show for Aarini Hayr?" | Still finds the patient | 
| Cohort counts | "How many stat MRI studies were done in 2023?" | Exact count from the database (see the limits below) | 
| Documents | "Summarise the uploaded discharge note for MRN0000011." | Answer from the transcription | 
 
**How to ask well** 
- Name the patient (full name or MRN) when the question is about one person. 
- Add the date, modality or body part if you know them ("CT chest", "in March 2024"); they narrow the search. 
- Ask one thing at a time. 
- Read the "Filters" line above the answer: it shows how the question was understood. 
- Check the cited sources [S1], [S2] in the Sources panel before acting on an answer. 
 
**What it does not do** 
- It does not diagnose or recommend treatment beyond what reports say. 
- It does not look at images, only report text. 
- It only knows what is in the indexed records; "not found" means not in the records, not "normal". 
- Counting patients by a finding works when the wording is known ("kidney stone", "pneumonia"). Unusual wording ("stones in the gallbladder") is not recognised yet, and the count then covers all studies (section 20.7). 
 
--- 
 
## 61. Project timeline 
 
| Date and time | Milestone | 
|---|---| 
| 2026-10-27 | Old versions archived; project rebuilt as `medrag` | 
| 2026-10-27 14:15-18:21 | First pipeline runs: synthetic data, cleaning, hash and bge indexes, first answers | 
| 2026-10-27 19:42 | First commit: rebuilt medical RAG pipeline | 
| 2026-10-27 19:58-20:33 | Harder test set, Open-i documents, reranker comparison (MedCPT chosen), ablations | 
| 2026-10-27 21:19 | Commit: outside documents, query expansion, MedCPT, harder evaluation | 
| 2026-10-27 21:46-2026-10-28 00:06 | First judged answer runs (Gemini judge; quota problems solved) | 
| 2026-10-28 00:00 | Code audit; findings 1-6 fixed with tests | 
| 2026-10-28 00:08-01:06 | MTSamples downloaded, cleaned, linked and indexed (50 min) | 
| 2026-10-28 07:49-08:03 | Documents evaluation with MTSamples; results committed | 
| 2026-10-28 10:35 | AWS Bedrock provider added | 
| 2026-10-28 10:44-14:46 | Five local LLMs compared | 
| 2026-10-28 14:59-17:07 | This report written and extended | 
| 2026-10-28 23:30 | Hidden-Unicode cleaning added to reports and documents (vendored from watermarks-remover) | 
 
About 27 hours from rebuild to the current state, most of it waiting for CPU-bound embedding and LLM runs. 
 
--- 
 
## 62. Hardware and software 
 
**Machine (all numbers in this report were measured here)** 
 
| Item | Value | 
|---|---| 
| Processor | Intel Core i7-8665U, 1.90 GHz, 4 cores / 8 threads (laptop) | 
| Memory | 15.8 GB RAM | 
| Graphics | Intel UHD Graphics 620 (integrated; not used, no CUDA GPU) | 
| Operating system | Windows 11 Pro | 
 
**Software versions** 
 
| Package | Version | Used for | 
|---|---|---| 
| Python | 3.14.3 | everything | 
| Ollama | 0.40.11 | local LLMs | 
| pandas | 3.1.0 | data processing | 
| numpy | 2.5.4 | vectors, maths | 
| pyarrow | 26.0.0 | parquet files | 
| faiss-cpu | 1.15.1 | vector index | 
| bm25s | 0.3.15 | keyword index | 
| PyStemmer | 3.1.0 | stemming for BM25 | 
| sentence-transformers | 6.2.0 | embeddings and reranker | 
| transformers | 5.21.0 | model loading | 
| torch | 2.14.1 | model inference on CPU | 
| rapidfuzz | 3.14.6 | fuzzy name matching | 
| Faker | 40.43.0 | synthetic names | 
| streamlit | 1.66.1 | user interface | 
| pydantic-settings | 2.16.1 | configuration | 
| httpx | 0.28.1 | Ollama calls | 
| psycopg | 3.3.6 | PostgreSQL | 
| defusedxml | 0.7.1 | safe XML parsing | 
| anthropic | 2.3.0 | Claude provider | 
| openai | 3.39.0 | OpenAI, Groq, OpenRouter providers | 
| google-genai | 2.33.0 | Gemini provider and judge | 
| boto3 | 1.43.120 | AWS Bedrock provider | 
| pytest | 9.1.1 | tests | 
 
`requirements.txt` lists the packages without versions; this table records the versions that produced the results (see section 67). 
 
--- 
 
## 63. Setup from scratch 
 
```bash 
# 1. get the code 
git clone https://github.com/nikhilhv414/medrag.git 
cd medrag 
 
# 2. Python environment 
python -m venv .venv 
.venv/Scripts/activate # Windows (Linux/macOS: source .venv/bin/activate) 
pip install -r requirements.txt 
 
# 3. settings 
cp .env.example .env # then set DATA_BACKEND=parquet for a no-database setup, and any API keys you use 
 
# 4. local LLM (install Ollama from ollama.com first) 
ollama pull qwen2.5:3b # or qwen3:1.7b, phi4-mini, llama3.2:3b 
 
# 5. data + index (about 3 hours on a laptop CPU for the first embedding) 
python -m medrag pipeline 
 
# 6. optional real documents 
# Open-i: download NLMCXR_reports.tgz into data/source/openi/ 
# MTSamples: put mtsamples.csv into data/documents/ 
python -m medrag documents 
python -m medrag index # only the new chunks are embedded 
 
# 7. use it 
streamlit run app.py 
python -m medrag ask "What was the most recent imaging study for <name>?" 
 
# 8. check it 
pytest 
python -m medrag eval --max-answers 16 
``` 
 
The embedding model (bge-small, 129 MB) and reranker (MedCPT, 837 MB) download automatically from Hugging Face on first use and are cached in `~/.cache/huggingface/`. 
 
--- 
 
## 64. Models used 
 
| Role | Model | Size on disk | Runs on | Where it comes from | 
|---|---|---:|---|---| 
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d, ~33M parameters) | 129 MB | CPU, in process | Hugging Face | 
| Reranker | `ncbi/MedCPT-Cross-Encoder` (BERT-base size) | 837 MB cache | CPU, in process | Hugging Face | 
| Reranker (tested) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | small | CPU | Hugging Face | 
| Reranker (tested, rejected) | `BAAI/bge-reranker-base` | | CPU | Hugging Face | 
| Answers (default) | `qwen2.5:3b` | 1.9 GB | CPU, Ollama | Ollama library | 
| Answers (compared) | `qwen3:1.7b` | 1.4 GB | CPU, Ollama | Ollama library | 
| Answers (compared) | `qwen3:4b` | 2.5 GB | CPU, Ollama | Ollama library | 
| Answers (compared) | `phi4-mini` (3.8B) | 2.5 GB | CPU, Ollama | Ollama library | 
| Answers (compared) | `llama3.2:3b` | 2.0 GB | CPU, Ollama | Ollama library | 
| Judge, and answers (compared) | `gemini-3-flash-preview` | cloud | Google AI Studio free tier | API | 
| Answers (compared) | `gemini-3.1-flash-lite-preview` | cloud | Google AI Studio free tier | API | 
| Answers (production option) | e.g. `us.amazon.nova-lite-v1:0` | cloud | AWS Bedrock | API, not yet tested live | 
 
Each model has its own licence. Check it before any commercial use. 
 
--- 
 
## 65. Speed, memory and disk (measured) 
 
**Start-up and search** (one Python process, this laptop, current index): 
 
| Step | Time | Process memory | 
|---|---:|---:| 
| Load index (121,335 chunks) + embedding model | 14.1 s | 1,138 MB | 
| First search (loads MedCPT on first use) | 10.3 s | 1,589 MB | 
| Next searches | 0.6 s for a simple question | about 1.6 GB | 
| Average search in the evaluation | 1.7 s easy · 2.6 s hard · 8 s documents | | 
 
Document questions are slowest because the document filter keeps a large set of rows (all documents and profiles) and document chunks are longer for the reranker to read. 
 
**Answers** (Ollama on CPU, per answer): qwen3:1.7b 44 s · qwen2.5:3b 55 s · phi4-mini 56 s · llama3.2:3b 57 s · qwen3:4b about 450 s. Ollama needs about 2-4 GB more RAM for a 3-4B model. 
 
**Disk** 
 
| Item | Size | 
|---|---:| 
| Source files (Open-i archive + XML, MTSamples) | 1 MB + 32 MB + 17 MB | 
| Raw stage (`data/raw`) | 19 MB | 
| Clean stage (`data/clean`) | 14 MB | 
| Index (bge-small): vectors 186 MB, FAISS 186 MB, BM25 64 MB, chunks 24 MB | 442 MB | 
| Hugging Face model cache (bge-small + MedCPT) | about 1 GB | 
| Ollama models (5 installed) | 10.3 GB | 
 
--- 
 
## 66. Scaling: more data and more users 
 
| If this grows... | What happens | What to change | 
|---|---|---| 
| Chunks above 150,000 | Index switches to approximate HNSW automatically | Nothing; HNSW keeps search in milliseconds | 
| Chunks in the millions | Vectors (1.5 KB each) and metadata no longer fit comfortably in RAM; pandas filtering slows | Move metadata filtering into PostgreSQL, vectors into pgvector or a vector database | 
| New reports every day | `watch` re-cleans and re-indexes incrementally | Run `watch` as a service; only new chunks are embedded | 
| Several users at once | One CPU process answers one question at a time; local LLM answers take about a minute each | Use a hosted LLM (Bedrock) for answers; run the app behind a web server with several workers sharing the memory-mapped index | 
| Faster search | MedCPT is most of the search time on CPU | Use MiniLM (0.7 s) or a small GPU for the reranker | 
| Embedding a new large collection | About 3.6 chunks per second on this CPU | Use a GPU or a cloud embedding API for the first build; incremental after that | 
 
--- 
 
## 67. Reproducibility 
 
**Fixed seeds** 
 
| What | Seed | 
|---|---:| 
| Synthetic data generation | 7 | 
| Document-to-patient linking | 13 | 
| Test set | 11 (hard +1, documents +2) | 
| Human review sheet sample | 5 | 
| Bootstrap intervals | 0 | 
| Generation "now" date | 2026-10-01 | 
 
With the same code, data files, seeds and package versions, the data, chunks, test questions and retrieval metrics come out the same. LLM answers can still vary slightly, even at temperature 0.1. 
 
**Gaps to know about** 
- `requirements.txt` has no version pins. Section 62 records the versions used; pinning them (`pip freeze > requirements.lock`) is recommended. 
- `reports/` is git-ignored, so the evaluation folders this report cites (`reports/eval_*`) exist on the development machine only. The key tables are copied into this report and the README. 
- The data, indexes and source files are git-ignored too; they are rebuilt with the commands in section 63 (Open-i and MTSamples must be downloaded again). 
 
--- 
 
## 68. Cost 
 
| Item | Cost | 
|---|---| 
| Software and models | Free and open source (local) | 
| Data | Synthetic (free); Open-i and MTSamples are free public downloads | 
| Compute | The developer's laptop; no cloud compute used | 
| LLM judge | Google Gemini free tier (limited by daily quotas per model); NVIDIA API catalog free credits for the D comparison (20.11) | 
| AWS Bedrock | Not used yet. It is pay-per-token. A typical question is about 1,100 input tokens and 100-300 output tokens, so cost per question is small for a "lite" model; check current Bedrock prices before planning | 
 
--- 
 
## 69. Risks, ethics, licences and compliance 
 
**Data licences** 
- **Open-i** reports are licensed CC BY-NC-ND 4.0: attribution, **non-commercial use only**, no derivatives. Using them in a commercial product is not allowed; for research and demos they must be credited to the US National Library of Medicine / Indiana University. 
- **MTSamples**: the original website's terms are unclear for redistribution; this project downloads them for local use only and does not commit them to git. 
- The synthetic data is generated by the project and has no licence restrictions. 
 
**Code licences** 
- `medrag/data/text_unicode.py` is copied from watermarks-remover (MIT, Copyright (c) 2026 Guillaume Meyer and contributors). The MIT licence allows reuse as long as the copyright notice stays; it is kept in the file header. 
 
**Privacy and compliance** 
- No real patient data is used. All patients, MRNs and dates are synthetic; Open-i and MTSamples are public, de-identified texts. 
- Linking real documents to synthetic patients is artificial and only for testing. It must never be presented as a real patient's record. 
- Using real patient data requires a HIPAA Business Associate Agreement (or the local equivalent, such as India's DPDP Act or the GDPR) with every service that sees the data, access control, audit logging and data-retention rules. None of these are built yet. 
- Local mode (Ollama) keeps all data on the machine; cloud providers receive the question and retrieved record text. 
 
**Clinical safety** 
- This is decision support, not a medical device. It is not validated for clinical use. 
- Answers can be wrong: the best local models failed 3 of 18 test questions. Every answer must be checked against its cited sources. 
- "Not found" means not in the indexed records, not that the patient is healthy. 
- The AI predictions in the data are deliberately imperfect (86% correct) and are shown as such. 
 
**Bias and coverage** 
- The synthetic data follows simple, uniform rules (for example, disease rises evenly with age); real populations differ. 
- Text is English only; names come from Indian, US and UK name lists. 
- Results on synthetic data overstate what to expect on real, messier reports. 
 
**Security** 
- API keys live only in the git-ignored `.env`; AWS uses its own credential chain. 
- Downloaded archives and XML are opened with safe settings. 
- The Streamlit app has no login; it must not be exposed beyond a trusted machine or network as it stands. 
 
--- 
 
## 70. Lessons learned 
 
1. **Filtering on metadata beat every model choice.** Narrowing to the right patient, date and modality doubled retrieval accuracy; no embedding or reranker change came close. 
2. **Do not let an LLM count.** It only sees 8 sources. Computing counts from the database and handing them over is exact and cheap. 
3. **Domain matters for rerankers.** A general web-trained reranker made results worse; a medical one helped. 
4. **Hybrid search is insurance.** BM25 won on documents, dense search on paraphrases; together they were never far from the best. 
5. **Build the test set from the data.** Known answers at scale made every change measurable in minutes, and the traps (shared names, typos, unknown patients) found real bugs. 
6. **Report uncertainty.** With 16-18 answers per model, one question moves a score by 12-25 points; bootstrap intervals kept conclusions honest. 
7. **Check the judge against exact checks.** 88% agreement gave confidence in both, and the disagreements showed where each was weak. 
8. **Small CPU models are usable but slow.** 40-60 seconds per answer is fine for a demo, not for clinicians; a hosted model is the path to real-time. 
9. **Incremental, versioned indexing pays off quickly.** Adding a new collection took 50 minutes instead of 3 hours, and the running app was never broken by a rebuild. 
10. **Audits find what tests miss.** The whole-repository audit found six real bugs (wrong counts, "tell us" read as ultrasound, dropped patients) that the tests at the time passed. 
11. **When every model fails the same question, suspect the test or the pipeline.** Adding Gemini showed that the "cohort count" failure shared by all local models was a bug in the expected answer, not in the models. 
 
--- 
 
## 71. Glossary 
 
**Clinical terms** 
 
| Term | Meaning | 
|---|---| 
| MRN | Medical Record Number: the hospital's unique id for a patient | 
| Accession number | Unique id for one imaging study | 
| Modality | Type of imaging: CT, MRI, XR (X-ray), US (ultrasound), MG (mammography), PET (PET-CT) | 
| Body part | Area imaged: chest, abdomen, head, brain, spine, knee, hand, thyroid, pelvis, breast, whole body | 
| Indication | Why the scan was ordered ("cough and fever") | 
| Findings | What the radiologist saw, in detail | 
| Impression | The radiologist's conclusion: the most important line of a report | 
| Recommendation | Suggested next step in the report | 
| Priority | Urgency of the study: routine, urgent, stat (immediate) | 
| Report status | final (signed), preliminary (draft), amended (corrected) | 
| Follow-up study | A later scan of the same area, compared with the earlier one ("interval improvement", "unchanged", "progression") | 
| BI-RADS, LI-RADS, TI-RADS | Standard risk scores for breast, liver and thyroid findings | 
| De-identified | Personal details removed (Open-i shows them as XXXX) | 
| Transcription | A dictated clinical note typed up (MTSamples) | 
| HIPAA / BAA | US health-privacy law / the contract a service provider must sign to handle patient data | 
 
**Technical terms** 
 
| Term | Meaning | 
|---|---| 
| RAG | Retrieval-augmented generation: find relevant records first, then let an LLM answer from them | 
| LLM | Large language model: the model that writes the answer | 
| Chunk | A piece of a document small enough to search and give to the LLM | 
| Embedding / vector | A list of numbers (384 here) representing the meaning of a text | 
| Cosine similarity | How close two embeddings are in meaning (1 = same) | 
| Dense search | Search by embedding similarity (meaning) | 
| BM25 | Classic keyword search that scores word matches, favouring rare words | 
| Hybrid search | Dense + BM25 together | 
| RRF | Reciprocal Rank Fusion: merges ranked lists using only positions | 
| Reranker / cross-encoder | A model that reads the question and a chunk together to re-score the top candidates | 
| FAISS | Facebook AI Similarity Search: the vector index library | 
| HNSW | A fast approximate nearest-neighbour graph index | 
| Metadata filter | Restricting search by fields like patient, date or modality | 
| Query expansion | Adding medical synonyms to the question ("slipped disc" → "disc herniation") | 
| Token | A word piece; models count input and output in tokens | 
| Context window | How many tokens a model can read at once (8,192 here for Ollama) | 
| Hallucination | An answer the model made up, not supported by the sources | 
| Grounding / citation | Tying each claim to a source [S1], [S2] | 
| Hit@k | Was a right document in the top k? | 
| MRR | Mean Reciprocal Rank: 1 / position of the first right document, averaged | 
| Recall@k | Share of all right documents found in the top k | 
| nDCG@k | Ranking quality that rewards right documents near the top | 
| Bootstrap 95% CI | A range likely to contain the true score, from resampling the results 1,000 times | 
| LLM judge | A second LLM that grades answers | 
| Ablation | Removing one component to measure what it contributes | 
| Parquet | A compressed column-based file format for tables | 
| Ollama | A tool to run open LLMs locally | 
| AWS Bedrock | Amazon's hosted service for running LLMs by API | 
 
--- 
 
## 72. FAQ 
 
**Is any of this real patient data?** 
No. Patients, studies and reports are synthetic. Open-i and MTSamples are real but public and de-identified texts, and their link to patients is synthetic. 
 
**Can it be used on our hospital's real data?** 
Technically yes: point `DATA_BACKEND=postgres` at the data, or use `import-legacy` as a template for mapping your tables. Before that you need the compliance steps in section 69, a login, and a validation on real reports. 
 
**Why not just ask ChatGPT or another chatbot?** 
A general chatbot does not know your patients' records, cannot count across 90,000 studies, and would need the data sent to it. This system finds the right records first, counts exactly from the database, cites sources, and can run fully on your own machine. 
 
**Why are answers slow?** 
The LLM runs on a laptop CPU (40-60 s per answer). Search itself takes 1-8 s. A hosted model (AWS Bedrock) or a GPU makes answers take a few seconds. 
 
**How accurate is it?** 
Finding the right record: 100% of easy questions, 99% of hard ones and 88% of document questions have it in the top 8. Answers: the best models, local qwen3 and cloud Gemini 3 Flash, pass 89% (16 of 18) of the exact checks. The known weaknesses are counting by unusual finding wording, and document questions that mention a body part. See sections 20.6-20.7. 
 
**What does "The records do not contain this information" mean?** 
Nothing in the indexed records answers the question. It does not mean the result is normal. 
 
**How do I add new reports or documents?** 
New database rows: run `python -m medrag watch` (or `clean` + `index`). New documents: put the files in `data/documents/` or `data/source/openi/`, then run `documents` and `index`. Only new chunks are embedded. 
 
**How do I switch the answer model?** 
Set `LLM_PROVIDER` and `LLM_MODEL` in `.env` (for example `ollama` and `qwen3:1.7b`), or type `provider:model` in the app's sidebar. 
 
**Can it read images?** 
No. It reads report text only. 
 
**Where are the detailed results?** 
In `reports/eval_*/summary.md` on the development machine, and summarised in sections 20 and 65 of this report and in the README.