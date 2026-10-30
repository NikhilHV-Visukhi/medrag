# MedRAG: end-to-end report

Retrieval-augmented question answering over hospital imaging records, from raw data to evaluated answers and production controls. This file is the single, complete account in pipeline order: data → cleaning → chunking → embedding and indexing → retrieval → answering → evaluation (methods, metrics, every result) → production → limits. `PROJECT_REPORT.md` holds the same work with more code-level detail (Part II) and practical notes (Part III); section numbers in brackets (§20.10) point there.

Repository: https://github.com/NikhilHV-Visukhi/medrag (branch `main`). Report date: 2026-10-30.

---

## Contents

1. [Summary](#1-summary)
2. [Goal and requirements](#2-goal-and-requirements)
3. [System architecture](#3-system-architecture)
4. [Data](#4-data)
5. [Data quality checks and cleaning](#5-data-quality-checks-and-cleaning)
6. [Outside documents: Open-i and MTSamples](#6-outside-documents-open-i-and-mtsamples)
7. [Chunking](#7-chunking)
8. [Embeddings and index](#8-embeddings-and-index)
9. [Query understanding and retrieval](#9-query-understanding-and-retrieval)
10. [Answer generation](#10-answer-generation)
11. [Evaluation method](#11-evaluation-method)
12. [Evaluation metrics](#12-evaluation-metrics)
13. [Results: retrieval](#13-results-retrieval)
14. [Results: A, embedding models](#14-results-a-embedding-models)
15. [Results: B, chunking and answer context](#15-results-b-chunking-and-answer-context)
16. [Results: C, retrieval tuning and ablations](#16-results-c-retrieval-tuning-and-ablations)
17. [Results: D, answer generation (8 models)](#17-results-d-answer-generation-8-models)
18. [E: production controls and load test](#18-e-production-controls-and-load-test)
19. [Final configuration and the decisions behind it](#19-final-configuration-and-the-decisions-behind-it)
20. [Engineering quality: tests, audit, problems solved](#20-engineering-quality-tests-audit-problems-solved)
21. [Hardware, models, speed and cost](#21-hardware-models-speed-and-cost)
22. [Limits, risks and next steps](#22-limits-risks-and-next-steps)
23. [How to reproduce](#23-how-to-reproduce)

---

## 1. Summary

**What it does.** A clinician asks a question in plain English ("What did Asha Rao's last CT show?", "How many stat MRIs were done in 2023?", "Has this patient ever had a slipped disc?"). MedRAG finds the relevant records, computes exact counts from the database, and has an LLM write a short answer that cites its sources, or says "The records do not contain this information" instead of guessing.

**Built on.** 20,000 synthetic patients with 89,868 imaging studies (generated with realistic errors, then cleaned), plus 6,211 real public de-identified documents (3,878 Open-i chest X-ray reports, 2,333 MTSamples transcriptions): **121,335 searchable chunks**.

**Key results**

| Area | Result |
|---|---|
| Data cleaning | 342,094 raw records → 330,378 clean; error-level quality checks failing: **9 → 0** |
| Retrieval, default (258 questions with a gold record) | **MRR 0.952, Recall@8 0.984, nDCG@8 0.960** (§20.10, `dense_weight=0.5`) |
| Retrieval by group | easy MRR 1.000 · hard 0.924 · documents 0.921 |
| Biggest retrieval factor | metadata filtering: MRR about 0.5 without filters → 0.9-1.0 with |
| A. Embedding models (6 compared) | none clearly beats bge-small (best +0.013 MRR, inside the intervals, at 2x vector size) → **kept bge-small** |
| B. Chunking (6 strategies) | MRR within 0.004 of each other → **kept section-220** |
| B. Answer context | whole record ("parent") **93%** vs chunk 83% correct on 60 document questions (+0.10 [0.03-0.18]) → **parent is the default** |
| C. Retrieval tuning (16 variants) | `dense_weight=0.5` best (+0.012 MRR); confirmed on 240 fresh questions (+0.016 [0.01-0.03]) → **adopted** |
| D. Answers, 8 models × 54 questions, LLM judge | Gemini 3 Flash preview **96%** correct, Gemini 3.5 Flash 94%, NVIDIA Nemotron 3.5 Super 94%†, Gemini Flash-Lite 93%, Nemotron Lightning 89%, qwen3:1.7b 78%, qwen2.5:3b 67% |
| Answer speed | NVIDIA 0.9-1.2 s median · Gemini 1.3-4.7 s · local 3B models on a laptop GPU about 4 s (43-57 s on CPU) |
| E. Production controls | sign-in + 4 roles, patient-level access enforced before the LLM, audit log, PHI pseudonymisation for cloud LLMs, prompt-injection guard, metrics, caching, load test, quality gate |
| Tests | **29 pass** |
| Cost | **0**: free and open-source models, free API tiers, no paid service |

† Nemotron 3.5 Super is also the judge; its independent keyword-check score is 91%.

**Status.** The pipeline works end to end and every design choice is measured. It is **not ready for real patient data**: that needs validation on governed real data with clinicians, a security review, and signed agreements with any cloud provider (§22).

---

## 2. Goal and requirements

A question-answering assistant over a hospital's radiology records. Requirements set during the project:

1. Answers come from the records only, cite their sources, and say "not found" instead of guessing.
2. Counts are exact: computed from the database, not from whatever the search returned.
3. Runs fully on a laptop with local models; switches to cloud models (Anthropic, OpenAI, Gemini, Groq, OpenRouter, NVIDIA, AWS Bedrock) by configuration only.
4. Every design choice is measured on a reproducible evaluation with known right answers.

The improvement round of 2026-10-28 to 2026-10-30 added five work items, all done:

| Item | Asked for | Where |
|---|---|---|
| A | embedding model benchmark | §14 |
| B | chunking comparison (and answer context) | §15 |
| C | retrieval tuning with ablations | §16 |
| D | generation comparison, separate answer metrics, LLM judge, continuous evaluation | §17 |
| E | production controls | §18 |

---

## 3. System architecture

```
 DATA                         PREPARE                                SEARCH + ANSWER
 ----                         -------                                ---------------
 synthetic generator ─┐
 legacy oads_db       ├─> raw stage ─> quality check ─> clean ─> clean stage ─┐
                      ┘               (report)         (log)                  │
 Open-i XML reports ──┐                                                       ├─> chunks (+ context header)
 MTSamples CSV ───────┴─> load ─> check ─> clean ─> link to patient ──────────┘          │
                                                                                         v
                                                       embed (bge-small) + BM25 ─> versioned index
                                                                                         │
 question ─> access mask ─> parse filters ─> filter rows ─> dense + BM25 ─> RRF ─> MedCPT rerank ─> pin profile/latest
                                                                                         │
                                          exact computed facts (counts, timeline) ───────┤
                                                                                         v
                                         LLM (local or cloud, PHI pseudonymised) ─> answer with [S1] citations
                                                                                         │
                                                              audit log + metrics ───────┘
```

| Layer | Code | Method |
|---|---|---|
| Settings | `medrag/config.py` | all settings from `.env` (pydantic-settings) |
| Data | `medrag/data/` | generator, 28 quality checks, logged cleaning rules, parquet or PostgreSQL storage |
| Documents | `medrag/documents/` | Open-i XML and MTSamples CSV loaders, cleaning, linking to patients |
| Chunking | `medrag/ingest/` | section-aware chunks with a contextual header |
| Embeddings / LLMs | `medrag/providers/` | one interface per role, many providers |
| Index | `medrag/index/store.py` | FAISS + BM25 + metadata, incremental and versioned |
| Retrieval | `medrag/retrieval/` | query parser, medical expansion, filters with relaxation, hybrid search, RRF, reranker, pinning, optional rewrite / multi-query / HyDE |
| Answering | `medrag/rag/answer.py` | computed facts, prompt, parent context, streaming |
| Security | `medrag/security/` | users and roles, access mask, audit and metrics logs, pseudonymiser, injection guard |
| Evaluation | `medrag/eval/` | test-set builder, retrieval metrics, answer checks, citation verification, LLM judge, bootstrap intervals, quality gate, benchmarks A-C |
| Interfaces | `medrag/__main__.py`, `app.py` | command line (`generate`, `check`, `clean`, `documents`, `index`, `ask`, `watch`, `eval`, `bench`, `loadtest`, `metrics`, `user`, `pipeline`) and a Streamlit app (Chat, Patients, Data quality, Evaluation, Audit tabs) |

---

## 4. Data

No real hospital data was available, and real data would need a HIPAA Business Associate Agreement, so the main data is synthetic. Real text comes from two public de-identified collections (§6).

### 4.1 Synthetic radiology data

`python -m medrag generate`: 20,000 patients, mean 5 studies per patient, 3% dirty rate, seed 7, "now" fixed at 2026-10-01.

| Part | How it is generated |
|---|---|
| Patients | names from Faker pools in 3 locales (en_IN, en_US, en_GB), about 400 first and 400 last names each, combined at random, which deliberately produces **shared names**; MRN, gender 49/51%, age 18-92, blood group, city |
| Chronic conditions | Poisson count rising with age (mean 0.4 + age/40, max 6) from 24 conditions, each with a diagnosis date |
| Studies | per-patient count from a gamma distribution (most have a few, some dozens) over 6 years; modality weights XR 35, CT 25, US 18, MRI 15, MG 4, PET 3; body part weighted per modality; no mammography or pelvic ultrasound for men |
| Findings | 32 findings, each allowed only on plausible body parts and modalities (pulmonary embolism only on CT, ACL tear only on MRI); chance of an abnormal study 45% + age/400 |
| Follow-ups | 70% of later studies of the same modality and body part after an abnormal one are follow-ups: "Unchanged" 50%, "Interval improvement" 30%, "Interval progression" 20%; this makes real histories and the trend questions possible |
| Reports | five sections (TECHNIQUE, COMPARISON, FINDINGS, IMPRESSION, RECOMMENDATION) from templates; 88% final, 7% preliminary, 5% amended; 60 radiologists |
| AI analysis | one prediction per study, **86% correct**, wrong ones with lower confidence (mean 0.62 vs 0.88), so counts must use the radiologist's impression, not the AI label |

### 4.2 Deliberately dirty records

Real exports are messy, so about 3% of records get realistic errors:

| Problem | Records |
|---|---:|
| Duplicate patient registrations (same MRN, new id) | +311 patients |
| Test / dummy patients | +5 |
| Names with wrong case or extra spaces | 656 |
| Non-standard gender values | 694 |
| Missing / future date of birth | 196 / 66 |
| Exact duplicate study rows | +1,416 |
| Orphan studies (patient does not exist) | 567 |
| Missing / future study date | 609 / 524 |
| Non-standard modality (misspellings + long names) | 29,846 |
| Non-standard priority | 2,826 |
| Duplicate reports for the same study | +1,407 |
| Empty report text | 858 |
| Reports wrapped in HTML | 2,687 |
| Reports with hidden Unicode (zero-width, bidi override) | 2,756 |
| Non-standard report status | 2,847 |
| AI confidence as a percentage or -1 / missing | 3,468 / 855 |

### 4.3 Data inventory (raw → clean)

| Table | Columns | Raw rows | Clean rows | Removed / merged |
|---|---:|---:|---:|---|
| patients | 7 | 20,316 | 20,000 | 5 test records, 311 duplicate registrations merged |
| conditions | 3 | 35,417 | 35,417 | re-pointed to merged patients |
| studies | 9 | 92,944 | 89,868 | 1,416 duplicates, 558 orphans, 1,102 missing or future dates |
| reports | 6 | 92,935 | 89,014 | 884 empty, 1,369 duplicates, the rest belonged to removed studies |
| ai_analysis | 5 | 91,528 | 89,868 | analyses of removed studies |
| documents | 10 | 8,954 | 6,211 | 158 empty or very short, 2,585 duplicate transcriptions merged |
| **Total** | **40** | **342,094** | **330,378** | |

Clean data at a glance: 20,000 patients (10,134 female, 9,587 male, 279 unknown); studies per patient mean 4.5, median 3, max 42; X-ray 33,375, CT 22,438, ultrasound 16,314, MRI 13,328, PET-CT 2,675, mammography 1,738; routine 61,956, urgent 17,821, stat 10,091; dates 2020-10-02 to 2026-09-29; 63.5% of reports have an abnormal finding; reports average 37 words.

---

## 5. Data quality checks and cleaning

### 5.1 Checks

`python -m medrag check --stage raw|clean` runs **28 checks** (error / warn / info) and writes a Markdown and JSON report; `--strict` exits with an error if an error-level check fails, so it can gate a pipeline. On the raw data, 9 error-level checks fail (test records, future birth dates, duplicate and orphan studies, missing and future study dates, empty reports, confidence outside 0-1). **After cleaning, all error-level checks pass.**

### 5.2 Cleaning rules and the actual run

`python -m medrag clean` applies these rules in order, logs each with a row count (`reports/cleaning_log.md`) and re-runs the checks.

| Table | Rule | Rows |
|---|---|---:|
| patients | remove test / dummy records | −5 |
| | normalise name case and spaces | 654 changed |
| | normalise gender (synonym map; anything else → unknown) | 694 changed |
| | clear impossible future date of birth | 66 cleared |
| | merge duplicate registrations by MRN (keep lowest id, fill gaps, remap foreign keys) | −311 |
| studies | drop exact duplicate rows and ids | −1,416 |
| | re-point to merged patients | 632 changed |
| | remove orphans | −558 |
| | normalise modality (`cat scan` → CT, `cxr` → XR, `usg` → US ...) | 29,221 changed |
| | normalise priority (`high` → urgent, `critical` → stat, empty → routine) | 2,761 changed |
| | remove missing or future dates | −1,102 |
| reports | strip HTML | 2,687 changed |
| | remove hidden Unicode (zero-width, bidi, tag characters; exotic spaces → spaces) | 2,756 changed |
| | remove empty reports | −884 |
| | keep only reports of surviving studies | −1,668 |
| | normalise status | 2,768 changed |
| | one report per study (final > amended > preliminary, then latest signed) | −1,369 |
| ai_analysis | keep only analyses of surviving studies | −1,660 |
| | convert percentage confidence to 0-1 | 2,602 changed |
| | clear confidence still outside 0-1 | 800 cleared |

854 studies end with no usable report; they are kept because their metadata still answers count questions. Hidden Unicode is removed because it is invisible to a reader but reaches BM25, the embedder and the LLM, where it can split words or hide instructions.

---

## 6. Outside documents: Open-i and MTSamples

| Source | What | Raw | Clean |
|---|---|---:|---:|
| Open-i (US NLM / Indiana University, CC BY-NC-ND 4.0) | chest X-ray reports, XML with MeSH labels | 3,955 | 3,878 |
| MTSamples | medical transcriptions, about 40 specialties | 4,999 | 2,333 |

**Loading safely:** the archive is extracted with Python's `data` filter (no path traversal), XML is parsed with `defusedxml`.

**Checks before → after cleaning:** empty text 61 → 0; under 15 words 99 → 0; duplicates 2,643 → 0; transcription artefacts (`HEADING:,`) 4,118 → 69; de-identification tokens `XXXX` 3,423 → 0; missing patient link 8,954 → 0.

**Cleaning:** rebuild section headings in transcriptions (4,118), mark `XXXX` as `[redacted]` (3,423), tidy whitespace, remove empty and very short (158), merge duplicate transcriptions filed under several specialties, keeping every specialty (2,585 merged).

**Linking to patients (synthetic):** documents have no patient, so each is attached to a plausible synthetic patient by gender (counted from pronouns and titles in the text), age ("45-year-old") and a random date: 1,058 by age and gender, 1,879 by gender only, 3,274 at random (mostly the de-identified Open-i reports). Seeded (13). This link exists only for testing.

---

## 7. Chunking

**Method** (`section-220`): split on upper-case section headings (FINDINGS:, IMPRESSION:, HISTORY OF PRESENT ILLNESS: ...); split over-long sections into sentences; pack into chunks of at most 220 words with a one-sentence overlap (only when that sentence is short).

**Contextual header on every chunk** so each chunk can be retrieved and understood on its own:

```
Patient: Aarini Hayre | MRN: MRN0000011 | Patient ID: 11 | male, age 51
Study: CT chest | Date: 2020-11-15 | Accession: ACC000000040 | Priority: urgent | Report status: preliminary
Indication: Follow-up of lung nodule
TECHNIQUE: ... FINDINGS: Moderate left pleural effusion ... IMPRESSION: Moderate left pleural effusion.
AI MODEL (DeepCT-Seg v4.2): predicted 'pleural effusion' (confidence 0.86).
```

| Chunk kind | Source | Chunks | Records |
|---|---|---:|---:|
| study | synthetic reports | 89,014 | 89,014 |
| patient profile (300-word budget: conditions, study history, documents, up to 40 recent impressions) | profiles | 19,923 | 19,897 patients |
| document | Open-i | 3,880 | 3,878 |
| document | MTSamples | 8,518 | 2,333 |
| **Total** | | **121,335** | |

Mean 88 words per chunk (study 79, profile 86, document 154). Each chunk id is a SHA-1 hash of its text, used for incremental re-indexing.

---

## 8. Embeddings and index

**Embeddings:** `BAAI/bge-small-en-v1.5`, 384 dimensions, L2-normalised, so the inner product is the cosine similarity. Asymmetric query/passage prefixes are added automatically for models trained with them (bge, e5, nomic). Batch size 64 on the 6 GB GPU.

**Index** (one folder per embedding model, 442 MB for bge-small):

| File | Content |
|---|---|
| `vectors-<v>.npy` | 121,335 × 384 float32, 186 MB, memory-mapped |
| `dense-<v>.faiss` | exact inner product (`IndexFlatIP`); above 150,000 chunks an HNSW graph (M 32, efConstruction 80, efSearch 128) |
| `bm25-<v>/` | BM25 (`bm25s`, k1 1.2, b 0.75, English stop words, Snowball stemmer), vocabulary 145,557 terms, 64 MB |
| `chunks-<v>.parquet` | chunk text and metadata, 24 MB |
| `manifest.json` | provider, model, dimension, counts, version |

**Incremental and safe:** vectors of unchanged chunks are reused (adding MTSamples embedded 10,749 new chunks and reused 110,586); each build writes versioned files and swaps `manifest.json` last with an atomic rename, so a running app never sees a half-written build. `python -m medrag watch` re-cleans and re-indexes when the raw data changes (file times and sizes, or PostgreSQL insert/update/delete counters).

---

## 9. Query understanding and retrieval

### 9.1 Parsing the question

| Filter | How it is found |
|---|---|
| Patient | MRN, accession, patient id, or a name (every 2-3 word window, exact first, then fuzzy with rapidfuzz ≥ 90 for typos); all patients sharing a name are returned |
| Unknown patient | a name or MRN that matches nobody → "No patient matching ... was found" without calling the LLM |
| Modality | synonym map; short codes (US, MR, MG, PET) only in capitals, so "tell us" is not ultrasound |
| Body part, priority, dates | synonyms (lung → chest), stat/urgent/routine, "last 3 months", "since 2023", "in March 2025" ... |
| Documents, latest, aggregate, abnormal | keyword families ("discharge summary", "most recent", "how many", "abnormal") |

**Medical query expansion:** 32 findings with lay synonyms ("slipped disc" → disc herniation, "fatty liver" → hepatic steatosis) plus the wording radiologists actually use (gallstones → "cholelithiasis"). Expanded terms go to BM25, dense search and the reranker, and a note goes to the LLM.

### 9.2 Retrieval pipeline (default `hybrid+rerank`)

1. **Access mask** (§18): only rows the user may see; never relaxed.
2. **Filter** rows by every parsed filter. If nothing matches, relax the least important filter first: abnormal, body part, documents, priority, dates, modality, patient. Each relaxation is reported.
3. **Expand** the query.
4. **Dense search**: top 60 by cosine similarity (exact search inside a filter).
5. **BM25 search**: top 60 over the same rows.
6. **Reciprocal Rank Fusion**: score = Σ w / (60 + rank), dense weight 0.5, BM25 weight 1.
7. **Rerank** the top 30 with the **MedCPT cross-encoder** (trained on PubMed search logs) and fuse its order with the hybrid order again by RRF, so the reranker cannot throw away what the first stage got right.
8. **Keep the top 8**, then **pin** the patient's profile chunk for a patient question and their 3 latest studies for a "latest" question (at least half the slots stay for search results).

Two measured examples on the full index: "Has Elaine Chanda ever had a slipped disc?" filters 121,335 → 15 rows and the computed facts report 0 matching studies (she never had one); "How many stat priority MRI studies were performed in 2023?" filters to 281 rows, and the facts give "281 studies across 273 patients".

---

## 10. Answer generation

**Computed facts** (exact, from the full metadata, not the 8 sources): number of matching studies and patients, date range, counts by modality and priority, abnormal count and share, top findings; for one patient, the latest study, documents and a timeline of up to 60 studies; for shared names, a warning and a summary per MRN. **Every filter is applied when counting**, even one the search relaxed, and findings are counted by the radiologist's impression, not the 14%-wrong AI label.

**Context given to the LLM:** by default each retrieved chunk is widened to its whole record (`ANSWER_CONTEXT=parent`, up to 900 words, duplicate lines removed); `chunk` gives the chunks alone (§15).

**Prompt rules:** use COMPUTED FACTS for counts and "latest"; otherwise only the SOURCES; never invent patients, dates, findings or numbers; cite inline as [S1], [S2]; if neither contains the answer, say "The records do not contain this information."; dates as YYYY-MM-DD; decision support only. Sources are fenced in `<source>` tags and declared to be record content, never instructions. Temperature 0.1.

**Providers** (one interface, `provider:model`): `ollama` (local, thinking switched off), `anthropic`, `openai`, `gemini`, `groq`, `openrouter`, `nvidia` (NVIDIA API catalog), `bedrock` (AWS Converse API). Rate limits and temporary overload (429, 503) are retried with back-off of 15, 30, 45, 60 s; a daily quota fails at once.

---

## 11. Evaluation method

### 11.1 Test set

Built from the indexed data itself (`medrag/eval/testset.py`, seed 11), so every question has a known right answer and, where it applies, a gold record. Only patients with a unique name are used for lookups, so the gold answer is unambiguous.

| Difficulty | Type | Example | Check |
|---|---|---|---|
| easy | `lookup_by_name` | What did the CT chest on 2024-03-02 show for Asha Rao? | impression keywords |
| easy | `lookup_by_mrn` | Summarise the MRI report for MRN0001234 dated 2023-07-11. | impression keywords |
| easy | `latest` | What was the most recent imaging study for X? | date + keywords |
| easy | `count_patient` | How many X-ray studies does X have? | exact number |
| easy | `ambiguous_name` | How many studies does <shared name> have? | every MRN named |
| easy | `semantic` | Which patient had a CT on 2024-03-02 showing pneumonia? | first name |
| easy | `count_cohort` | How many stat MRI studies were performed in 2023? | exact number |
| easy | `not_found` | Summarise the history of patient Zebulon Quixote. | says not found |
| hard | `hard_paraphrase_patient` | When did X have imaging that showed a slipped disc? | date + modality |
| hard | `hard_semantic_cohort` | Which patients had fluid around the lung on a CT in March 2024? | every surname |
| hard | `hard_trend` | On the follow-up of 2025-01-04, had X's fatty liver improved, worsened or stayed the same? | trend word |
| hard | `hard_count_finding` | How many patients had a kidney stone reported in 2023? | exact number |
| hard | `hard_negative` | Has X ever had a torn ACL? (never) | says no |
| hard | `hard_typo_name` | ... for "Asha Ro"? (one letter dropped) | impression keywords |
| documents | `doc_patient_lookup` | What did X's chest X-ray report from 2025-02-10 conclude? | keywords |
| documents | `doc_semantic` | Which patient has a chest X-ray report describing <5 distinctive words>? (must match exactly one report) | surname |
| documents | `doc_transcription_lookup` | Summarise X's Laparoscopic Cholecystectomy note. | keywords |
| documents | `doc_transcription_semantic` | Which patient had a cardiology document about <description>? | surname |

Hard questions use lay wording; some phrasings are deliberately **not** in the expansion dictionary, and each question records whether it is, so the effect of expansion is measured honestly.

**Size at `--per-type 20`:** 313 questions (125 easy, 108 hard, 80 documents); **258 have a gold record** and are used for retrieval scoring; all types are used for answer scoring. A human review sheet (30 random non-easy questions with their evidence) is written with every run.

### 11.2 Answer runs

- **Earlier (2026-10-28):** 18 questions (one per type), 7 models on the laptop CPU and Gemini (§17.3).
- **D (2026-10-30):** **54 questions per model** (3 per type: 24 easy, 18 hard, 12 documents), the same questions, retrieval, prompt and computed facts for every model, chunk context, 8 models, 432 answers, each scored by the deterministic checks and the LLM judge. Resumable files: `reports/gen_large.jsonl` (Gemini), `gen_large_local.jsonl` (qwen), `gen_large_nvidia.jsonl` (NVIDIA); `scripts/summarise_d.py` builds the tables.

---

## 12. Evaluation metrics

### 12.1 Retrieval (k = 8)

Scored after removing duplicates and the always-pinned profile chunk (unless the gold is the profile), so the search itself is measured.

| Metric | Definition |
|---|---|
| Hit@8 | 1 if any gold record is in the top 8 |
| MRR | 1 / rank of the first gold record (0 if not in the top 8) |
| Recall@8 | share of gold records in the top 8 (cohort questions have several) |
| nDCG@8 | Σ rel / log2(rank + 1), divided by the best possible |
| Latency | seconds per search |

### 12.2 Answers: deterministic checks

| Metric | Definition |
|---|---|
| `fact_recall` | share of expected key facts present (a fact may list alternatives, `ct\|computed tomography`) |
| `number_correct` | the exact expected number appears |
| `negative_correct` | says no / never / none and does not start with "yes" |
| `abstain_correct` | says "not found" exactly when it should |
| **`auto_correct`** | one pass/fail per question from the check that fits its type |
| False "not found" | share of answerable questions where the model said the records lack the answer |
| Cites sources | answer cites at least one [S#] |
| Citation precision | share of cited sentences whose content words are found in the source they cite |
| Citation coverage | share of factual sentences that carry a citation |
| Invalid citations | citations to a source number that does not exist (0 for every model) |
| Generation time | seconds to generate the answer (median and p95) |

### 12.3 Answers: LLM judge

A separate model gets the question, the reference answer, the computed facts and the retrieved sources, and returns JSON with four 1-5 scores and a one-sentence reason:

| Score | Meaning |
|---|---|
| correctness | agrees with the reference answer ("correct" = 4 or 5) |
| faithfulness | every claim supported by the sources or facts |
| answer_relevance | answers what was asked, without padding |
| context_relevance | the retrieved sources contain what is needed (measures retrieval) |

**Agreement** = share of answers where the judge's correct/incorrect matches `auto_correct`. **Judge:** NVIDIA `nemotron-3.5-super-vl-120b-a12b` (free API credits, the default since 2026-10-30). Earlier runs used Gemini (`gemini-3-flash-preview`, then `gemini-3.6-flash`), whose free tier allowed only about 10 judge calls a day. If the judge's reply is not valid JSON (it sometimes drops a quote before `reason`), the four scores are read from the reply directly.

### 12.4 Confidence intervals

Every mean is reported with a **bootstrap 95% interval** (1,000 resamples, seeded), and paired differences are bootstrapped per question. With small samples the intervals are wide, and differences inside them are treated as directions, not proof.

---

## 13. Results: retrieval

### 13.1 Default configuration on the full test set

Full index, 258 questions with a gold record, after the filter-order fix (`eval_20261028_2246`, `dense_weight=1.0`):

| Group | Questions | Hit@8 | MRR | Recall@8 | nDCG@8 | Latency (CPU) |
|---|---:|---|---|---|---|---:|
| Easy | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.5 s |
| Hard | 78 | 0.987 [0.96-1.00] | 0.905 [0.85-0.95] | 0.983 [0.95-1.00] | 0.920 [0.88-0.96] | 2.4 s |
| Documents | 80 | 0.925 [0.86-0.97] | 0.859 [0.79-0.93] | 0.925 [0.86-0.97] | 0.875 [0.81-0.94] | 8.3 s |

With the current default (`dense_weight=0.5`) and the fixed `doc_semantic` questions, on the RTX 4050 GPU (§16): **MRR 0.952, Recall@8 0.984, nDCG@8 0.960; hard MRR 0.924, documents MRR 0.921**, about 0.25 s per document search.

MRR by type: lookups, latest, counts, semantic and typo names 1.00; hard paraphrase 0.938; hard trend 0.875; hard semantic cohort 0.815; document patient lookup 0.975; transcription lookup 1.00; transcription semantic 0.90; `doc_semantic` (an Open-i report by content) 0.73-0.84, the hardest case, because many chest X-ray reports use near-identical wording.

### 13.2 Ablations: what each component contributes

`eval_20261027_2033` (100 easy, 78 hard, 40 Open-i document questions), MRR@8:

| Configuration | Easy | Hard | Documents | Latency |
|---|---:|---:|---:|---:|
| **hybrid + MedCPT reranker (fuse), default** | 1.000 | 0.905 | 0.808 | 2.8 s |
| hybrid + MedCPT reranker (replace) | 1.000 | 0.915 | 0.793 | 2.8 s |
| hybrid + MiniLM reranker (fuse) | 1.000 | 0.885 | 0.797 | 0.7 s |
| hybrid + MiniLM reranker (replace) | 1.000 | 0.872 | 0.851 | 0.6 s |
| hybrid, no reranker | 1.000 | 0.875 | 0.762 | 0.1 s |
| dense only | 0.995 | 0.913 | 0.621 | 0.1 s |
| BM25 only | 1.000 | 0.865 | 0.837 | 0.05 s |
| no query expansion | 0.995 | 0.793 | 0.797 | 0.6 s |
| **no metadata filters** | **0.495** | **0.509** | **0.507** | 4.1 s |

- **Metadata filtering is the biggest single win:** it roughly doubles MRR.
- **Query expansion** adds about 0.11 MRR on hard questions. On paraphrases outside the dictionary the default still finds 92.6% of gold records (Recall@8), vs 100% inside it.
- **The reranker must be medical:** a general web-trained reranker (`bge-reranker-base`) made hard questions worse than no reranker (0.845 vs 0.876); MedCPT helps, and fusing its order with the hybrid order is the safest.
- **Dense alone** is weak on keyword-like document questions; **BM25 alone** is weak on paraphrases; hybrid covers both.
- An earlier check showed bge-small ≥ a hashing baseline everywhere; on easy questions the filters do almost all the work (`eval_20261027_1726`).

---

## 14. Results: A, embedding models

**Setup:** a benchmark subset of 11,605 records and 15,749 chunks (every gold record plus a random sample), the same 258 questions, the default pipeline, one index per model (`python -m medrag bench embed`, `reports/bench_embed_20261029_2258/`).

| Model | Dim. | Dense MRR | Dense Recall@8 | Dense nDCG@8 | Hybrid+rerank MRR | Recall@8 | nDCG@8 | Documents MRR | Query embedding |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **bge-small-en-v1.5 (kept)** | 384 | 0.918 | 0.953 | 0.927 | 0.938 | 0.981 | 0.948 | 0.855 [0.78-0.92] | 52 ms (CPU) |
| bge-base-en-v1.5 | 768 | 0.926 | 0.965 | 0.936 | 0.950 | 0.984 | 0.958 | 0.888 [0.83-0.94] | 8 ms |
| e5-base-v2 | 768 | 0.928 | 0.957 | 0.935 | 0.947 | 0.988 | 0.957 | 0.877 [0.81-0.93] | 7 ms |
| MedCPT (query / article) | 768 | 0.895 | 0.934 | 0.905 | 0.942 | 0.977 | 0.950 | 0.881 [0.81-0.94] | 5 ms |
| pubmedbert-base-embeddings | 768 | 0.908 | 0.934 | 0.915 | 0.934 | 0.988 | 0.947 | 0.834 [0.76-0.89] | 8 ms |
| nomic-embed-text (Ollama) | 768 | 0.928 | 0.965 | 0.938 | **0.951** | 0.988 | **0.960** | **0.895** [0.83-0.95] | 2,129 ms |

GPU query times except bge-small (CPU) and nomic (Ollama HTTP API per query).

- Easy questions score 1.000 for every model; hard questions 0.931-0.952 with overlapping intervals.
- Models differ only on document content search (`doc_semantic`): hybrid + rerank 0.49 (pubmedbert) to 0.75 (nomic), bge-small 0.55.
- **Medical embedding models are not better here**; BM25 and the reranker close most of the gap for every model.
- Every alternative doubles the vector size (768 vs 384 dimensions).

**Decision: keep bge-small.** The rule was to switch only for a clear win; the best gain (+0.013 MRR, nomic) is inside the intervals, for twice the vector size and about 2 s per query.

---

## 15. Results: B, chunking and answer context

### 15.1 Chunking strategies

Same subset, questions and pipeline, bge-small (`reports/bench_chunk_20261029_2325/`). `section-N` splits at report sections with at most N words; `recursive-N-oM` is a generic recursive splitter with an M-word overlap.

| Chunking | Chunks | Mean words | Dense MRR | Hybrid+rerank MRR | Recall@8 | nDCG@8 | Documents MRR |
|---|---:|---:|---:|---:|---:|---:|---|
| **section-220 (kept)** | 15,749 | 119 | 0.918 | 0.938 | 0.981 | 0.948 | 0.855 [0.78-0.92] |
| section-120 | 24,124 | 97 | 0.920 | 0.938 | 0.977 | 0.947 | 0.857 [0.79-0.92] |
| section-350 | 13,408 | 129 | 0.919 | 0.938 | 0.981 | 0.948 | 0.855 [0.79-0.92] |
| recursive-220-o40 | 16,422 | 125 | 0.920 | 0.938 | 0.981 | 0.948 | 0.855 [0.78-0.92] |
| recursive-220-o0 | 15,404 | 117 | 0.920 | 0.940 | 0.981 | 0.949 | 0.861 [0.79-0.92] |
| recursive-120-o20 | 27,259 | 100 | 0.916 | 0.936 | 0.977 | 0.946 | 0.849 [0.78-0.91] |

No difference (overall MRR within 0.004): most records are short (mean 119 words), so the size limit rarely applies; smaller chunks only make the index 53-73% larger. **Decision: keep `section-220`.**

### 15.2 Chunk vs whole-record (parent) context for answers

Retrieval unchanged; only what the LLM reads changes. 60 document questions (15 per type), Gemini 3.5 Flash-Lite, deterministic checks and citation verification:

| Context | Correct | Fact recall | False "not found" | Cites sources | Citation precision | Citation coverage | Median s |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunk | 83% [73-93] | 0.89 | 5% | 92% | 0.96 | 0.91 | 1.4 |
| **parent** | **93%** [87-98] | 0.93 | 5% | 97% | 0.96 | 0.88 | 1.7 |

Paired per question: parent better on 6, worse on none, **+0.10 [0.03-0.18]**. Transcription lookups rise from 67% to 100%: long notes are split into several chunks, and a question about the conclusion often retrieves a chunk without it. Citations stay as precise. **Decision: parent context is the default** (`ANSWER_CONTEXT=parent`).

---

## 16. Results: C, retrieval tuning and ablations

**Setup:** full index, 258 questions, RTX 4050 GPU, one setting changed at a time from the default (`rrf_k=60`, `dense_weight=1.0`, `candidates=60`, `rerank_pool=30`, MedCPT fuse). The `doc_semantic` questions were fixed first: 9 of the old 20 matched several near-identical reports, so questions must now match exactly one (`doc_semantic` MRR 0.562 → 0.727 from that fix alone). `reports/bench_tune_20261029_2344/`.

| Variant | MRR | Recall@8 | nDCG@8 | Hard MRR | Documents MRR | `doc_semantic` | Latency (documents) |
|---|---:|---:|---:|---:|---|---:|---:|
| default (then) | 0.940 | 0.983 | 0.950 | 0.905 | 0.901 [0.84-0.95] | 0.727 | 0.36 s |
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
| no reranker | 0.927 | 0.983 | 0.940 | 0.875 | 0.887 [0.82-0.95] | 0.700 | 0.03 s |

LLM query transformations (qwen2.5:3b before searching), on the 100 semantic questions only:

| Variant | MRR | Recall@8 | nDCG@8 | Hard MRR | Documents MRR | Latency |
|---|---:|---:|---:|---:|---|---:|
| default | 0.876 | 0.957 | 0.893 | 0.876 | 0.814 [0.71-0.91] | 0.42 s |
| query rewrite | 0.884 | 0.957 | 0.898 | 0.848 | 0.861 [0.75-0.95] | 4.58 s |
| multi-query | 0.859 | 0.953 | 0.878 | 0.852 | 0.807 [0.69-0.91] | 4.54 s |
| HyDE | 0.863 | 0.947 | 0.882 | 0.832 | 0.826 [0.72-0.93] | 4.93 s |

**Confirmation of `dense_weight=0.5` on 240 fresh document questions** (60 per type, new seed):

| dense_weight | MRR | Recall@8 | nDCG@8 | `doc_semantic` |
|---|---:|---:|---:|---:|
| 1.0 | 0.873 [0.83-0.91] | 0.929 | 0.887 | 0.690 |
| **0.5** | **0.889** [0.85-0.93] | **0.946** | **0.903** | **0.752** |

Paired: better on 18 questions, worse on 2, **+0.016 [0.01-0.03]** overall and +0.062 [0.02-0.11] on `doc_semantic`; the interval excludes zero.

Findings:
- **`dense_weight=0.5` adopted** (BM25 counts twice as much as dense): exact terms matter where reports share near-identical wording.
- **The reranker earns its place:** removing it costs 0.013 MRR overall and 0.030 on hard questions. bge-reranker-v2-m3 matches MedCPT but is a 2.3 GB model and 2-4x slower.
- `rrf_k`, `candidates`, `rerank_pool` move MRR by 0.006 or less.
- **LLM query transformations are not worth it:** query rewrite +0.008 net for about 4 s and an LLM call per question; multi-query and HyDE are worse. None adopted.

---

## 17. Results: D, answer generation (8 models)

### 17.1 Deterministic checks (54 questions per model)

| Model | Runs on | Easy | Hard | Documents | **All** | Fact recall | Correct "not found" | False "not found" | Cites sources | Citation precision | Citation coverage | Generation s (median / p95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3 Flash preview | Google free tier | 100% | **89%** | 92% | **94%** | 0.99 | 3 of 3 | 4% | **94%** | 0.92 | 0.74 | 3.7 / 30.7 |
| Gemini 3.5 Flash | Google free tier | 100% | 83% | **100%** | **94%** | 0.99 | 3 of 3 | 6% | 78% | **0.98** | 0.84 | 4.7 / 30.0 |
| Gemini 3.1 Flash-Lite | Google free tier | 100% | 78% | 92% | 91% | 0.94 | 3 of 3 | 12% | 78% | 0.90 | **0.89** | 1.3 / 32.3 |
| Gemini 3.5 Flash-Lite | Google free tier | 100% | 83% | 83% | 91% | 0.94 | 3 of 3 | 8% | 74% | 0.93 | 0.84 | 3.8 / 9.0 |
| NVIDIA Nemotron 3.5 Super † | NVIDIA API | 100% | 83% | 83% | 91% | 0.95 | 3 of 3 | **2%** | 85% | **0.98** | 0.66 | **0.9** / 13.0 |
| NVIDIA Nemotron 3.5 Lightning | NVIDIA API | 88% | 72% | 67% | 78% | 0.79 | 3 of 3 | **0%** | 81% | 0.97 | 0.20 | 1.2 / 5.6 |
| qwen3:1.7b | laptop GPU (Ollama) | 92% | 83% | 83% | 87% | 0.93 | 3 of 3 | 29% | 48% | 0.65 | 0.45 | 3.8 / 7.0 |
| qwen2.5:3b (default local) | laptop GPU (Ollama) | 96% | 83% | 33% | 78% | 0.79 | 3 of 3 | 10% | 31% | 0.88 | 0.10 | 3.9 / 9.2 |

No answer cited a source that did not exist.

### 17.2 LLM judge (NVIDIA Nemotron 3.5 Super, 1-5)

| Model | Judged | **Correct** | Easy | Hard | Documents | Correctness | Faithfulness | Answer relevance | Context relevance | Agrees with checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemini 3 Flash preview** | 54 | **96%** | 100% | **89%** | 100% | **4.83** | **4.93** | 4.78 | 4.74 | 98% |
| Gemini 3.5 Flash | 54 | 94% | 100% | 83% | 100% | 4.81 | 4.87 | **4.81** | 4.74 | 100% |
| NVIDIA Nemotron 3.5 Super † | 54 | 94% | 100% | 83% | 100% | 4.74 | 4.83 | 4.80 | 4.74 | 96% |
| Gemini 3.1 Flash-Lite | 54 | 93% | 100% | 78% | 100% | 4.69 | 4.91 | 4.74 | 4.67 | 98% |
| Gemini 3.5 Flash-Lite | 54 | 93% | 100% | 83% | 92% | 4.70 | 4.87 | 4.72 | 4.70 | 98% |
| NVIDIA Nemotron 3.5 Lightning | 54 | 89% | 96% | 78% | 92% | 4.48 | 4.63 | 4.65 | 4.74 | 85% |
| qwen3:1.7b | 54 | 78% | 100% | 50% | 75% | 4.19 | 4.52 | 4.22 | 4.72 | 80% |
| qwen2.5:3b | 54 | 67% | 83% | 56% | 50% | 3.67 | 4.43 | 3.96 | 4.72 | 81% |

† Nemotron 3.5 Super is also the judge, so it grades its own answers; its keyword-check score (91%) is the independent measure.

Findings:
- **The Gemini models are within two answers of each other** (93-96%). Gemini 3 Flash preview is best overall and on hard questions; Gemini 3.5 Flash answered all 12 document questions correctly.
- **NVIDIA is the fastest** (median under 1.2 s) and almost never says "not found" wrongly (0-2%). Nemotron Super matches the Flash-Lite models on the checks (91%). Nemotron Lightning is weaker on the checks (78%) but rated 89% by the judge, the only model the judge rates above the checks, so the judge may favour its own model family.
- **The judge catches what keyword checks miss.** For the local models the judge is about 10 points stricter (qwen3 87% → 78%, qwen2.5 78% → 67%), while for Gemini the two agree on 98-100%. Of 21 disagreements on local answers, 16 are passes the judge rejects: a trend read backwards ("interval improvement" read as "unchanged"), a negative question answered "not found" instead of "no study reports it", study counts swapped between two patients with the same name, and cohort lists padded with non-matching patients. The other 5 are right answers worded differently from the expected keywords.
- **Context relevance is 4.7 for every model**: all models get the same retrieved context, so this score measures retrieval, which is good.
- **On a GPU, local 1.7-3B models are as fast as the cloud** (about 4 s median), but clearly less accurate, especially on hard and document questions.
- **Every model fails most finding counts** (`hard_count_finding`, 0-33%): a pipeline issue (§22), not a model one.
- **Sample size:** 3 questions per type and 12 document questions, so one answer moves a type by 33 points and documents by 8.

### 17.3 Earlier comparison: 7 models, 18 questions, laptop CPU (2026-10-28)

| Model | Runs on | Easy (8) | Hard (6) | Documents (4) | **Total** | "Not found" only when right | Cites sources | Seconds per answer (mean / median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3 Flash | cloud | 100% | 83% | 75% | **89%** | 89% | 89% | 6.8 / 3.1 |
| Gemini 3.1 Flash-Lite | cloud | 100% | 83% | 75% | **89%** | 78% | 67% | 15.3 / 11.2 |
| qwen3:1.7b | CPU | 100% | 83% | 75% | **89%** | 44% | 56% | 43 / 46 |
| qwen3:4b | CPU | 100% | 83% | 75% | **89%** | 83% | 89% | 446 / 499 |
| llama3.2:3b | CPU | 100% | 83% | 50% | 83% | 72% | 22% | 57 / 55 |
| phi4-mini (3.8B) | CPU | 100% | 50% | 75% | 78% | 83% | 39% | 56 / 57 |
| qwen2.5:3b | CPU | 75% | 67% | 50% | 67% | 89% | 17% | 55 / 54 |

This run found a **test-set bug**: the cohort-count answer counted only patients with a unique name (expected 99, truth 101), so all 7 models were marked wrong on a question they got right. Fixed, with a regression test, and all 126 answers were re-scored without calling any model again. The 54-question D run (17.1-17.2) supersedes this one: three times the sample, scored by the judge too.

---

## 18. E: production controls and load test

Built and tested (`medrag/security/`, `tests/test_production.py`), **not** security-reviewed and not run on real patient data: a design to build on, not a certification.

| Control | How it works |
|---|---|
| Sign-in | `USERS_FILE` (git-ignored JSON) turns it on; PBKDF2-SHA256 hashes, 200,000 iterations, random salt, constant-time comparison; wrong user and wrong password look and time the same; passwords at least 12 characters (`python -m medrag user`) |
| Roles | **admin**: all records; **clinician**: only assigned patients (optionally documents of their specialties); **researcher**: aggregate counts only; **auditor**: no records, reads the audit log |
| Access before the LLM | allowed rows are a retrieval mask the filter relaxation can never drop, and the same mask limits the computed counts; a clinician naming another patient gets "No patient matching..." without an LLM call, whatever the question says |
| Audit log | `reports/audit.jsonl`, append-only: user, role, time, patients and records read or refused, LLM used, pseudonymised or not, injection flags; question text stored with names, MRNs and dates redacted; sign-ins logged; Audit tab in the app |
| PHI for cloud LLMs | `PHI_CLOUD=pseudonymize` (default): names, MRNs, patient ids, accessions → `[PATIENT_1]`, `[MRN_1]` ... before the prompt leaves the machine, restored in the streamed answer; `block` refuses cloud models; users need `cloud_llm: true` |
| Prompt injection | hidden Unicode removed at ingest; record lines that look like instructions ("ignore previous instructions") replaced and counted; sources fenced in `<source>` tags; access never depends on the prompt |
| Monitoring | `reports/metrics.jsonl` per request: stage latencies, tokens, estimated cost, errors; `python -m medrag metrics` gives p50/p95/p99, error rate, cost and cache hits per LLM |
| Caching | search results (256 entries, keyed by question, settings and user) and query embeddings (1,024 entries) in memory |
| Continuous evaluation | `eval --save-baseline` stores metrics; `eval --gate baseline.json` fails if retrieval MRR or recall drops by more than 0.02 or the answer pass rate by more than 0.10; tests check retrieval on easy and hard sets against a fixed floor |

**Load test** (`python -m medrag loadtest`, retrieval + facts + prompt, no LLM, full index, 4-core laptop CPU, 77-question mix):

| Run | Requests | Concurrency | Throughput | p50 | p95 | p99 | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| cache off | 40 | 1 | 0.18 req/s | 3.2 s | 14.8 s | 16.6 s | 0 |
| cache off | 40 | 4 | 0.29 req/s | 9.0 s | 28.9 s | 30.7 s | 0 |
| cache on | 80 | 4 | 0.24 req/s | 9.9 s | 49.0 s | 68.5 s | 0 |

No errors in 160 requests. The CPU is the limit: the slow tail is document questions, where MedCPT reads 30 long reports (15-30 s on this CPU vs about 0.4 s on the laptop GPU). For more users: a GPU for the reranker, `rerank_pool=20` (same MRR, half the time), several app workers.

---

## 19. Final configuration and the decisions behind it

| Setting | Value | Evidence |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d) | A: no model clearly better (§14) |
| Chunking | section-aware, 220 words, contextual header | B: all six strategies equal (§15.1) |
| Answer context | `ANSWER_CONTEXT=parent` | B: 93% vs 83% (§15.2) |
| Search | hybrid dense + BM25, 60 candidates each, RRF k 60 | ablations (§13.2) |
| Fusion weight | `DENSE_WEIGHT=0.5` | C: +0.012 / +0.016 confirmed (§16) |
| Reranker | `ncbi/MedCPT-Cross-Encoder`, pool 30, fused with hybrid order | §13.2, §16 |
| Query transformations | off (rewrite / multi-query / HyDE available) | C: not worth 4 s per question (§16) |
| Metadata filters | on, with relaxation in a fixed order | biggest win (§13.2) |
| Sources to the LLM | 8 + pinned profile / latest | |
| Answer LLM (local default) | `ollama:qwen2.5:3b`; `qwen3:1.7b` scores higher (78% vs 67% by the judge) | D (§17) |
| Answer LLM (cloud option) | Gemini 3 Flash / 3.5 Flash-Lite, NVIDIA Nemotron, AWS Bedrock | D (§17) |
| Judge | `nvidia:nvidia/nemotron-3.5-super-vl-120b-a12b` | free credits; Gemini free tier about 10 calls a day |
| PHI to cloud | `pseudonymize` | E (§18) |

---

## 20. Engineering quality: tests, audit, problems solved

**Tests:** 29, all passing (`.venv/Scripts/python -m pytest -q`). They build a small dataset with a hashing embedder (no downloads) and run the real pipeline: dirty raw data fails the checks and clean data passes; context headers; incremental rebuild reuses vectors; parser and retrieval; exact counts; metric formulas; query expansion; answer scoring; hard and document test sets; document cleaning and linking; long-text chunking; every audit fix; Bedrock streaming with a fake client; hidden-Unicode removal; production controls (roles, access mask, audit, pseudonymiser, injection guard, quality gate, retrieval floor).

**Code audit** (whole repository): 6 findings fixed with regression tests, each confirmed to fail on the old code:

| # | Finding | Fix |
|---|---|---|
| 1 | counts used the relaxed filter ("how many MRIs" for a patient with none returned their other studies) | count with every filter applied |
| 2 | "tell **us**" parsed as ultrasound | short modality codes only in capitals |
| 3 | patients with no MRN dropped during de-duplication | kept and logged |
| 4 | `watch` missed updates that kept the row count | signature from file times and sizes or PostgreSQL counters |
| 5 | a real error reading documents silently dropped all of them | only "not ingested yet" is ignored |
| 6 | re-indexing overwrote memory-mapped files of a running app | versioned files + atomic manifest swap |
| 13 | cohort-count test answer excluded shared names (99 vs 101) | fixed, 126 answers re-scored |
| 15 | a body-part word in a document question dropped the documents filter | relax body part before documents (documents Hit@8 0.875 → 0.925) |

Still open (6): relaxation notes can mention unset filters; `--max-answers` rounding per type; a partly extracted Open-i archive is not repaired; the "abnormal" rule is written twice; a duplicated not-found message and an unused property; finding counts ignore wording outside the dictionary (§22).

**Problems hit and solved:**

| Problem | Cause | Solution |
|---|---|---|
| general reranker hurt hard questions | web-trained, not medical | MedCPT, fused with the hybrid order |
| counts depended on retrieval | LLM counted the 8 sources | exact computed facts from the full metadata |
| counts by finding were off | AI label 14% wrong | count by the radiologist's impression |
| name questions mixed two patients | shared names | return all matches, per-MRN facts, `ambiguous_name` tests |
| lay terms missed reports | "slipped disc" vs "disc extrusion" | medical query expansion |
| MTSamples headings glued to text | CSV artefact | rebuild headings before chunking |
| judge blocked for days | Gemini free tier about 10 judge calls a day | NVIDIA judge; resume files; stored answers re-judged without regenerating |
| judge replies not valid JSON | judge dropped a quote | scores read from the reply directly |
| GPU crashes and stalls | 6 GB VRAM: batch 128 overflowed; two Ollama models at once thrashed | batch 64; one Ollama model at a time |
| runs frozen | laptop sleep / battery GPU cap (35 W) | charger on, sleep off |
| Ollama embedding runner died mid-batch | runner crash | retry with back-off |
| Qwen3 printed its reasoning | thinking mode | thinking off, `<think>` removed |

---

## 21. Hardware, models, speed and cost

**Machines:** the original measurements (§13.1, §13.2, §17.3, load test) ran on a 4-core Intel i7-8665U laptop with 15.8 GB RAM and no GPU. The 2026-10-29/30 benchmarks and D ran on a laptop with an **NVIDIA RTX 4050 6 GB** GPU (driver 617.42, CUDA 13.4), Python 3.13 in a uv venv, torch 2.15.0+cu132, Ollama 0.40.13.

| Role | Model | Size | Where |
|---|---|---:|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | 129 MB | in process (GPU when present) |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | 837 MB | in process |
| Local answers | `qwen2.5:3b` (default), `qwen3:1.7b` | 1.9 GB, 1.4 GB | Ollama |
| Cloud answers | Gemini 3 Flash preview, 3.5 Flash, 3.1 / 3.5 Flash-Lite; NVIDIA Nemotron 3.5 Super / Lightning | | free API tiers |
| Judge | NVIDIA `nemotron-3.5-super-vl-120b-a12b` | | NVIDIA API free credits |
| Production option | AWS Bedrock (e.g. `us.amazon.nova-lite-v1:0`) | | provider built, not tested live |

**Speed:** loading the index and embedder 14 s (1.1 GB RAM); first search 10 s (loads MedCPT); later searches 0.6 s for a simple question on CPU, 0.25-0.4 s for document questions on the GPU. Answers: local 3B models about 4 s on the laptop GPU vs 43-57 s on CPU (qwen3:4b about 450 s on CPU); Gemini 1.3-4.7 s; NVIDIA 0.9-1.2 s (medians).

**Disk:** raw 19 MB, clean 14 MB, index 442 MB, Hugging Face model cache about 1 GB.

**Cost: zero.** Open-source models, synthetic and public data, the developer's laptop, and free API tiers (Gemini, NVIDIA). Bedrock would be pay-per-token: a typical question is about 1,100 input and 100-300 output tokens.

---

## 22. Limits, risks and next steps

**Limits**
- **Finding counts with unknown wording fail for every model** (`hard_count_finding`): "stones in the gallbladder" is not in the dictionary, so no finding filter is applied and the facts count every study. Fix: recognise findings by meaning as a fallback, or grow the dictionary.
- **Finding an Open-i report by content alone** is the weakest retrieval task (`doc_semantic` MRR 0.73-0.84): many reports are nearly identical.
- **Keyword checks overstate small local models** by about 10 points (§17.2); use the judge as well.
- **Small answer samples:** 54 questions per model; differences of a few points are directions.
- **One judge model**, which also answered; a second judge would separate judge bias from model quality.
- **Synthetic data:** real reports are more varied and messier, so results overstate real-world accuracy.
- The production controls are not security-reviewed; six audit findings are open (§20).

**Risks and compliance**
- Open-i is CC BY-NC-ND 4.0 (non-commercial, attribution); MTSamples' redistribution terms are unclear, so neither file is committed.
- No real patient data is used. Real data needs a HIPAA BAA (or local equivalent: DPDP Act, GDPR) with every service that sees it.
- Local mode keeps everything on the machine; cloud providers receive pseudonymised record text (dates and clinical text still leave the machine).
- Decision support only, not a medical device; "not found" means not in the indexed records, not that the patient is healthy; every answer must be checked against its cited sources.

**Before any clinical use:** validation on governed real data with clinicians reviewing answers; a security review and penetration test; single sign-on instead of the JSON user file; tamper-evident audit storage; signed agreements with cloud LLM providers.

**Next steps**
1. Answer count questions directly from the computed facts (no LLM needed).
2. Finding detection by meaning, to fix `hard_count_finding`.
3. Repeat D with parent context (now the default) and a second, independent judge.
4. Consider `qwen3:1.7b` as the local default (78% vs 67% by the judge).
5. Live test on AWS Bedrock; fix the open audit findings.

---

## 23. How to reproduce

```bash
# setup (Windows; uv-managed Python 3.13)
uv venv .venv --python 3.13 --seed
uv pip install torch --torch-backend=auto
uv pip install -r requirements.txt
copy .env.example .env          # set DATA_BACKEND=parquet and your own keys; never commit .env

# data and index
python -m medrag pipeline       # generate -> check -> clean -> index
python -m medrag documents      # Open-i + MTSamples (download them first)
python -m medrag index          # only new chunks are embedded

# use it
python -m medrag ask "What was the most recent imaging study for <name>?"
streamlit run app.py

# evaluate
python -m medrag eval --max-answers 16 --judge
python -m medrag bench embed --models ...      # A
python -m medrag bench chunk                   # B
python -m medrag bench tune                    # C
powershell -File scripts/run_d_loop.ps1        # D, resumable, NVIDIA judge
python scripts/summarise_d.py                  # D tables
python -m medrag loadtest --concurrency 4 --requests 40
python -m medrag eval --save-baseline          # then --gate baseline.json in CI

# tests
python -m pytest -q
```

Fixed seeds: data 7, document linking 13, test set 11 (hard +1, documents +2), review sample 5, bootstrap 0, "now" 2026-10-01. With the same code, data files, seeds and package versions, the data, chunks, questions and retrieval metrics come out the same; LLM answers can vary slightly even at temperature 0.1. Data, indexes and `reports/` are git-ignored: they are rebuilt with the commands above.
