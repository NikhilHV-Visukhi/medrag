# MedRAG

Question answering over radiology records: patients, imaging studies, AI model predictions and
radiology reports. Rebuilt from the earlier `LLM_6_Auto` prototype (archived in `../_archive`).

```
PostgreSQL / parquet ──► check ──► clean ──► check ──► chunk ──► embed ──► hybrid index
                                                                              │
question ──► parse filters ──► filtered dense + BM25 ──► RRF ──► rerank ──► LLM ──► answer + sources
                                     └──── exact computed facts (counts, timelines) ────┘
```

## What's in each step

| Step | How it works |
|---|---|
| **Data** | `generate` creates a large synthetic hospital dataset (default 20k patients, about 100k studies). It includes demographics, chronic conditions, CT/MRI/X-ray/US/mammo/PET studies, AI predictions and structured reports with technique, comparison, findings, impression and recommendation. Follow-up scans refer back to earlier ones. Studies per patient follow a heavy-tailed distribution, so some patients have dozens. `import-legacy` loads the original `oads.*` tables instead. |
| **Data check** | Runs 27 checks: duplicates, orphans, test records, non-standard codes, impossible dates, empty or HTML reports, and confidence values out of range. Writes `reports/quality_raw.md` / `quality_clean.md`. |
| **Cleaning** | Each rule logs how many rows it changed (`reports/cleaning_log.md`). It merges duplicate registrations by MRN and re-points their studies, maps gender/modality/priority/status synonyms, strips HTML, removes test, orphan and future-dated rows, keeps the best report per study, and converts percent confidences to 0-1. Hidden Unicode (zero-width characters, bidi overrides, tag characters, which can carry invisible prompt-injection text) is removed from reports and outside documents with the Layer A cleaner vendored from [watermarks-remover](https://github.com/guillaumemeyer/watermarks-remover) (MIT, `medrag/data/text_unicode.py`). |
| **Outside documents** | `documents` loads real de-identified documents: 3,955 Open-i chest X-ray reports (NLM, with expert MeSH labels; put `NLMCXR_reports.tgz` in `data/source/openi/`) and the Kaggle MTSamples set of 4,999 transcriptions across 40 specialties (put `mtsamples.csv` in `data/documents/`; 2,333 remain after merging the same note filed under several specialties). Neither file is in git. They are checked and cleaned: de-identification marks become `[redacted]`, headings are rebuilt in transcriptions, duplicates are merged and short or empty ones removed (`reports/quality_documents.md`). Each document is then attached to a synthetic patient of matching gender and age. The text is real; the patient link is synthetic. |
| **Ingestion & chunking** | Builds three document types: study reports, patient profiles (conditions, counts, timeline, list of uploaded documents) and uploaded documents. Chunks split on section headings first (any upper-case heading such as `FINDINGS:` or `HISTORY OF PRESENT ILLNESS:`), then on sentences. Every chunk starts with a contextual header (patient, MRN, age, study, date, priority, indication), so it still makes sense when retrieved on its own. |
| **Embeddings** | Pluggable: `sentence-transformers` (local, default `BAAI/bge-small-en-v1.5`), `ollama`, `openai` or `gemini`. Query/document instruction prefixes are applied per model. Each model gets its own index folder, so you can compare them. Rebuilds are incremental: only new or changed chunks are embedded. |
| **Index** | Dense vectors use FAISS (exact search, switching to HNSW above 150k chunks). Sparse search uses BM25 via `bm25s`, with stemming and stopwords. Metadata is stored in parquet. |
| **Retrieval** | The question is parsed into filters: patient (MRN, ID, accession, or fuzzy name match), modality, body part, priority, date range, "latest", "how many", "abnormal", a finding, documents and specialty. **Medical query expansion** maps lay terms to report wording ("slipped disc" → disc herniation / "disc extrusion", 32 findings, 104 synonyms) for BM25, embeddings, the reranker and the LLM. Search runs only inside the matching records, so no other patient's chunks leak in. If nothing matches, filters are relaxed one at a time. Dense and BM25 results are fused with Reciprocal Rank Fusion, then reranked with a medical cross-encoder (`ncbi/MedCPT-Cross-Encoder`). The reranker order is fused with the hybrid order (`RERANK_BLEND=fuse`) rather than replacing it, so a reranker cannot undo a good first-stage result. The patient profile and latest studies are always included. |
| **Answering** | The LLM receives the sources plus **exact computed facts** (counts, breakdowns, the full timeline), so numbers never depend on which chunks happened to be retrieved. Answers cite `[S1]`... An unknown patient returns "not found" without calling the LLM. |
| **LLMs** | `ollama` (local), `anthropic`, `openai`, `gemini`, `groq`, `openrouter`, `bedrock` (AWS: any Bedrock model through the Converse API). Gemini, Groq and OpenRouter (`:free` models) have free API tiers, which are enough for evaluation and judging. Choose them in `.env` or per command as `provider:model`. |
| **Evaluation** | Builds a test set from the data itself, so every question has a known answer: lookups by name or MRN, latest study, counts, semantic search, cohort counts and unknown patients. Three sets: **easy** (template lookups), **hard** (lay wording, some deliberately outside the expansion dictionary; questions without dates; follow-up trends; patients who never had the finding; misspelt names; cohort counts by finding) and **documents** (find an uploaded report by patient or by content alone). `golden_review.csv` holds a sample with the gold evidence for a person to check. It scores retrieval (Hit@k, MRR, Recall@k, nDCG) for every configuration, with ablations (no reranker, no expansion, no filters) and bootstrap 95% confidence intervals. Answers are scored on fact recall, exact numbers, correct "not found" answers and citations. Each answer also gets one pass/fail from whichever exact check fits the question. The LLM judge, a different model from the one answering, scores correctness, faithfulness, answer relevance and context relevance from 1 to 5. The report shows how often the judge agrees with the exact checks. |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env          # then edit .env: DB password, models, API keys
```

To work without PostgreSQL, set `DATA_BACKEND=parquet` in `.env`. Tables are then stored under `data/`.

To use PostgreSQL, load the original database once if it doesn't exist yet (the dump is in `data/source/`):

```bash
createdb -U postgres oads_db
psql -U postgres -d oads_db -f data/source/oads_db.sql
```

MedRAG writes its own schemas, `medrag_raw` and `medrag`, and never modifies `oads`.

## Run

```bash
python -m medrag pipeline --patients 20000     # generate -> check -> clean -> index
python -m medrag check --stage clean           # quality report on the cleaned data
python -m medrag ask "What was the most recent imaging study for MRN0000042?"
python -m medrag ask                           # interactive
streamlit run app.py                           # web UI: chat, patient lookup, data quality, evaluation
```

Or run each step on its own:

```bash
python -m medrag generate --patients 100000 --dirty-rate 0.03   # bigger dataset
python -m medrag import-legacy                                 # or: use the original oads.* data
python -m medrag check --stage raw
python -m medrag clean
python -m medrag index --embed-provider sentence-transformers --embed-model BAAI/bge-base-en-v1.5
python -m medrag documents                                     # outside documents -> then run index again
python -m medrag watch --interval 30                           # keep index in sync with the DB
```

## Comparing models

Build one index per embedding model, then compare them, along with several LLMs:

```bash
python -m medrag index --embed-provider sentence-transformers --embed-model BAAI/bge-small-en-v1.5
python -m medrag index --embed-provider ollama --embed-model nomic-embed-text
python -m medrag index --embed-provider openai --embed-model text-embedding-3-small

python -m medrag eval ^
  --indexes sentence-transformers:BAAI/bge-small-en-v1.5,ollama:nomic-embed-text ^
  --llms ollama:qwen2.5:3b,anthropic:claude-opus-5-5 --judge
```

Results are written to `reports/eval_<timestamp>/` (`summary.md`, `retrieval.csv`, `generation.csv`, `testset.json`) and appear in the app's Evaluation tab.

## Real-time answers with AWS Bedrock

On CPU an answer takes about a minute, almost all of it the LLM. Bedrock runs the LLM as a managed service:
answers in a few seconds, many users at once, no servers or GPUs to manage, pay per request.

1. In the AWS console, open Bedrock > Model access and enable a model (e.g. Amazon Nova Lite, Llama, Claude).
2. Give this machine credentials: `aws configure` (or an IAM role when running on AWS). Keys never go in `.env`.
3. In `.env`: `LLM_PROVIDER=bedrock`, `LLM_MODEL=<model id from the console>`, `AWS_REGION=<region>`.

```bash
python -m medrag ask "Latest scan for MRN0000042?" --llm bedrock:us.amazon.nova-lite-v1:0
python -m medrag eval --llms bedrock:us.amazon.nova-lite-v1:0,ollama:qwen2.5:3b --max-answers 16
```

For real patient data, sign the AWS Business Associate Agreement (HIPAA) and use HIPAA-eligible services only.
The search index and MedCPT reranker still run on the app's CPU (about 3 s); set
`RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` to cut that to under a second.

Notes:
- API providers read their keys from `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`). Claude requests enable server-side refusal fallback.
- Embedding time on CPU: about 100k chunks with `bge-small` takes 10-20 minutes. Rebuilds after a data change only embed what changed.

## Latest results (2026-10-27)

20,000 patients, 89,868 studies, 3,878 Open-i reports, 2,333 MTSamples transcriptions, 121,335 chunks. `BAAI/bge-small-en-v1.5` embeddings, CPU only.
The first index build took 2 h 50 min. Adding Open-i re-embedded only 7,428 chunks (12 min), and MTSamples 10,749 (50 min).

**Retrieval MRR@8** (218 questions with known gold documents: 100 easy, 78 hard, 40 documents; full table with 95% CIs in `reports/eval_20261027_2033/summary.md`):

| Configuration | Easy | Hard | Documents | Latency |
|---|---:|---:|---:|---:|
| **hybrid + MedCPT reranker (fuse) - default** | 1.000 | 0.905 | 0.808 | 2.8 s |
| hybrid + MedCPT reranker (replace) | 1.000 | 0.915 | 0.793 | 2.8 s |
| hybrid + MiniLM reranker (fuse) | 1.000 | 0.885 | 0.797 | 0.7 s |
| hybrid + MiniLM reranker (replace) | 1.000 | 0.872 | 0.851 | 0.6 s |
| hybrid, no reranker | 1.000 | 0.875 | 0.762 | 0.1 s |
| dense only | 0.995 | 0.913 | 0.621 | 0.1 s |
| BM25 only | 1.000 | 0.865 | 0.837 | 0.05 s |
| hybrid + rerank, **no query expansion** | 0.995 | 0.793 | 0.797 | 0.6 s |
| hybrid + rerank, **no filters** | 0.495 | 0.509 | 0.507 | 4.1 s |

What this shows:
- Metadata filtering is the biggest win: it doubles MRR.
- Medical query expansion adds about 0.11 MRR on hard questions (0.905 vs 0.793). On wording that is *not* in the dictionary, hybrid search still recovers 98% of gold documents (92.6% with the reranker).
- A general web-trained reranker (`bge-reranker-base`, tested earlier) made results *worse* than no reranker, and was slower. The medical MedCPT reranker helps. Fusing its order with the hybrid order is the safest choice overall.
- Dense search alone is weak on keyword-style document questions (0.27 on `doc_semantic`). BM25 is weak on paraphrases. Hybrid covers both.
- Differences of a few hundredths are inside the confidence intervals. Treat the top three rerank configurations as equivalent.

**Documents after adding MTSamples** (80 questions, default configuration; `reports/eval_20261028_2246`, after the filter-order fix): MRR 0.859, Hit@8 0.925 (0.809 and 0.875 before).
Finding a patient's document scores 0.98 (Open-i) and 1.00 (transcriptions). Finding a document by content alone is harder:
0.90 for transcriptions (0.70 before the fix) and 0.56 for chest X-ray reports, many of which use near-identical wording ("no acute cardiopulmonary disease").

**Answers** from local `qwen2.5:3b`, judged by `gemini-3-flash-preview` (16 questions, one per type; `reports/eval_20261028_0006`):

| | Easy | Hard | Documents |
|---|---:|---:|---:|
| Pass the exact check | 63% (75% after the test fix below) | 83% | 50% |
| Judge correctness (1-5) | 3.6 | 3.2 | 3.0 |
| Judge faithfulness (1-5) | 4.1 | 3.7 | 3.0 |
| Seconds per answer (CPU) | 76 | 97 | 82 |

- Lookups by name or MRN, latest study, per-patient counts, "not found", misspelt names, paraphrases and document lookups all score 5/5.
- The 3B model fails finding counts ("how many patients had X"), content search across many records and names shared by two patients. (It was also marked wrong on the cohort count, but its answer was right; see the test fix below.)
- The judge agrees with the exact checks on 88% of answers. Both disagreements are on the looser checks ("never had", trends), where the judge is likely right.
- With 16 questions the confidence intervals are wide. Free-tier quotas (Gemini: per model, per day) are the limit; `JUDGE_MODEL` can be any model with quota left.

**Seven models, local and cloud**, on the same 18 questions (one per type: 8 easy, 6 hard, 4 documents), exact checks only (`reports/eval_20261028_1044`, `_1350`, `_1847`). The scores are after fixing a test-set bug: the cohort-count question expected 99, but the true answer was 101, which every model gave (re-scored in `reports/model_comparison_7.csv`).

| Model | Runs on | Easy | Hard | Documents | Total | Says "not found" only when right | Cites sources | Generation seconds (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Gemini 3 Flash** | Google cloud | 100% | 83% | 75% | **89%** | 89% | 89% | **3** |
| Gemini 3.1 Flash-Lite | Google cloud | 100% | 83% | 75% | **89%** | 78% | 67% | 11 |
| **qwen3:1.7b** | laptop CPU | 100% | 83% | 75% | **89%** | 44% | 56% | 46 |
| qwen3:4b | laptop CPU | 100% | 83% | 75% | **89%** | 83% | 89% | 499 |
| llama3.2:3b | laptop CPU | 100% | 83% | 50% | 83% | 72% | 22% | 55 |
| phi4-mini (3.8B) | laptop CPU | 100% | 50% | 75% | 78% | 83% | 39% | 57 |
| qwen2.5:3b (default) | laptop CPU | 75% | 67% | 50% | 67% | 89% | 17% | 54 |

- **Accuracy:** the cloud models are no more accurate than the best local ones (16 of 18 each). Retrieval and the computed facts do most of the work.
- **Gemini 3 Flash is the best all-rounder:** top accuracy, careful about "not found", cites its sources, and answers in about 3 s. qwen3:4b is as careful but takes 7-8 minutes on CPU.
- **Two questions fail for every model, so they are pipeline issues:**
  - a finding count whose wording ("stones in the gallbladder") is not in the expansion dictionary;
  - a document question where "back" set a spine filter, which pushed out the documents filter. Now fixed: body part is relaxed before the documents filter, so the right patient is found (documents Hit@8 0.875 → 0.925).
- **Privacy:** Gemini sees the question and the retrieved records. That is fine for synthetic and public data, not for real patient data without an agreement.

## Benchmarks (2026-10-29)

Run with `python -m medrag bench embed|chunk|tune` (full tables in `PROJECT_REPORT.md` sections 20.8-20.10). Embedding models and chunking were compared on a 15,749-chunk subset; tuning on the full index. All 258 questions with a gold document, MRR@8 with hybrid search + MedCPT reranker.

| Experiment | Best variant | Current | Decision |
|---|---|---|---|
| Embedding model (6 tried) | nomic-embed-text 0.951 | bge-small 0.938 | **Keep bge-small.** +0.013 is inside the confidence intervals, at twice the vector size; nomic takes about 2 s per query through Ollama. |
| Chunking (6 tried) | recursive-220, no overlap 0.940 | section-220 0.938 | **Keep section-220.** All within 0.004; smaller chunks only add 53-73% more chunks. |
| Retrieval settings (16 tried) | `dense_weight=0.5` 0.952 | default 0.940 | **Keep the default for now.** `dense_weight=0.5` lifts Open-i content search from 0.73 to 0.84 but needs confirming on more than 20 questions. |

- Medical embedding models (MedCPT, pubmedbert) did no better than general ones; the reranker and BM25 close most gaps.
- Removing the reranker costs 0.013 MRR overall and 0.030 on hard questions. MedCPT beats MiniLM on hard questions; bge-reranker-v2-m3 matches MedCPT but is 2-4 times slower.
- LLM query rewrite, multi-query and HyDE add about 4 s per question and do not help overall (rewrite +0.008, the other two worse).

**Answers on 54 questions per model** (3 per question type, deterministic checks; `PROJECT_REPORT.md` section 20.11):

| Model | Easy | Hard | Documents | All | False "not found" | Median seconds |
|---|---:|---:|---:|---:|---:|---:|
| Gemini 3.5 Flash-Lite | 100% | 67% | 83% | **85%** | 8% | 2.7 |
| qwen3:1.7b (laptop GPU) | 88% | 72% | 75% | **80%** | 31% | 4.2 |
| qwen2.5:3b (laptop GPU, default) | 96% | 83% | 33% | **78%** | 10% | 3.9 |

On a GPU the local models are nearly as fast as the cloud. qwen2.5:3b fails document content questions; qwen3:1.7b too often says "not found". LLM-judge scores are still pending: the Gemini free tier allowed only 40 of 324 judge calls.

Helper scripts are in `scripts/`: `run_chain.sh` (benchmarks A-C), `run_d_loop.ps1` (answer evaluation with the judge, resumable), `summarise_d.py` (table above), `probe_quota.py` (checks Gemini quota without printing the key).

## Layout

```
medrag/
  config.py               settings from .env
  data/      generate.py  catalog.py  store.py  quality.py  clean.py
  documents/ loaders.py   prepare.py   (Open-i, MTSamples: load, check, clean, link)
  ingest/    documents.py chunking.py
  providers/ embeddings.py llms.py
  index/     store.py     (FAISS + BM25 + metadata, incremental)
  retrieval/ query_parser.py retriever.py expansion.py
  rag/       answer.py
  eval/      testset.py   run.py
  __main__.py             CLI
app.py                    Streamlit UI
tests/                    offline end-to-end test (pytest)
```

The data is synthetic. This is a research and decision-support prototype, not a medical device.
