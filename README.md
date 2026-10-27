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
| **Cleaning** | Each rule logs how many rows it changed (`reports/cleaning_log.md`). It merges duplicate registrations by MRN and re-points their studies, maps gender/modality/priority/status synonyms, strips HTML, removes test, orphan and future-dated rows, keeps the best report per study, and converts percent confidences to 0-1. |
| **Ingestion & chunking** | Builds two document types. Each study report becomes one document. Each patient has a profile with conditions, counts and a timeline of recent impressions. Chunks split on section boundaries first, then on sentences. Every chunk starts with a contextual header (patient, MRN, age, study, date, priority, indication), so it still makes sense when retrieved on its own. |
| **Embeddings** | Pluggable: `sentence-transformers` (local, default `BAAI/bge-small-en-v1.5`), `ollama`, `openai` or `gemini`. Query/document instruction prefixes are applied per model. Each model gets its own index folder, so you can compare them. Rebuilds are incremental: only new or changed chunks are embedded. |
| **Index** | Dense vectors use FAISS (exact search, switching to HNSW above 150k chunks). Sparse search uses BM25 via `bm25s`, with stemming and stopwords. Metadata is stored in parquet. |
| **Retrieval** | The question is parsed into filters: patient (MRN, ID, accession, or fuzzy name match), modality, body part, priority, date range, "latest", "how many", "abnormal". Search runs only inside the matching records, so no other patient's chunks leak in. If nothing matches, filters are relaxed one at a time. Dense and BM25 results are fused with Reciprocal Rank Fusion, then reranked with a cross-encoder. The patient profile and latest studies are always included. |
| **Answering** | The LLM receives the sources plus **exact computed facts** (counts, breakdowns, the full timeline), so numbers never depend on which chunks happened to be retrieved. Answers cite `[S1]`... An unknown patient returns "not found" without calling the LLM. |
| **LLMs** | `ollama` (local), `anthropic`, `openai`, `gemini`. Choose them in `.env` or per command as `provider:model`. |
| **Evaluation** | Builds a test set from the data itself, so every question has a known answer: lookups by name or MRN, latest study, counts, semantic search, cohort counts and unknown patients. It scores retrieval (Hit@k, MRR, Recall@k, nDCG) for every index × mode, including "no filters" as a baseline. Answers are scored on fact recall, exact numbers, correct "not found" answers and citations. An optional judge model, separate from the answering model, scores correctness and faithfulness from 1 to 5. |

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

Notes:
- API providers read their keys from `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`). Claude requests enable server-side refusal fallback.
- Embedding time on CPU: about 100k chunks with `bge-small` takes 10-20 minutes. Rebuilds after a data change only embed what changed.

## Latest results (2026-10-27)

20,000 patients, 89,868 studies, 108,934 chunks, `BAAI/bge-small-en-v1.5` + `ms-marco-MiniLM-L-6-v2` reranker, CPU only
(index build: 2 h 50 min once; later rebuilds only embed changed chunks).

| Retrieval mode | Hit@8 | MRR | Latency |
|---|---:|---:|---:|
| hybrid + rerank (with filters) | 1.000 | 0.997 | 0.5 s |
| hybrid | 1.000 | 0.997 | 0.15 s |
| dense only | 1.000 | 0.993 | 0.17 s |
| hybrid + rerank, **no filters** (old approach) | 0.767 | 0.664 | 6.7 s |

On semantic questions with no patient named, dense search alone scores MRR 0.85 and adding BM25 and the reranker raises it to 0.95.

Answers from local `qwen2.5:3b` (24 questions): every count exact (100%), correct "not found" on all unknown-patient questions (100%), 79% of key report facts present,
about 52 s per answer on CPU. Its weak spot is wording mismatches (question says "disc herniation", report says "disc extrusion"):
retrieval finds the right report, but the 3B model then answers "no patient". A stronger model (`anthropic:claude-opus-5-5`, OpenAI, Gemini,
or a larger local model) is recommended for production; compare with `eval --llms ... --judge`. `qwen3:4b` answered correctly but
always prints its reasoning and takes 3-6 min per answer on this CPU.

Full reports: `reports/eval_*/summary.md` (also in the app's Evaluation tab).

## Layout

```
medrag/
  config.py               settings from .env
  data/      generate.py  catalog.py  store.py  quality.py  clean.py
  ingest/    documents.py chunking.py
  providers/ embeddings.py llms.py
  index/     store.py     (FAISS + BM25 + metadata, incremental)
  retrieval/ query_parser.py retriever.py
  rag/       answer.py
  eval/      testset.py   run.py
  __main__.py             CLI
app.py                    Streamlit UI
tests/                    offline end-to-end test (pytest)
```

The data is synthetic. This is a research and decision-support prototype, not a medical device.
