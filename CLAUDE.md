# MedRAG: notes for Claude Code

Retrieval-augmented question answering over hospital imaging records (synthetic patients + public Open-i
chest X-ray reports + MTSamples transcriptions). Repo: https://github.com/nikhilhv414/medrag (branch main).
README.md explains the pipeline; PROJECT_REPORT.md is the full write-up with all evaluation tables.

## Rules (from the owner, always apply)
- Never print, echo or commit API keys. Keys live only in the git-ignored `.env`; never ask for keys in chat.
  AWS credentials come from the normal AWS chain only.
- Never push to GitHub without the owner's OK. Ask before downloading models or data.
- Data, indexes and reports stay out of git (`data/raw`, `data/clean`, `data/source`, `indexes/`, `reports/`,
  `users*.json`). Real patient data would need a HIPAA BAA: do not load any.
- Do not touch or commit the owner's untracked `LLM_Project_report.md` / `.pdf`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Environment
- Windows, Python 3.13 venv in `.venv` (made with uv; 3.14 also works). Run as `PYTHONIOENCODING=utf-8 .venv/Scripts/python -m medrag ...`.
- Current laptop: RTX 4050 6 GB, driver 617.42 (CUDA 13.4), torch 2.15.0+cu132 installed with `uv pip install torch --torch-backend=auto`.
  GPU runs fail or crawl when it is short of memory: keep `EMBED_BATCH_SIZE=64`, and run one Ollama model at a time
  (two models plus the reranker overflow 6 GB). Keep the laptop on the charger with sleep off: on battery the GPU is
  capped at 35 W, and sleep freezes runs.
- Ollama for local models: `qwen2.5:3b`, `qwen3:1.7b`, `nomic-embed-text`.
- GPU laptop: install CUDA PyTorch before requirements
  (`pip install torch --index-url https://download.pytorch.org/whl/cu126`), check
  `torch.cuda.is_available()`. sentence-transformers, CrossEncoder and MedCPTEmbedder use the GPU automatically.
  With 6 GB VRAM keep `EMBED_BATCH_SIZE=64`.
- Data/indexes/reports are not in git: they come from `medrag_transfer.tar` (unpack in the repo root).
  Without it, rebuild with the pipeline commands in README.md.
- Many files have CRLF line endings: edit with the Edit tool, not multi-line string replacement scripts.
- Tests: `.venv/Scripts/python -m pytest -q` (29 pass).

## Key settings (.env, non-secret)
EMBED_PROVIDER=sentence-transformers, EMBED_MODEL=BAAI/bge-small-en-v1.5, RERANK_MODEL=ncbi/MedCPT-Cross-Encoder,
RERANK_POOL=30, RERANK_BLEND=fuse, LLM_PROVIDER=ollama, LLM_MODEL=qwen2.5:3b, JUDGE_PROVIDER=gemini.
Judge: run with `JUDGE_MODEL=gemini-3.6-flash` set in the environment (the .env still says 3.8-flash, which
returned 503s). `gemini-2.5-flash` is closed to new keys. Free-tier daily quotas reset at midnight Pacific
(12:30 IST); a 429 `GenerateRequestsPerDayPerProjectPerModel-FreeTier` means wait for the reset.

## Work in progress (started 2026-10-28): improvements A-E
The owner asked for: (A) embedding model benchmark, (B) chunking comparison, (C) retrieval tuning with
ablations, (D) generation comparison + separate answer metrics + continuous evaluation, (E) production
controls. All code for A-E is written, committed and tested.

**Status 2026-10-30:** A, B and C are run and written up (PROJECT_REPORT.md 20.8-20.10, README Benchmarks).
D's deterministic checks are done for qwen2.5:3b, qwen3:1.7b and gemini-3.5-flash-lite (54 answers each, 20.11);
gemini-3.5-flash has 35, the other two Gemini models 3. (Update 2026-10-30 evening: D is judged by NVIDIA
`nemotron-3.5-super-vl-120b-a12b`, key NVIDIA_API_KEY in .env, `run_d_loop.ps1 -Judge nvidia:<model>`; 276 of 324
judged, report 20.11. Left: Gemini 3.5 Flash documents and 3 Flash preview hard/documents need Gemini answer quota.)
Old note: the Gemini judge was blocked by the free-tier quota (40 of 324
judged). Also done (2026-10-30, report 20.9, 20.10, 25.1-25.3): chunk vs parent context (parent 93% vs 83%,
now the default via `answer_context`; `run_d_loop.ps1` keeps `--context chunk` so the D resume files stay consistent), `dense_weight=0.5` confirmed on 240 fresh questions and made the
default, the E write-up and a CPU load test. Still to do: judged D runs (`scripts/run_d_loop.ps1 -Only cloud|local`,
check quota with `scripts/probe_quota.py`). The sections below are the original plan.

Done in code: `medrag/eval/bench.py` (A/B/C), `medrag/retrieval/rewrite.py` (query rewrite, multi-query, HyDE),
`medrag/security/` (users/RBAC, patient-level access before the LLM, audit + metrics JSONL, PHI
pseudonymisation for cloud LLMs, prompt-injection guard), citation verification + quality gate in
`medrag/eval/run.py`, `loadtest` / `metrics` / `user` CLI commands, Audit tab in app.py.

### A. Embedding benchmark (subset: 11,605 records / 15,749 chunks, same 258 questions for every model)
`indexes/bench/` holds the subset, its test set and one index per model (built ones are reused).

| model | dense MRR | dense Recall@8 | dense nDCG@8 | hybrid+rerank MRR | Recall@8 | nDCG@8 | embed s / 1k chunks (CPU) | query ms |
|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 (current) | 0.918 | 0.953 | 0.927 | 0.938 | 0.981 | 0.948 | (reused) | 52 |
| bge-base-en-v1.5 | 0.926 | 0.965 | 0.936 | 0.950 | 0.984 | 0.958 | 484 | 79 |
| e5-base-v2 | 0.928 | 0.957 | 0.935 | 0.947 | 0.988 | 0.957 | 462 | 74 |
| MedCPT (Query/Article) | 0.895 | 0.934 | 0.905 | 0.942 | 0.977 | 0.950 | 651 | 72 |

Still to run: `NeuML/pubmedbert-base-embeddings` (its index folder is incomplete: delete
`indexes/bench/embed__sentence-transformers__NeuML_pubmedbert-base-embeddings` if it has no manifest.json)
and `ollama:nomic-embed-text`. Known issue: `model_ram_mb` is 0 in models.csv; measure memory separately.
Decision rule: rebuild the full index with a new embedder only if it clearly beats bge-small (gains so far
are about +0.01 MRR, inside the confidence intervals, at ~2x vector size and slower queries).

### B. Chunking (not run yet)
`bench chunk` compares section-220 (current), section-120, section-350, recursive-220-o40, recursive-220-o0,
recursive-120-o20 on the same subset. Then compare answer faithfulness with `eval --context chunk` vs
`--context parent` on the document questions.

### C. Retrieval tuning (not run yet)
`bench tune` runs 16 variants on the full index (rrf_k, dense_weight, candidates 40/100, rerank_pool,
rerankers MedCPT / bge-reranker-v2-m3 / MiniLM / none, and query rewrite / multi-query / HyDE on the semantic
question types). Rewrites are cached in `reports/rewrite_cache.json`.

### D. Generation (partly run)
Resume file `reports/gen_large.jsonl` keeps finished answers; rerunning the same command skips them:
```
JUDGE_MODEL=gemini-3.6-flash PYTHONIOENCODING=utf-8 .venv/Scripts/python -m medrag eval --skip-retrieval --per-type 20 --max-answers 54 --llms gemini:gemini-3.5-flash,gemini:gemini-3.5-flash-lite,gemini:gemini-3.1-flash-lite,gemini:gemini-3-flash-preview --judge --resume reports/gen_large.jsonl
```
Status: 18 judged answers for gemini-3.5-flash, 2 for 3.5-flash-lite (the rest failed on judge quota), none yet
for 3.1-flash-lite and 3-flash-preview. Afterwards run the same command with
`--llms ollama:qwen3:1.7b,ollama:qwen2.5:3b` for the local models.

### All remaining benchmark steps in one go
`sh reports/run_chain.sh > reports/run_chain.log 2>&1` (in the transfer tar; runs `bench embed` for the
models above, then `bench chunk`, then `bench tune`, stopping at the first failure). Results land in
`reports/bench_<kind>_<timestamp>/summary.md`.

### Then
- Write A-E methods and results into PROJECT_REPORT.md and README.md, including: the doc_semantic test-set
  fix (9 of 20 old questions matched several reports; questions now must match exactly one), the
  keyword-coverage ranking experiment (not adopted: change within noise), the judge changes and quota notes,
  the security design, and that clinical readiness needs validation on governed real data.
- Optional: `loadtest` once the CPU/GPU is free; `eval --save-baseline` for the quality gate.
- Commit; push only with the owner's OK.
