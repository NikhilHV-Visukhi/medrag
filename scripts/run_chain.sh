#!/bin/sh
set -e
# A -> B -> C one after another on the GPU (RTX 4050, 6 GB); jobs would compete for VRAM if run together.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export EMBED_BATCH_SIZE=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python
$PY -m medrag bench embed --models medcpt:ncbi/MedCPT,sentence-transformers:BAAI/bge-base-en-v1.5,sentence-transformers:intfloat/e5-base-v2,sentence-transformers:NeuML/pubmedbert-base-embeddings,ollama:nomic-embed-text > reports/bench_embed_gpu.log 2>&1
echo "embed done $?"
$PY -m medrag bench chunk > reports/bench_chunk_gpu.log 2>&1
echo "chunk done $?"
$PY -m medrag bench tune > reports/bench_tune_gpu.log 2>&1
echo "tune done $?"
