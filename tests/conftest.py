import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import get_settings  # noqa: E402


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    """Offline settings for a small dataset: parquet backend, hash embeddings, no reranker, no LLM."""
    tmp = tmp_path_factory.mktemp("medrag")
    mp = pytest.MonkeyPatch()
    for k, v in {"DATA_BACKEND": "parquet", "DATA_DIR": str(tmp / "data"), "INDEX_DIR": str(tmp / "idx"),
                 "REPORTS_DIR": str(tmp / "reports"), "EMBED_PROVIDER": "hash", "EMBED_MODEL": "bow", "RERANK_MODEL": "none",
                 "USERS_FILE": "", "LLM_PROVIDER": "ollama"}.items():
        mp.setenv(k, v)
    get_settings.cache_clear()
    yield get_settings()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def built(env):
    from medrag.data import store
    from medrag.data.clean import clean
    from medrag.data.generate import generate
    from medrag.index.store import build
    from medrag.ingest.documents import build_chunks
    from medrag.providers.embeddings import get_embedder

    raw = generate(n_patients=400, mean_studies=4, dirty_rate=0.05, seed=3)
    store.write_all("raw", raw)
    cleaned, log = clean(store.read_all("raw"))
    store.write_all("clean", cleaned)
    chunks = build_chunks(store.read_all("clean"))
    manifest = build(chunks, get_embedder(), env.index_path())
    return raw, cleaned, log, chunks, manifest
