"""All settings come from environment variables / .env (see .env.example)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    data_backend: str = "postgres"  # postgres | parquet
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "oads_db"
    db_user: str = "postgres"
    db_password: str = ""

    embed_provider: str = "sentence-transformers"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 64
    rerank_model: str = "ncbi/MedCPT-Cross-Encoder"

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    judge_provider: str = "nvidia"
    judge_model: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"

    ollama_url: str = "http://localhost:11434"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    nvidia_api_key: str = ""
    aws_region: str = "us-east-1"  # Bedrock region; credentials come from the AWS chain, not .env
    aws_profile: str = ""

    top_k: int = 8
    candidates: int = 60
    rerank_pool: int = 30  # fused candidates re-scored by the cross-encoder
    rerank_blend: str = "fuse"  # fuse = RRF of hybrid and reranker order; replace = reranker order only
    rrf_k: int = 60  # Reciprocal Rank Fusion constant: lower = top ranks count more
    dense_weight: float = 0.5  # weight of the dense ranking against BM25 (1.0) in the fusion; 0.5 chosen in PROJECT_REPORT 20.10
    query_rewrite: str = "none"  # none | rewrite | multi | hyde (LLM-based query transformation, see retrieval/rewrite.py)
    rewrite_llm: str = ""  # provider:model for query_rewrite; default: the answering LLM
    retrieval_cache_size: int = 256  # identical questions from the same access scope reuse the result; 0 = off
    answer_context: str = "parent"  # chunk | parent: what the LLM reads per source; parent chosen in PROJECT_REPORT 20.9

    # production controls (medrag/security)
    users_file: str = ""  # JSON of users with password hashes; empty = single-user local mode (admin)
    audit_log: str = ""  # JSONL audit trail; default reports/audit.jsonl
    metrics_log: str = ""  # JSONL request metrics; default reports/metrics.jsonl
    phi_cloud: str = "pseudonymize"  # pseudonymize | block | allow: identifiers in prompts sent to cloud LLMs
    llm_price_in: float = 0.0  # USD per 1M input tokens for the answering LLM, for cost monitoring (0 = local / free tier)
    llm_price_out: float = 0.0  # USD per 1M output tokens

    data_dir: Path = ROOT / "data"
    index_dir: Path = ROOT / "indexes"
    reports_dir: Path = ROOT / "reports"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"

    def index_path(self, provider: str | None = None, model: str | None = None) -> Path:
        provider = provider or self.embed_provider
        model = model or self.embed_model
        slug = f"{provider}__{model}".replace("/", "_").replace(":", "_")
        return self.index_dir / slug


@lru_cache
def get_settings() -> Settings:
    return Settings()
