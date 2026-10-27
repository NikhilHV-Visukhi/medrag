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
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    judge_provider: str = "anthropic"
    judge_model: str = "claude-opus-5-5"

    ollama_url: str = "http://localhost:11434"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    top_k: int = 8
    candidates: int = 60

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
