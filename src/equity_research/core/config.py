"""Application settings.

Secrets are read only from environment variables (optionally via a local
`.env` file, never committed). Settings are intentionally permissive at
construction time — e.g. a missing `GROQ_API_KEY` does not raise here — so
that unit tests can build a `Settings()` without secrets configured. Each
consumer that actually needs a secret (the SEC client, the Groq LLM
adapter, ...) is responsible for failing loudly when it is missing and not
running in test mode.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets (env-only; see .env.example) ---
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_environment: str | None = Field(default=None, alias="PINECONE_ENVIRONMENT")
    sec_user_agent: str | None = Field(default=None, alias="SEC_USER_AGENT")

    # --- Non-secret configuration ---
    groq_model: str = Field(default="llama-3.1-70b-versatile", alias="GROQ_MODEL")
    db_path: Path = Field(
        default=Path("./data/equity_research.db"), alias="EQUITY_RESEARCH_DB_PATH"
    )
    http_cache_dir: Path = Field(
        default=Path("./data/http_cache"), alias="EQUITY_RESEARCH_HTTP_CACHE_DIR"
    )

    # --- Bounds and budgets (design spec: per-node timeout/retry/source caps) ---
    request_timeout_seconds: float = 15.0
    max_retries: int = 3
    max_sources_per_node: int = 8
    run_budget_seconds: float = 300.0

    test_mode: bool = Field(default=False, alias="EQUITY_RESEARCH_TEST_MODE")

    @property
    def has_groq_credentials(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_pinecone_credentials(self) -> bool:
        return bool(self.pinecone_api_key)
