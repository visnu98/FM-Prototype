"""Application configuration.

All settings are loaded from environment variables (via a `.env` file) using
Pydantic Settings. Credentials are NEVER hard-coded and NEVER printed.

Usage:
    from app.core.config import get_settings
    settings = get_settings()
    engine_url = settings.sqlalchemy_url()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the `prototype/` directory.
# This file is at app/core/config.py, so go up three levels: core -> app -> prototype.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Typed application settings, validated at load time.

    The database password is held as a :class:`~pydantic.SecretStr` so it does
    not leak into logs or `repr()` output.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate later-phase variables in the same .env
    )

    # ── Database connection ──────────────────────────────────────────────────
    db_host: str = Field(..., description="PostgreSQL host")
    db_port: int = Field(5432, description="PostgreSQL port")
    db_name: str = Field(..., description="Database name")
    db_user: str = Field(..., description="Database user (read-only preferred)")
    db_password: SecretStr = Field(..., description="Database password (kept secret)")
    db_ssl_mode: str = Field("prefer", description="libpq sslmode")

    db_connect_timeout: int = Field(10, description="Connection timeout (seconds)")
    db_statement_timeout: int = Field(30, description="Per-statement timeout (seconds)")

    # ── Optional domain scoping (used by later phases) ───────────────────────
    default_facility_id: int | None = None
    default_project_id: int | None = None

    # ── LLM configuration ────────────────────────────────────────────────────
    llm_provider: str = "groq"
    groq_api_key: SecretStr | None = None
    # The two models compared in the evaluation (A then B).
    model_a: str = "qwen/qwen3-32b"
    model_b: str = "llama-3.1-8b-instant"
    # Note: decoding temperature is hard-coded to 0.0 in app/llm/groq_client.py
    # (deterministic decoding) and is intentionally NOT configurable.

    def models(self) -> list[str]:
        """The two models compared in the evaluation (A then B)."""
        return [self.model_a, self.model_b]

    # ── Derived helpers ──────────────────────────────────────────────────────
    def sqlalchemy_url(self) -> str:
        """Build a SQLAlchemy URL using the psycopg (v3) driver.

        The password is rendered only here, at connection-build time, and never
        logged. ``sslmode`` is passed via query parameters.
        """
        pwd = self.db_password.get_secret_value()
        return (
            "postgresql+psycopg://"
            f"{self.db_user}:{pwd}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?sslmode={self.db_ssl_mode}"
        )

    def safe_summary(self) -> dict[str, str | int | None]:
        """Connection details suitable for logging — password is redacted."""
        return {
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": "***redacted***",
            "db_ssl_mode": self.db_ssl_mode,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (loaded once per process)."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
