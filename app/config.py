"""
Application configuration.

All secrets and tuneable settings are read from environment variables so
the service can be deployed to any environment without code changes.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from project root (parent of app/) so it works regardless of cwd
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Typed settings backed by environment variables (and an optional .env file)."""

    model_config = SettingsConfigDict(
        env_file=_ENV_PATH if _ENV_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── RapidAPI credentials ──────────────────────────────────────────────────
    rapidapi_key: str = ""
    rapidapi_host: str = ""

    # ── RapidAPI wine-recognition2 endpoint ───────────────────────────────────
    # The base URL is fixed by api4ai; only the host header varies per tenant.
    rapidapi_base_url: str = "https://wine-recognition2.p.rapidapi.com"
    rapidapi_results_path: str = "/v1/results"
    rapidapi_version_path: str = "/v1/version"

    # ── HTTP client tunables ──────────────────────────────────────────────────
    http_timeout_seconds: float = 10.0   # per-request timeout
    http_max_retries: int = 1            # number of retries on transient failures

    # ── File upload constraints ───────────────────────────────────────────────
    max_file_size_bytes: int = 10 * 1024 * 1024          # 10 MB
    allowed_content_types: frozenset[str] = frozenset(
        {"image/jpeg", "image/png", "image/webp"}
    )
    allowed_extensions: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp"}
    )

    # ── Response defaults ─────────────────────────────────────────────────────
    default_top_k: int = 5
    max_top_k: int = 10

    # ── Persist results ───────────────────────────────────────────────────────
    # Directory to save each recognition response as JSON (e.g. "results").
    # Relative to project root; set to "" to disable.
    results_dir: str = "results"

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins, e.g. "http://localhost:3000,https://myapp.com"
    frontend_origin: str = "*"

    @property
    def allowed_origins(self) -> list[str]:
        """Return CORS origins as a list."""
        return [o.strip() for o in self.frontend_origin.split(",") if o.strip()]

    @property
    def rapidapi_configured(self) -> bool:
        """Return True only when both RapidAPI secrets are present."""
        return bool(self.rapidapi_key and self.rapidapi_host)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton so the .env file is parsed once per process."""
    return Settings()
