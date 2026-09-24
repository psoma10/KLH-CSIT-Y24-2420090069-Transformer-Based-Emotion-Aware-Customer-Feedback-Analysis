"""App-wide settings, loaded from environment variables (.env in dev).

Imported by app/main.py (lifespan), app/core/db.py, app/services/emotion_model.py,
app/services/cache.py, and app/services/celery_app.py — every module that
needs a path, URL, or credential reads it from here rather than os.environ
directly, so there is exactly one place that defines what config exists.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", protected_namespaces=()
    )

    # Model
    model_dir: Path = Path("../../results/ml-artifacts/model-v1")
    model_max_length: int = 128
    shap_max_evals: int = 200
    shap_cache_ttl_seconds: int = 60 * 60 * 24 * 7  # SHAP output is deterministic per (text, model version)

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/emotion_feedback"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Upload limits. Every CSV row costs a model inference in the Celery
    # worker, so an unbounded upload is both a memory and a CPU problem.
    max_upload_bytes: int = 5 * 1024 * 1024
    max_batch_rows: int = 5000
    max_review_chars: int = 5000  # matches PredictRequest/AnalyzeRequest

    # CORS
    frontend_origin: str = "http://localhost:5173"

    # App
    model_version: str = "model-v1"
    api_v1_prefix: str = "/api"


@lru_cache
def get_settings() -> Settings:
    return Settings()
