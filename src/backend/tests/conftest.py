"""Shared fixtures.

These tests run against the real fine-tuned model and the real FastAPI app.
The model is the system under test, so it is never mocked — a suite that
stubbed out inference would keep passing if the artifact went missing or the
label ordering drifted, which are exactly the failures worth catching.

The database is not required: app/main.py degrades when Postgres is
unreachable, and these tests exercise that path deliberately.
"""
import os
import sys
import uuid
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.ml_path import ML_DIR  # noqa: E402

MODEL_DIR = (ML_DIR / "artifacts" / "model-v1") if ML_DIR else None

requires_model = pytest.mark.skipif(
    MODEL_DIR is None or not (MODEL_DIR / "config.json").is_file(),
    reason="no trained artifact at ml/artifacts/model-v1 (train first — see ml/train_roberta.py)",
)


@pytest.fixture
def unique_text():
    """A short string that has never been explained before.

    SHAP results are cached in Redis keyed on the text, so a test that reuses
    a fixed string gets a replay of an earlier run's answer and would keep
    passing even after the code that produced it regressed.
    """
    return f"and it broke immediately {uuid.uuid4().hex}"


@pytest.fixture(scope="session")
def model_service():
    """The real EmotionModelService, loaded once for the whole session."""
    from app.services.emotion_model import EmotionModelService

    return EmotionModelService(MODEL_DIR, "model-v1", max_length=128)


@pytest.fixture(scope="session")
def client():
    """TestClient over the real app, including its lifespan.

    Point MODEL_DIR at the artifact so startup loads it regardless of the cwd
    pytest was invoked from.
    """
    from fastapi.testclient import TestClient

    os.environ["MODEL_DIR"] = str(MODEL_DIR)
    get_settings = __import__("app.core.config", fromlist=["get_settings"]).get_settings
    get_settings.cache_clear()

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
