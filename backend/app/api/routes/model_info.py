"""GET {api_v1_prefix}/model/info — raw training/eval metrics for the active model.

Registered on the app router by app/main.py. Reads straight from the model
directory's JSON artifacts written by ml/train_roberta.py and ml/evaluate.py;
returns 404 (not a 500) when training hasn't produced them yet.
"""
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _read_metrics(path: Path) -> dict | None:
    """Reads one metrics file, or returns None if it isn't usable.

    Checking `.exists()` first and reading afterwards is a race here, not a
    guard: docker-compose bind-mounts ./ml/artifacts into the container, so a
    redeploy can delete or half-write a file between the two calls and turn
    this into a 500. Just attempt the read and treat any failure as absent.
    """
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("Could not read metrics file '%s'.", path, exc_info=True)
        return None


@router.get("/model/info")
async def model_info() -> dict:
    settings = get_settings()

    train_metrics = _read_metrics(settings.model_dir / "train_metrics.json")
    test_metrics = _read_metrics(settings.model_dir / "test_metrics.json")

    if train_metrics is None and test_metrics is None:
        # Deliberately no filesystem path here — this endpoint is unauthenticated
        # and the path is an internal detail of the deployment.
        raise HTTPException(
            status_code=404,
            detail=(
                "No model metrics found. Training and evaluation haven't been run "
                "yet — see ml/train_roberta.py and ml/evaluate.py."
            ),
        )

    return {
        "model_version": settings.model_version,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
