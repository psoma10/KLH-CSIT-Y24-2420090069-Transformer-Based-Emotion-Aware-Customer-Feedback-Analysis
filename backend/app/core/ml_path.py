"""Makes the repo's `ml/` package importable from the backend.

`ml/` is a sibling of `backend/`, not an installed package, so modules like
`emotion_labels` are not on sys.path by default. Both the API and the Celery
worker need it, and so do the Pydantic schemas (for the label list) — which
must not pull in torch just to validate a string. Keeping the lookup here
means there is exactly one definition of where `ml/` lives.

The directory sits at a different depth depending on the layout: a host
checkout has this file at <repo>/backend/app/core/, while the Docker image
flattens it to /app/app/core/ with ml/ at /app/ml. Walk up looking for the
directory rather than hardcoding a parent index.
"""
import sys
from pathlib import Path


def find_ml_dir() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "ml"
        if (candidate / "emotion_labels.py").is_file():
            return candidate
    return None


ML_DIR = find_ml_dir()

if ML_DIR is not None and str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
