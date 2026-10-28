"""Shared plumbing for the baseline / hybrid comparison experiments.

Every model in the comparison — classical ML, DL, transformer, hybrid — goes
through the same three functions here, which is what makes the comparison fair:

  * load_split()      the same GoEmotions train/validation/test rows
  * save_predictions() a model's probabilities are stored once, then scored by
                       compare.py with one metric suite and one threshold rule
  * tune_thresholds()  per-label cutoffs chosen on VALIDATION only, applied to
                       test — identical for every model

Models never report their own metrics. They emit probabilities; compare.py
turns probabilities into every number in the results tables.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from emotion_labels import GOEMOTIONS_LABELS, NUM_LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "ml-data"
PRED_DIR = REPO_ROOT / "results" / "predictions"
COMPARE_DIR = REPO_ROOT / "results" / "comparison"

THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)  # same grid as evaluate.py


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # classical-ML scripts run without torch
        pass


def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[list[str], np.ndarray]:
    """Returns (texts, multi-hot label matrix of shape [n, 28]) for one split."""
    path = data_dir / f"goemotions_{split}.jsonl"
    texts, rows = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            rows.append(row["labels"])
    Y = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    for i, labels in enumerate(rows):
        Y[i, labels] = 1.0
    return texts, Y


def tune_thresholds(probs: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-label threshold maximising F1 on the given (validation) split.

    Vectorised over the grid. A label with no positives in `probs`' split falls
    back to 0.3, matching evaluate.py's fallback.
    """
    thresholds = np.full(NUM_LABELS, 0.30)
    for j in range(NUM_LABELS):
        y = Y[:, j] > 0.5
        if y.sum() == 0:
            continue
        preds = probs[:, j][:, None] >= THRESHOLD_GRID[None, :]
        tp = (preds & y[:, None]).sum(0)
        fp = (preds & ~y[:, None]).sum(0)
        fn = y.sum() - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        thresholds[j] = THRESHOLD_GRID[int(np.argmax(f1))]
    return thresholds


def save_predictions(
    name: str,
    seed: int,
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    meta: dict,
    pred_dir: Path = PRED_DIR,
) -> Path:
    """Persist a model's probabilities plus cost metadata (params, timings)."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{name}__seed{seed}.npz"
    np.savez_compressed(
        path,
        val=val_probs.astype(np.float32),
        test=test_probs.astype(np.float32),
        meta=json.dumps({"name": name, "seed": seed, **meta}),
    )
    print(f"[saved] {path.relative_to(REPO_ROOT)}")
    return path


def load_predictions(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {"val": z["val"], "test": z["test"], "meta": json.loads(str(z["meta"]))}


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


__all__ = [
    "GOEMOTIONS_LABELS", "NUM_LABELS", "REPO_ROOT", "DATA_DIR", "PRED_DIR",
    "COMPARE_DIR", "THRESHOLD_GRID", "set_seed", "load_split",
    "tune_thresholds", "save_predictions", "load_predictions", "Timer",
]
