"""Late fusion of a neural model with the TF-IDF ML ensemble.

    python fusion.py

For every seed that has a saved neural model, builds

    p = (1 - w) * p_neural + w * p_ml_ensemble

with the single global weight w chosen on the VALIDATION split (macro-F1 at
tuned thresholds). One global weight rather than one per label: per-label
weights would be fit on the ~10-80 validation positives of the rare labels and
mostly memorise them.

Why fuse at all. The transformer sees subword pieces through a contextual
encoder; the TF-IDF ensemble sees exact word/character n-grams through
linear/tree models. Their errors are only partly correlated, so a small ML
weight can recover rare-label cues (specific lexical markers, emoticons,
repeated characters) that fine-tuning smooths over. Whether that actually pays
off is an empirical question, answered by compare.py — not assumed here.

Outputs (per seed), each saved like any other model:
    roberta_plain      + ml_ensemble -> roberta_plain+ml     (Transformer + ML)
    roberta_bilstm_attn+ ml_ensemble -> roberta_bilstm_attn+ml  (PROPOSED)
"""
from __future__ import annotations

import re

import numpy as np

from common import PRED_DIR, load_predictions, load_split, save_predictions, tune_thresholds
from metrics import _confusion, _safe_div, apply_thresholds

ML_NAME = "ml_ensemble"
PAIRS = ["roberta_plain", "roberta_bilstm_attn", "bert_plain"]
WEIGHT_GRID = np.round(np.arange(0.0, 0.61, 0.05), 2)


def val_macro_f1(probs, Y) -> float:
    th = tune_thresholds(probs, Y)
    tp, fp, fn, _ = _confusion(Y, apply_thresholds(probs, th))
    return float(_safe_div(2 * tp, 2 * tp + fp + fn).mean())


def main() -> None:
    _, Yva = load_split("validation")
    ml_path = next(PRED_DIR.glob(f"{ML_NAME}__seed*.npz"), None)
    if ml_path is None:
        raise SystemExit("run baselines_ml.py first — no ml_ensemble predictions found")
    ml = load_predictions(ml_path)

    for base in PAIRS:
        for path in sorted(PRED_DIR.glob(f"{base}__seed*.npz")):
            seed = int(re.search(r"seed(\d+)", path.name).group(1))
            nn_ = load_predictions(path)
            scores = {float(w): val_macro_f1((1 - w) * nn_["val"] + w * ml["val"], Yva) for w in WEIGHT_GRID}
            w_best = max(scores, key=scores.get)
            print(f"{base} seed {seed}: w={w_best:.2f}  val macro-F1 {scores[w_best]:.4f} "
                  f"(w=0: {scores[0.0]:.4f})")
            save_predictions(
                f"{base}+ml", seed,
                (1 - w_best) * nn_["val"] + w_best * ml["val"],
                (1 - w_best) * nn_["test"] + w_best * ml["test"],
                {"family": "Transformer + ML fusion" if base != "roberta_bilstm_attn" else "Proposed hybrid",
                 "fusion_weight_ml": w_best, "neural_component": base, "ml_component": ML_NAME,
                 "n_params": nn_["meta"].get("n_params"),
                 "train_seconds": nn_["meta"].get("train_seconds", 0) + ml["meta"].get("train_seconds", 0),
                 "infer_ms_per_1k": nn_["meta"].get("infer_ms_per_1k")},
            )


if __name__ == "__main__":
    main()
