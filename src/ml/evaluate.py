"""Tunes per-label decision thresholds and reports final test-set metrics.

Run after train_roberta.py:
    python evaluate.py --config configs/train_config.yaml

Thresholds are tuned on the validation split (never test — that would leak),
then applied once to the held-out test split for the final report. Writes
thresholds.json into the model directory; backend/app/services/emotion_model.py
loads it alongside the weights to convert sigmoid scores into label calls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotion_labels import GOEMOTIONS_LABELS, NUM_LABELS
from train_roberta import GoEmotionsDataset


@torch.no_grad()
def get_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def tune_thresholds(probs: np.ndarray, labels: np.ndarray, cfg: dict) -> dict:
    """Per-label threshold sweep maximizing F1 independently for each of the 28 labels."""
    candidates = np.linspace(cfg["search_min"], cfg["search_max"], cfg["search_steps"])
    thresholds = {}
    for i, label in enumerate(GOEMOTIONS_LABELS):
        best_f1, best_t = -1.0, cfg["fallback"]
        for t in candidates:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[label] = best_t
    return thresholds


def apply_thresholds(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    cutoffs = np.array([thresholds[l] for l in GOEMOTIONS_LABELS])
    return (probs >= cutoffs[None, :]).astype(int)


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.open())
    model_dir = Path(cfg["output"]["dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    val_ds = GoEmotionsDataset(Path("data/goemotions_validation.jsonl"), tokenizer, cfg["model"]["max_length"])
    test_ds = GoEmotionsDataset(Path("data/goemotions_test.jsonl"), tokenizer, cfg["model"]["max_length"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["eval_batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["eval_batch_size"])

    print("Scoring validation split...")
    val_probs, val_labels = get_probs(model, val_loader, device)

    print("Tuning per-label thresholds on validation split...")
    thresholds = tune_thresholds(val_probs, val_labels, cfg["thresholds"])
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    print("Scoring held-out test split with tuned thresholds...")
    test_probs, test_labels = get_probs(model, test_loader, device)
    test_preds = apply_thresholds(test_probs, thresholds)

    report = classification_report(
        test_labels, test_preds, target_names=GOEMOTIONS_LABELS, zero_division=0, output_dict=True
    )
    macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)

    summary = {"macro_f1": macro_f1, "micro_f1": micro_f1, "per_label": report}
    (model_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\nTest macro-F1: {macro_f1:.4f}  micro-F1: {micro_f1:.4f}")
    print(f"Full report written to {model_dir / 'test_metrics.json'}")
    print(f"Thresholds written to {model_dir / 'thresholds.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    args = parser.parse_args()
    main(args.config)
