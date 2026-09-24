"""Fine-tunes roberta-base for multi-label emotion classification on GoEmotions.

Run on Colab (T4/A100) or any CUDA box:
    python train_roberta.py --config configs/train_config.yaml

Requires data/goemotions_{train,validation,test}.jsonl — run data_prep.py
--goemotions first. Writes the fine-tuned model, tokenizer, and per-label
thresholds to artifacts/model-v1/, which backend/app/services/emotion_model.py
loads at FastAPI startup.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from emotion_labels import GOEMOTIONS_LABELS, NUM_LABELS


class GoEmotionsDataset(Dataset):
    """Rows from one or more JSONL corpora in the GoEmotions label space.

    Accepts several paths so auxiliary corpora train alongside GoEmotions in a
    single pass. Each row may carry a `mask` of the label indices its source
    annotates; rows without one (the pre-aux format) are treated as fully
    annotated, which is what GoEmotions always was.
    """

    def __init__(self, paths: Path | list[Path], tokenizer, max_length: int):
        if isinstance(paths, Path):
            paths = [paths]
        self.rows = [json.loads(line) for p in paths for line in p.open()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # Multi-hot label vector for BCEWithLogitsLoss — NOT a class index.
        target = torch.zeros(NUM_LABELS, dtype=torch.float32)
        target[row["labels"]] = 1.0
        # 1.0 where this row's source annotates the label. Auxiliary corpora
        # cover a subset, and an unannotated label is unknown rather than
        # negative — see aux_datasets.py.
        mask = torch.zeros(NUM_LABELS, dtype=torch.float32)
        mask[row.get("mask", list(range(NUM_LABELS)))] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": target,
            "label_mask": mask,
        }


def compute_metrics(eval_pred) -> dict:
    """Macro/micro F1 at a fixed 0.5 cutoff, for tracking training progress only.

    Final reported metrics use per-label tuned thresholds — see evaluate.py.
    This function exists solely so the Trainer can pick a best checkpoint.
    """
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
    }


def _compute_pos_weight(dataset, cap: float) -> torch.Tensor:
    """Per-label BCE positive-class weight from training-set label frequency.

    GoEmotions is severely imbalanced: `grief` has single-digit support against
    ~1800 for `neutral`. Under plain BCE the loss is dominated by the common
    labels, and predicting "absent" for a rare label is nearly free — which is
    how three labels ended up at 0.000 F1. Weighting each label's positive term
    by neg/pos raises the cost of ignoring them.

    The cap matters: an uncapped weight for a label with a handful of examples
    reaches the hundreds and destabilizes training for every other label.
    """
    labels = torch.stack([torch.as_tensor(dataset[i]["labels"]) for i in range(len(dataset))])
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives

    # clamp(min=1) keeps a label with zero positives from producing infinity.
    weight = negatives / positives.clamp(min=1)
    return weight.clamp(max=cap).float()


def _make_masked_trainer(pos_weight: torch.Tensor | None):
    """Subclasses Trainer for per-label masking, and optional pos_weight.

    Two reasons the loss cannot be left to the model. `problem_type=
    "multi_label_classification"` computes an unweighted BCEWithLogitsLoss
    internally with no way to pass a weight through. And it has no concept of
    an unannotated label: it treats every zero as a negative, which is wrong
    for the auxiliary corpora, where a zero usually means the annotators were
    never asked about that label.

    So BCE is computed per element with `reduction="none"`, multiplied by the
    row's mask, and averaged over annotated positions only. Dividing by the
    number of unmasked positions rather than by the full 28 keeps the loss
    scale comparable between a GoEmotions row (28 positions) and a dair-ai row
    (6), so mixed batches do not quietly down-weight the auxiliary data.
    """

    class MaskedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            mask = inputs.pop("label_mask", None)
            outputs = model(**inputs)
            logits = outputs.logits

            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=None if pos_weight is None else pos_weight.to(logits.device),
                reduction="none",
            )
            per_element = loss_fn(logits, labels.float())

            if mask is None:
                loss = per_element.mean()
            else:
                mask = mask.to(logits.device)
                # clamp(min=1) guards against a row that somehow masks
                # everything, which would otherwise divide by zero.
                loss = (per_element * mask).sum() / mask.sum().clamp(min=1.0)

            return (loss, outputs) if return_outputs else loss

    return MaskedTrainer


def main(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.open())
    data_dir = Path("data")
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["training"]["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",  # forces BCEWithLogitsLoss internally
        hidden_dropout_prob=cfg["model"]["dropout"],
        id2label={i: l for i, l in enumerate(GOEMOTIONS_LABELS)},
        label2id={l: i for i, l in enumerate(GOEMOTIONS_LABELS)},
    )

    # Auxiliary corpora join the TRAIN split only. Validation and test stay
    # pure GoEmotions so every metric in MODEL_CARD.md remains comparable
    # across runs and is measured on fully-annotated data — mixing partially
    # annotated rows into eval would make macro F1 mean something different
    # from one config to the next.
    aux_keys = cfg["data"].get("aux_datasets") or []
    train_paths = [data_dir / "goemotions_train.jsonl"]
    for key in aux_keys:
        path = data_dir / f"{key}_train.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing — run `python data_prep.py --aux {key} --out data/` first."
            )
        train_paths.append(path)
    if aux_keys:
        print(f"Training on GoEmotions + {len(aux_keys)} auxiliary corpora: {aux_keys}")

    train_ds = GoEmotionsDataset(train_paths, tokenizer, cfg["model"]["max_length"])
    val_ds = GoEmotionsDataset(
        [data_dir / "goemotions_validation.jsonl"], tokenizer, cfg["model"]["max_length"]
    )

    fp16 = cfg["training"]["fp16"] and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["eval_batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg["output"]["save_best_metric"],
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=cfg["training"]["seed"],
    )

    pos_weight = None
    if cfg["training"].get("use_pos_weight", False):
        pos_weight = _compute_pos_weight(
            train_ds, cap=float(cfg["training"].get("pos_weight_cap", 20.0))
        )
        print(
            f"Using pos_weight (capped at {cfg['training'].get('pos_weight_cap', 20.0)}): "
            f"min={pos_weight.min():.2f} max={pos_weight.max():.2f}"
        )

    # Always the masked trainer: every row now carries a label_mask, and the
    # stock Trainer would both ignore it and choke on the extra batch key.
    trainer_cls = _make_masked_trainer(pos_weight)

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["training"]["early_stopping_patience"])],
    )

    trainer.train()

    final_dir = output_dir  # final weights live directly under artifacts/model-v1/
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics = trainer.evaluate()
    (final_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {final_dir}")
    print(f"Validation metrics: {metrics}")
    print("Next: python evaluate.py --config configs/train_config.yaml  (tunes per-label thresholds)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    args = parser.parse_args()
    main(args.config)
