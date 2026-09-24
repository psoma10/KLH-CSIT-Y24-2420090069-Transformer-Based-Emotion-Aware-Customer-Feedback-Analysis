"""Loads the fine-tuned RoBERTa emotion classifier once and serves predictions.

Instantiated in app/main.py's lifespan and stored on app.state.emotion_model;
app/api/routes/predict.py reads it from there rather than constructing its own
copy, so the (~500MB) weights are loaded exactly once per process.
"""
import json
import logging
import threading
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# app/core/ml_path puts the repo's ml/ directory on sys.path so
# `from emotion_labels import ...` resolves the same module ml/train_roberta.py
# and ml/shap_explainer.py use. Importing it is the whole point — see that
# module for why the location is searched for rather than hardcoded.
import app.core.ml_path  # noqa: F401

from emotion_labels import GOEMOTIONS_LABELS, label_to_bucket  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.30

# Web serving gets its throughput from handling requests concurrently, not from
# parallelizing one small matmul. Left at torch's default, every one of the up
# to 40 threads in anyio's pool would fan out across all cores for a 128-token
# forward pass, and they would spend more time contending than computing.
torch.set_num_threads(1)


class EmotionModelService:
    """Holds the tokenizer, model, and per-label thresholds for inference."""

    def __init__(self, model_dir: Path, model_version: str, max_length: int = 128):
        self.model_dir = Path(model_dir)
        self.model_version = model_version
        self.max_length = max_length

        if not self.model_dir.exists():
            raise RuntimeError(
                f"Model directory '{self.model_dir}' does not exist. Training happens "
                "separately (see ml/train_roberta.py, run on Colab) — copy the resulting "
                "artifacts/model-v1/ directory here before starting the API."
            )

        self.device = "cpu"  # demo app; no assumption of GPU availability
        # HuggingFace's Rust-backed fast tokenizer mutates shared truncation and
        # padding state per call, and raises "RuntimeError: Already borrowed"
        # when two threads enter it at once. FastAPI runs these calls in a
        # threadpool, so without this lock concurrent requests fail: measured
        # 179 failures out of 180 calls across 12 threads. The lock covers only
        # tokenization — the forward pass itself is thread-safe under no_grad.
        self._tokenizer_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> dict[str, float]:
        """Per-label decision thresholds, falling back to a flat default.

        Tuned thresholds are an optimization on top of the weights, not a
        requirement for them: a corrupt or partially-written 636-byte metadata
        file must not discard an otherwise-valid 500MB checkpoint. The caller
        (app/main.py) treats any exception here as "model failed to load" and
        reports "train the model first", which would send an operator off to
        retrain something that is perfectly fine — so degrade instead.
        """
        defaults = {label: _DEFAULT_THRESHOLD for label in GOEMOTIONS_LABELS}

        thresholds_path = self.model_dir / "thresholds.json"
        if not thresholds_path.exists():
            return defaults

        try:
            raw = json.loads(thresholds_path.read_text())
        except (OSError, ValueError):
            logger.warning(
                "Could not read '%s'; falling back to a flat %.2f threshold for every label.",
                thresholds_path,
                _DEFAULT_THRESHOLD,
                exc_info=True,
            )
            return defaults

        if not isinstance(raw, dict):
            logger.warning("'%s' is not a JSON object; using default thresholds.", thresholds_path)
            return defaults

        thresholds = {}
        for label in GOEMOTIONS_LABELS:
            try:
                thresholds[label] = float(raw[label])
            except (KeyError, TypeError, ValueError):
                thresholds[label] = _DEFAULT_THRESHOLD
        return thresholds

    def _to_result(self, probs: list[float], truncated: bool) -> dict:
        scores = {label: float(p) for label, p in zip(GOEMOTIONS_LABELS, probs)}
        predicted_labels = [label for label, score in scores.items() if score >= self.thresholds[label]]

        top_label = max(scores, key=scores.get)
        above_threshold = top_label in predicted_labels

        # The business bucket drives the Dashboard's aggregate counts, so it
        # must reflect a label the model was actually confident about. argmax
        # always returns something: "terrible awful broken" repeated scores
        # disgust 0.261 against a tuned threshold of 0.50, clearing nothing at
        # all — yet bucketing it as frustration would let per-label threshold
        # tuning be bypassed entirely and count noise as signal. Where nothing
        # clears its threshold the honest answer is no bucket.
        bucket_label = (
            max(predicted_labels, key=lambda label: scores[label]) if predicted_labels else None
        )

        return {
            "scores": scores,
            "predicted_labels": predicted_labels,
            # Kept as the raw argmax: the UI shows it as the model's best guess
            # even when nothing is confident, which is informative as long as
            # aggregates don't treat it as a decision.
            "top_label": top_label,
            "top_score": scores[top_label],
            "top_label_above_threshold": above_threshold,
            "business_bucket": label_to_bucket(bucket_label) if bucket_label else None,
            "truncated": truncated,
        }

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """Runs sigmoid multi-label inference for a single piece of text."""
        return self.predict_many([text])[0]

    @torch.no_grad()
    def predict_many(self, texts: list[str]) -> list[dict]:
        """Scores several texts in one forward pass.

        Segmenting a review means classifying each part separately, which would
        otherwise cost one HTTP round trip and one model call per segment on
        every keystroke. Batching keeps that to a single pass, so analyzing a
        five-sentence review costs roughly what analyzing one sentence did.
        """
        if not texts:
            return []

        with self._tokenizer_lock:
            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )

        # Requests may carry far more text than the model reads (the schema
        # allows 5000 characters against a 128-token window), and the dropped
        # tail is often where a complaint lives. The caller cannot infer this
        # from the response, so report it explicitly rather than returning a
        # confident score derived from a fraction of the input. Measure per
        # text: padding makes the batch's shared width meaningless here.
        with self._tokenizer_lock:
            truncated_flags = [
                len(self.tokenizer(text, truncation=False)["input_ids"]) > self.max_length
                for text in texts
            ]

        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.sigmoid(logits).cpu().tolist()

        return [self._to_result(row, flag) for row, flag in zip(probs, truncated_flags)]
