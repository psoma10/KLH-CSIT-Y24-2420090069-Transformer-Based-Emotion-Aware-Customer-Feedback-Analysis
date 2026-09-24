"""SHAP token-attribution explainer for the fine-tuned emotion model.

Imported by backend/app/services/shap_explainer.py (thin wrapper that adds
caching) and by ml/notebooks/ for offline sanity-checking explanations before
they're trusted in the product. Kept dependency-light (no FastAPI/Redis
imports here) so it also runs standalone in a notebook.
"""
from pathlib import Path

import numpy as np
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotion_labels import GOEMOTIONS_LABELS


class EmotionExplainer:
    def __init__(
        self,
        tokenizer,
        model,
        device: str | None = None,
        max_evals: int = 200,
        max_length: int = 128,
    ):
        """Takes an already-loaded tokenizer and model.

        Loading its own copy from disk would put a second set of ~476MB
        weights in the process for no benefit — SHAP only ever runs the same
        forward pass the classifier does. Callers that have no model to hand
        (notebooks, scripts) should use `from_pretrained` below.
        """
        self.device = device or next(model.parameters()).device.type
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.max_evals = max_evals
        self.max_length = max_length

        self._masker = shap.maskers.Text(self.tokenizer)
        self._explainer = shap.Explainer(self._predict_proba, self._masker, output_names=GOEMOTIONS_LABELS)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Path,
        device: str | None = None,
        max_evals: int = 200,
        max_length: int = 128,
    ) -> "EmotionExplainer":
        """Loads its own copy of the weights from disk.

        For standalone use (notebooks, scripts) where there is no already-loaded
        model to share. The API passes in the classifier's instance instead.
        """
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(resolved)
        return cls(tokenizer, model, device=resolved, max_evals=max_evals, max_length=max_length)

    @torch.no_grad()
    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            list(texts), truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        return torch.sigmoid(logits).cpu().numpy()

    def _truncate_to_window(self, text: str) -> str:
        """Cuts text down to the token window the model actually reads.

        Without this, SHAP tokenizes the full text and returns an attribution
        for every token, while _predict_proba only ever sees the first
        max_length of them. Perturbing a token past the cutoff cannot change
        the output, so those positions come back at ~0 — indistinguishable in
        the UI from "this word genuinely didn't matter". Explaining only the
        text the model saw keeps every reported attribution meaningful.
        """
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        # -2 leaves room for the <s>/</s> special tokens added during scoring.
        budget = max(self.max_length - 2, 1)
        if len(ids) <= budget:
            return text
        return self.tokenizer.decode(ids[:budget], skip_special_tokens=True)

    def explain(self, text: str, top_label: str | None = None) -> dict:
        """Returns per-token SHAP values for one text.

        If top_label is None, explains the model's highest-scoring emotion.

        SHAP's Partition explainer needs at least 2*num_tokens+1 evaluations and
        silently raises max_evals to meet that floor, so max_evals alone does
        not bound latency — the token count does. Truncating to the model's own
        window is what actually keeps this bounded, and it is required for
        correctness anyway (see _truncate_to_window).
        """
        explained_text = self._truncate_to_window(text)

        probs = self._predict_proba([explained_text])[0]
        label = top_label or GOEMOTIONS_LABELS[int(np.argmax(probs))]
        if label not in GOEMOTIONS_LABELS:
            raise ValueError(f"unknown emotion label: {label!r}")
        label_idx = GOEMOTIONS_LABELS.index(label)

        shap_values = self._explainer([explained_text], max_evals=self.max_evals)

        tokens = shap_values.data[0]
        values = shap_values.values[0][:, label_idx]

        return {
            "explained_label": label,
            "label_score": float(probs[label_idx]),
            "tokens": [str(t) for t in tokens],
            "attributions": [float(v) for v in values],
            "base_value": float(shap_values.base_values[0][label_idx]),
            "truncated": explained_text != text,
        }
