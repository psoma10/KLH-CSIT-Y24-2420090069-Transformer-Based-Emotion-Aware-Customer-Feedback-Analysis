"""Pydantic request/response schemas for prediction, explanation, and review endpoints.

Imported by app/api/routes/predict.py, explain.py, reviews.py, analytics.py.
"""
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

import app.core.ml_path  # noqa: F401  — puts ml/ on sys.path for the import below
from emotion_labels import GOEMOTIONS_LABELS


# str.strip() only removes characters Python considers whitespace, which does
# not include the zero-width and formatting characters below — a review of
# "​" survives .strip() and is served as a confident `neutral`.
_INVISIBLE_CHARS = "​‌‍⁠﻿­"


def _require_non_blank(value: str) -> str:
    """Rejects text that carries no visible content.

    `min_length` counts characters, so "   " and a zero-width space both pass
    it while saying nothing — the model returns a high-confidence `neutral`
    and /predict persists the row, polluting the analytics tables. Validate at
    the boundary instead.
    """
    if not value.strip().strip(_INVISIBLE_CHARS).strip():
        raise ValueError("must contain non-whitespace text")
    return value


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    _validate_text = field_validator("text")(_require_non_blank)


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    scores: dict[str, float]
    predicted_labels: list[str]
    top_label: str
    top_score: float
    # False when top_label is the argmax but cleared no tuned threshold — the
    # model's best guess rather than a confident call. business_bucket is null
    # in that case, so aggregates never count an unconfident prediction.
    top_label_above_threshold: bool = True
    business_bucket: str | None
    model_version: str
    truncated: bool = False


class SegmentResult(BaseModel):
    """One independently-scored part of a review."""

    text: str
    scores: dict[str, float]
    predicted_labels: list[str]
    top_label: str
    top_score: float
    # False when top_label is the argmax but cleared no tuned threshold — the
    # model's best guess rather than a confident call. business_bucket is null
    # in that case, so aggregates never count an unconfident prediction.
    top_label_above_threshold: bool = True
    business_bucket: str | None
    truncated: bool = False


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    # False for the live-preview call fired on every debounced keystroke;
    # true only for the visitor's actual Submit click. Gates whether this
    # request writes a Review row at all — see analyze.py.
    persist: bool = False
    star_rating: int | None = Field(default=None, ge=1, le=5)
    product_id: str | None = Field(default=None, max_length=64)

    _validate_text = field_validator("text")(_require_non_blank)


class AnalyzeResponse(BaseModel):
    """Per-segment emotion analysis plus the overall read.

    `overall` is the segment chosen to represent the review — the most
    confident negative one where present — rather than a score over the
    concatenated text, which averages opposing sentiment into mush.
    """

    model_config = ConfigDict(protected_namespaces=())

    segments: list[SegmentResult]
    overall: SegmentResult
    overall_segment_index: int
    is_mixed: bool
    model_version: str


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    label: str | None = None  # defaults to the model's top-scoring label

    _validate_text = field_validator("text")(_require_non_blank)

    @field_validator("label")
    @classmethod
    def _known_label(cls, value: str | None) -> str | None:
        # ml/shap_explainer.py does GOEMOTIONS_LABELS.index(label), which raises
        # an uncaught ValueError (HTTP 500 with a stack trace) for anything not
        # in the list — including a correctly-spelled but capitalized label.
        # Reject it here as a 422 instead, before any SHAP compute is paid for.
        if value is not None and value not in GOEMOTIONS_LABELS:
            raise ValueError(f"unknown emotion label: {value!r}")
        return value


class ExplainResponse(BaseModel):
    explained_label: str
    label_score: float
    tokens: list[str]
    attributions: list[float]
    base_value: float
    cached: bool
    truncated: bool = False


class ReviewOut(BaseModel):
    id: str
    product_id: str | None
    review_text: str
    star_rating: int | None
    review_date: date | None
    top_label: str | None = None
    business_bucket: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class BatchJobOut(BaseModel):
    id: str
    status: str
    total_rows: int
    processed_rows: int
    error: str | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class EmotionDistributionItem(BaseModel):
    label: str
    count: int


class BucketDistributionItem(BaseModel):
    bucket: str
    count: int


class TrendPoint(BaseModel):
    date: date
    bucket: str
    count: int
