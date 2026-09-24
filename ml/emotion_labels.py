"""Canonical GoEmotions label list and business-facing groupings.

Imported by data_prep.py, train_roberta.py, evaluate.py, shap_explainer.py,
and backend/app/services/emotion_model.py — this is the single source of
truth for label order. The order matches the HuggingFace go_emotions
"simplified" config exactly; changing it silently breaks trained checkpoints.
"""

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

NUM_LABELS = len(GOEMOTIONS_LABELS)
LABEL_TO_ID = {label: i for i, label in enumerate(GOEMOTIONS_LABELS)}

# Maps each of the 27 fine-grained emotions to one of 4 business-facing
# buckets. Used on the dashboard, where "62% frustration" is more actionable
# than a bar chart of 27 near-zero bars. "neutral" is intentionally excluded
# from all buckets — it is reported separately as a coverage stat.
BUSINESS_BUCKETS = {
    "satisfaction": [
        "admiration", "amusement", "approval", "caring", "excitement",
        "gratitude", "joy", "love", "optimism", "pride", "relief",
    ],
    "frustration": [
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "remorse", "sadness", "grief", "embarrassment",
    ],
    "confusion": [
        "confusion", "curiosity", "realization", "surprise", "nervousness",
    ],
    "concern": [
        "desire", "fear",
    ],
}


def label_to_bucket(label: str) -> str | None:
    for bucket, labels in BUSINESS_BUCKETS.items():
        if label in labels:
            return bucket
    return None
