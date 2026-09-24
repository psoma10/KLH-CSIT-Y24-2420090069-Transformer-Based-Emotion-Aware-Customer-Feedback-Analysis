"""Tests for the model service itself, against the real checkpoint.

Every test here fails if the behavior it describes is removed — the label
ordering invariant in particular has no other guard, and ml/emotion_labels.py
warns that changing it silently breaks trained checkpoints.
"""
import json

import pytest

from tests.conftest import MODEL_DIR, requires_model

pytestmark = requires_model


def test_checkpoint_label_order_matches_the_hardcoded_list(model_service):
    """The single most dangerous silent failure in this codebase.

    predict() zips GOEMOTIONS_LABELS against the raw logits and ignores the
    checkpoint's own id2label. If a future checkpoint is trained with a
    different ordering, every score is attached to the wrong emotion and
    nothing raises — the API just returns confident, wrong answers forever.
    """
    from emotion_labels import GOEMOTIONS_LABELS

    id2label = model_service.model.config.id2label
    checkpoint_order = [id2label[i] for i in range(len(id2label))]

    assert checkpoint_order == list(GOEMOTIONS_LABELS)


def test_predict_returns_a_score_for_every_label(model_service):
    from emotion_labels import GOEMOTIONS_LABELS

    result = model_service.predict("This is a perfectly ordinary sentence.")

    assert set(result["scores"]) == set(GOEMOTIONS_LABELS)
    assert all(0.0 <= score <= 1.0 for score in result["scores"].values())


def test_predict_identifies_clearly_signposted_emotions(model_service):
    """A behavioral floor: if these flip, the artifact is wrong, not just weak.

    Deliberately unambiguous phrasings rather than borderline cases, so this
    asserts the model works at all without being brittle about its macro-F1.
    """
    cases = [
        ("Thank you so much, this genuinely saved my week!", "gratitude"),
        ("I have no idea how this works, the docs contradict themselves.", "confusion"),
    ]
    for text, expected in cases:
        assert model_service.predict(text)["top_label"] == expected, text


def test_top_label_maps_to_a_business_bucket(model_service):
    result = model_service.predict("The support team ignored three of my emails.")

    assert result["business_bucket"] in {"satisfaction", "frustration", "confusion", "concern", None}


def test_business_bucket_is_null_when_no_label_clears_its_threshold(model_service):
    """The bucket feeds Dashboard aggregates, so it must mean something.

    argmax always returns a label. "terrible awful broken" repeated scores
    disgust 0.261 against a tuned threshold of 0.50 — the model is confident
    in nothing, and counting that as frustration would bypass per-label
    threshold tuning entirely and let noise drive the business metric.
    """
    result = model_service.predict("terrible awful broken " * 40)

    assert result["predicted_labels"] == []
    assert result["business_bucket"] is None
    assert result["top_label_above_threshold"] is False
    # The best guess is still reported, for display.
    assert result["top_label"]


def test_business_bucket_comes_from_a_thresholded_label(model_service):
    result = model_service.predict("Thank you so much, this genuinely saved my week!")

    assert result["top_label_above_threshold"] is True
    assert result["business_bucket"] == "satisfaction"


def test_long_text_is_reported_as_truncated(model_service):
    """Silent truncation previously discarded the tail of long reviews.

    The tail is often where the complaint lives, and /predict persists the
    full text alongside a score derived from only its first 128 tokens, so the
    stored row looks authoritative. The flag is the only way a caller can tell.
    """
    short = model_service.predict("Short review.")
    long_text = "word " * 300 + "and the product broke immediately"

    assert short["truncated"] is False
    assert model_service.predict(long_text)["truncated"] is True


def test_concurrent_predictions_do_not_fail(model_service):
    """The fast tokenizer is not thread-safe.

    It mutates shared truncation/padding state per call and raises
    "RuntimeError: Already borrowed" when two threads enter it at once.
    FastAPI dispatches these calls into a threadpool, so before the lock was
    added 179 of 180 concurrent calls failed — the app broke under any real
    load while looking fine in single-threaded testing.
    """
    import threading

    errors = []
    successes = []
    lock = threading.Lock()

    def work():
        for _ in range(10):
            try:
                result = model_service.predict("The battery died and support ignored my emails")
                with lock:
                    successes.append(result["top_label"])
            except Exception as exc:  # noqa: BLE001 — the point is to catch anything
                with lock:
                    errors.append(repr(exc))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(successes) == 80
    # Concurrency must not change the answer either.
    assert len(set(successes)) == 1


def test_thresholds_come_from_the_artifact(model_service):
    """Tuned per-label thresholds must actually be used, not silently defaulted."""
    tuned = json.loads((MODEL_DIR / "thresholds.json").read_text())

    assert model_service.thresholds["gratitude"] == pytest.approx(tuned["gratitude"])
    # A flat default everywhere would mean the tuning was silently dropped.
    assert len(set(model_service.thresholds.values())) > 1


def test_corrupt_thresholds_do_not_discard_a_valid_checkpoint(tmp_path):
    """A 636-byte metadata file must not invalidate 500MB of weights.

    main.py treats any exception from loading as "model failed to load" and
    tells the operator to retrain — sending them to Colab to fix a checkpoint
    that was never broken.
    """
    import shutil

    from app.services.emotion_model import _DEFAULT_THRESHOLD, EmotionModelService

    staged = tmp_path / "model"
    staged.mkdir()
    for item in MODEL_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, staged / item.name)
    (staged / "thresholds.json").write_text('{"admiration": nonsense')  # truncated/invalid

    service = EmotionModelService(staged, "model-v1")

    assert service.predict("Thank you!")["top_label"] == "gratitude"
    assert set(service.thresholds.values()) == {_DEFAULT_THRESHOLD}
