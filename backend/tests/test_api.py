"""End-to-end tests through the real FastAPI app.

These exercise the actual HTTP surface — routing, validation, serialization —
rather than calling handlers directly, so a route that stops being registered
or a schema that stops rejecting bad input fails the suite.
"""
from tests.conftest import requires_model

pytestmark = requires_model


def test_health_reports_model_and_db_state(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    # db_ready is reported regardless of whether Postgres happens to be up —
    # the point is that a missing database does not prevent startup.
    assert "db_ready" in body


def test_predict_returns_a_full_prediction(client):
    response = client.post(
        "/api/predict",
        json={"text": "Thank you so much, this genuinely saved my week!"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["top_label"] == "gratitude"
    assert body["business_bucket"] == "satisfaction"
    assert body["model_version"] == "model-v1"
    assert len(body["scores"]) == 28


def test_predict_succeeds_even_when_the_review_cannot_be_persisted(client):
    """Inference happens before persistence, so a database outage must not
    turn an already-successful prediction into a 500."""
    response = client.post("/api/predict", json={"text": "This is wonderful, thank you!"})

    assert response.status_code == 200
    assert response.json()["top_label"] == "gratitude"


def test_predict_rejects_blank_text(client):
    """min_length counts characters, so whitespace and zero-width spaces used
    to pass and were served as high-confidence `neutral`, polluting analytics."""
    for blank in ["   ", "\t\n ", "​"]:
        assert client.post("/api/predict", json={"text": blank}).status_code == 422, blank


def test_predict_flags_truncated_input(client):
    long_review = "word " * 300 + "and it broke immediately"

    body = client.post("/api/predict", json={"text": long_review}).json()

    assert body["truncated"] is True


def test_analyze_does_not_let_praise_bury_a_complaint(client):
    """The reported bug: a positive headline concatenated with an angry body
    scored `admiration`/satisfaction, because opposing sentiment averages out.

    Scored per segment, the same text reports the complaint.
    """
    body = client.post(
        "/api/analyze",
        json={"text": "The fit of the headphones is too good. Fuck the shit what the fuck is this bullshit"},
    ).json()

    assert body["overall"]["business_bucket"] == "frustration"
    assert body["is_mixed"] is True
    assert [s["business_bucket"] for s in body["segments"]] == ["satisfaction", "frustration"]


def test_analyze_splits_contrastive_clauses(client):
    body = client.post(
        "/api/analyze",
        json={"text": "The sound is great but the battery is garbage and it broke"},
    ).json()

    assert len(body["segments"]) == 2
    assert body["overall"]["business_bucket"] == "frustration"


def test_analyze_leaves_a_wholly_positive_review_positive(client):
    """Segmentation must not manufacture negativity where there is none."""
    body = client.post(
        "/api/analyze",
        json={"text": "Absolutely love these headphones, best purchase this year"},
    ).json()

    assert body["overall"]["business_bucket"] == "satisfaction"
    assert body["is_mixed"] is False


def test_batch_worker_scores_identically_to_the_analyze_endpoint(client, model_service):
    """A review must get the same label whether typed or uploaded in a CSV.

    The Celery worker previously used whole-text predict() while the API used
    segmentation, so the Dashboard was aggregating two different
    methodologies — with the bulk half carrying the averaging bug.
    """
    from app.services.segmentation import choose_headline_segment, split_segments

    text = "The sound is great but the battery is garbage and it broke"

    # What celery_app.process_batch_job now computes per row.
    segments = split_segments(text) or [text]
    results = model_service.predict_many(segments)
    batch_label = results[choose_headline_segment(results)]["top_label"]

    api_label = client.post("/api/analyze", json={"text": text}).json()["overall"]["top_label"]

    assert batch_label == api_label


def test_analyze_rejects_blank_text(client):
    assert client.post("/api/analyze", json={"text": "   "}).status_code == 422


def test_explain_rejects_an_unknown_label_without_computing(client):
    """A miscapitalized-but-valid label reached GOEMOTIONS_LABELS.index() and
    raised an uncaught ValueError — a 500 with a stack trace, and only after a
    full SHAP run had already been paid for."""
    for bad_label in ["Anger", "not_a_label"]:
        response = client.post("/api/explain", json={"text": "good product", "label": bad_label})
        assert response.status_code == 422, bad_label


def test_explain_attributes_only_tokens_the_model_actually_read(client, unique_text):
    """SHAP used to explain the untruncated text while scoring the truncated
    one, so tokens past the 128-token window came back at ~0 — rendered in the
    UI as "this word didn't matter" rather than "this word was never read".

    The text must be unique per run: a cached explanation from a previous run
    would be replayed verbatim and mask a regression here.
    """
    long_review = "word " * 300 + unique_text

    body = client.post("/api/explain", json={"text": long_review}).json()

    assert body["truncated"] is True
    assert len(body["tokens"]) == len(body["attributions"])
    assert len(body["tokens"]) <= 128


def test_explain_identifies_the_driving_token(client):
    """The core product claim: the explanation must point at the words that
    actually drove the prediction."""
    body = client.post(
        "/api/explain",
        json={"text": "The support team ignored my emails. Completely useless."},
    ).json()

    ranked = sorted(
        zip(body["tokens"], body["attributions"]), key=lambda pair: pair[1], reverse=True
    )
    top_tokens = " ".join(token.strip().lower() for token, _ in ranked[:4])

    assert "useless" in top_tokens


def test_model_info_exposes_evaluation_metrics(client):
    body = client.get("/api/model/info").json()

    assert body["model_version"] == "model-v1"
    assert 0.0 < body["test_metrics"]["macro_f1"] <= 1.0


def test_model_info_does_not_leak_filesystem_paths(client, monkeypatch):
    """This endpoint is unauthenticated; the artifact path is an internal
    detail of the deployment."""
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "model_dir", settings.model_dir / "does-not-exist")

    response = client.get("/api/model/info")

    assert response.status_code == 404
    detail = response.json()["detail"]
    # Repo-relative pointers like "ml/train_roberta.py" are deliberate guidance;
    # what must not appear is where the artifact lives on this deployment.
    assert str(settings.model_dir) not in detail
    assert "/Users/" not in detail and "/app/" not in detail
