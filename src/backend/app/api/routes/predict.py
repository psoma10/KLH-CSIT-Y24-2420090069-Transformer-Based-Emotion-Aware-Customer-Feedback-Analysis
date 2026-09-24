"""POST {api_v1_prefix}/predict — classify a single review and log it.

Registered on the app router by app/main.py. Unlike /explain, this endpoint
persists a Review + EmotionPrediction row so the Analyze page's usage feeds
straight into the Dashboard's analytics queries.
"""
import logging

from anyio import to_thread
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import get_db
from app.core.ratelimit import rate_limit
from app.models.review import EmotionPrediction, Review
from app.schemas.emotion import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_predict_limit = rate_limit(limit=120, window_seconds=60, name="/predict")


@router.post(
    "/predict", response_model=PredictResponse, dependencies=[Depends(_predict_limit)]
)
async def predict(
    payload: PredictRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> PredictResponse:
    if request.app.state.emotion_model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Train the model first.")

    settings = get_settings()

    # The torch forward pass is synchronous and CPU-bound. Calling it directly
    # from this async handler would run it on the event loop thread and stall
    # the whole process — including /health — for the duration, so hand it to a
    # worker thread the way app/services/shap_service.py already does.
    result = await to_thread.run_sync(request.app.state.emotion_model.predict, payload.text)

    # Persisting the review feeds the Dashboard's analytics, but it is not what
    # the caller asked for — the prediction above already succeeded. If the
    # database is unreachable, log the failure loudly and still return the
    # result rather than turning a good prediction into a 500.
    try:
        review = Review(review_text=payload.text, source="manual")
        db.add(review)
        await db.flush()

        prediction = EmotionPrediction(
            review_id=review.id,
            model_version=settings.model_version,
            scores=result["scores"],
            predicted_labels=result["predicted_labels"],
            top_label=result["top_label"],
            top_score=result["top_score"],
            business_bucket=result["business_bucket"],
        )
        db.add(prediction)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.warning(
            "Prediction succeeded but could not be persisted; it will not appear "
            "in Dashboard analytics.",
            exc_info=True,
        )

    return PredictResponse(
        scores=result["scores"],
        predicted_labels=result["predicted_labels"],
        top_label=result["top_label"],
        top_score=result["top_score"],
        top_label_above_threshold=result["top_label_above_threshold"],
        business_bucket=result["business_bucket"],
        model_version=settings.model_version,
        truncated=result["truncated"],
    )
