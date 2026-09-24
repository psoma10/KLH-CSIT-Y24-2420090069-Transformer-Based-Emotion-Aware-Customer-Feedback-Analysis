"""POST {api_v1_prefix}/analyze — segment-aware classification of a review.

Registered on the app router by app/main.py. Where /predict returns one score
for the whole text, this splits the review into sentences and scores each in a
single batched forward pass, so a review that praises one thing and complains
about another reports both instead of averaging them into a single misleading
label.

This is the same endpoint the Analyze page calls on every debounced keystroke
for the live preview, so it must not write to the database by default — early
versions did, and every half-typed sentence landed in the Dashboard as its own
"review". Persistence is opt-in via `payload.persist`, set only by the
visitor's actual Submit click, along with the star rating and product they
picked. When it does write, the row records the segment chosen to represent
the review, so the Dashboard counts the complaint rather than the praise that
happened to score marginally higher.
"""
import logging
from datetime import date

from anyio import to_thread
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import get_db
from app.core.ratelimit import rate_limit
from app.models.review import EmotionPrediction, Review
from app.schemas.emotion import AnalyzeRequest, AnalyzeResponse, SegmentResult
from app.services.segmentation import choose_headline_segment, split_segments

logger = logging.getLogger(__name__)

router = APIRouter()

# Generous, because the UI calls this while the visitor types: a 500ms debounce
# means an engaged user legitimately produces a request every second or so.
# This bounds automated abuse without interrupting real use.
_analyze_limit = rate_limit(limit=120, window_seconds=60, name="/analyze")


@router.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(_analyze_limit)])
async def analyze(
    payload: AnalyzeRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    model = request.app.state.emotion_model
    if model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Train the model first.")

    settings = get_settings()

    segments = split_segments(payload.text)
    if not segments:
        raise HTTPException(status_code=422, detail="Review contains no analyzable text.")

    # One batched forward pass for every segment, off the event loop — see
    # app/services/emotion_model.py:predict_many.
    results = await to_thread.run_sync(model.predict_many, segments)

    segment_results = [
        SegmentResult(
            text=text,
            scores=result["scores"],
            predicted_labels=result["predicted_labels"],
            top_label=result["top_label"],
            top_score=result["top_score"],
            top_label_above_threshold=result["top_label_above_threshold"],
            business_bucket=result["business_bucket"],
            truncated=result["truncated"],
        )
        for text, result in zip(segments, results)
    ]

    overall_index = choose_headline_segment(results)
    overall = segment_results[overall_index]

    buckets = {s.business_bucket for s in segment_results if s.business_bucket}
    is_mixed = len(buckets) > 1

    # Persistence is opt-in (see module docstring) and best-effort: the
    # analysis already succeeded and is what the caller asked for, so a DB
    # failure here should not turn a good prediction into a 500.
    if payload.persist:
        try:
            review = Review(
                review_text=payload.text,
                star_rating=payload.star_rating,
                product_id=payload.product_id,
                review_date=date.today(),
                source="manual",
            )
            db.add(review)
            await db.flush()

            db.add(
                EmotionPrediction(
                    review_id=review.id,
                    model_version=settings.model_version,
                    scores=overall.scores,
                    predicted_labels=overall.predicted_labels,
                    top_label=overall.top_label,
                    top_score=overall.top_score,
                    business_bucket=overall.business_bucket,
                )
            )
            await db.commit()
        except Exception:
            await db.rollback()
            logger.warning(
                "Review submission succeeded but could not be persisted; it "
                "will not appear in Dashboard analytics.",
                exc_info=True,
            )

    return AnalyzeResponse(
        segments=segment_results,
        overall=overall,
        overall_segment_index=overall_index,
        is_mixed=is_mixed,
        model_version=settings.model_version,
    )
