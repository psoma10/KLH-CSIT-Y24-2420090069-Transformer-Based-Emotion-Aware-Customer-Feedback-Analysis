"""Review listing, CSV batch upload, and batch job status.

Registered on the app router by app/main.py. GET /reviews and POST /batch
both read/write via the get_db dependency; POST /batch enqueues the actual
row-by-row inference work onto Celery rather than doing it inline, since a
CSV can contain thousands of rows.
"""
import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import get_db
from app.core.ratelimit import rate_limit
from app.models.review import BatchJob, EmotionPrediction, Review
from app.schemas.emotion import BatchJobOut, ReviewOut
from app.services.celery_app import process_batch_job

router = APIRouter()

_REQUIRED_COLUMNS = {"review_id", "product_id", "review_text", "star_rating", "review_date"}


@router.get("/reviews", response_model=list[ReviewOut])
async def list_reviews(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    business_bucket: str | None = Query(default=None),
    product_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> list[ReviewOut]:
    stmt = select(Review, EmotionPrediction).outerjoin(
        EmotionPrediction, EmotionPrediction.review_id == Review.id
    )
    if business_bucket is not None:
        stmt = stmt.where(EmotionPrediction.business_bucket == business_bucket)
    if product_id is not None:
        stmt = stmt.where(Review.product_id == product_id)
    stmt = stmt.order_by(Review.created_at.desc()).limit(limit).offset(offset)

    rows = (await db.execute(stmt)).all()
    return [
        ReviewOut(
            id=review.id,
            product_id=review.product_id,
            review_text=review.review_text,
            star_rating=review.star_rating,
            review_date=review.review_date,
            top_label=prediction.top_label if prediction else None,
            business_bucket=prediction.business_bucket if prediction else None,
            created_at=review.created_at,
        )
        for review, prediction in rows
    ]


# Each accepted upload occupies a Celery worker for as long as it takes to
# score every row, so this is about queue depth, not request cost.
_batch_limit = rate_limit(limit=5, window_seconds=300, name="/batch")


@router.post(
    "/batch", response_model=BatchJobOut, dependencies=[Depends(_batch_limit)]
)
async def upload_batch(file: UploadFile, db: AsyncSession = Depends(get_db)) -> BatchJobOut:
    settings = get_settings()

    # Read in chunks and stop at the limit. `await file.read()` with no bound
    # would pull the whole upload into memory before anything could reject it,
    # so a single large file could exhaust the API container — and every row
    # then costs a model inference in the worker.
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"CSV exceeds the {settings.max_upload_bytes // (1024 * 1024)}MB upload limit."
                ),
            )
        chunks.append(chunk)
    raw = b"".join(chunks)

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or not _REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain columns: {sorted(_REQUIRED_COLUMNS)}",
        )
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV has no data rows.")
    if len(rows) > settings.max_batch_rows:
        # Every row costs a model inference in the worker, so an oversized CSV
        # pins a worker for hours and delays every other job behind it.
        raise HTTPException(
            status_code=413,
            detail=f"CSV has {len(rows)} rows; the limit is {settings.max_batch_rows}.",
        )

    # Cap per-cell text too: review_text feeds an unbounded Text column, and
    # the interactive endpoints already cap at 5000 characters.
    for row in rows:
        if len(row.get("review_text") or "") > settings.max_review_chars:
            raise HTTPException(
                status_code=400,
                detail=f"A review exceeds the {settings.max_review_chars}-character limit.",
            )

    job = BatchJob(status="pending", total_rows=len(rows))
    db.add(job)
    await db.commit()
    await db.refresh(job)

    process_batch_job.delay(job.id, rows)

    return BatchJobOut.model_validate(job)


@router.get("/jobs/{job_id}", response_model=BatchJobOut)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)) -> BatchJobOut:
    job = await db.get(BatchJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Batch job '{job_id}' not found.")
    return BatchJobOut.model_validate(job)
