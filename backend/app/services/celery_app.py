"""Celery app + batch-prediction task.

Imported by app/api/routes/reviews.py (`from app.services.celery_app import
process_batch_job`) to enqueue CSV batch-upload jobs, and run by the
celery_worker container/process via:
    celery -A app.services.celery_app worker --loglevel=info

Celery tasks are synchronous, but app/core/db.py's engine/session are async
SQLAlchemy (asyncpg). Rather than fight sync/async across a process
boundary, this module builds its own *sync* SQLAlchemy engine from
settings.database_url with the driver swapped from asyncpg to psycopg2 —
core/db.py itself is not touched.

The model is loaded lazily, once per worker process (module-level singleton
checked/set inside the task), so importing this module in the API process
(to reference process_batch_job) never triggers a model load there.
"""
import logging
import sys
from datetime import datetime

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.models.review import BatchJob, EmotionPrediction, Review
from app.services.segmentation import choose_headline_segment, split_segments

logger = logging.getLogger(__name__)

settings = get_settings()

celery_app = Celery(
    "emotion_feedback",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Sync engine for this task only — swap the async driver for the sync one.
_SYNC_DATABASE_URL = settings.database_url.replace(
    "postgresql+asyncpg://", "postgresql+psycopg2://"
)
_sync_engine = create_engine(_SYNC_DATABASE_URL, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(bind=_sync_engine, expire_on_commit=False)

# Lazy, per-worker-process singleton for the model. Loaded on first task
# execution in this process, not at module import time.
_model_service = None


def _get_model_service():
    global _model_service
    if _model_service is None:
        # ml/ is a sibling of backend/ — not an installed package. Importing
        # EmotionModelService pulls in app.core.ml_path, which owns the
        # host/container-safe sys.path setup.
        from app.services.emotion_model import EmotionModelService

        _model_service = EmotionModelService(
            settings.model_dir, settings.model_version, settings.model_max_length
        )
    return _model_service


def _parse_review_date(raw: str | None):
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        logger.warning("Could not parse review_date '%s'; storing as null.", raw)
        return None


def _update_job(session: Session, job: BatchJob, **fields) -> None:
    for key, value in fields.items():
        setattr(job, key, value)
    session.commit()


@celery_app.task(name="app.services.celery_app.process_batch_job")
def process_batch_job(job_id: str, rows: list[dict]) -> None:
    """Runs emotion inference over a batch of CSV rows and persists results.

    Each row is a dict with keys review_id, product_id, review_text,
    star_rating, review_date — matching the CSV schema written by
    ml/data_prep.py's amazon-sample export and expected by the reviews
    upload route. Progress (processed_rows) is committed every 10 rows so
    a client polling GET /reviews/batch/{job_id} sees live progress.
    """
    session = SyncSessionLocal()
    try:
        job = session.get(BatchJob, job_id)
        if job is None:
            logger.error("BatchJob %s not found; aborting task.", job_id)
            return

        _update_job(session, job, status="running", total_rows=len(rows), processed_rows=0)

        model = _get_model_service()

        for i, row in enumerate(rows, start=1):
            review_text = (row.get("review_text") or "").strip()
            if not review_text:
                continue

            review = Review(
                product_id=row.get("product_id"),
                review_text=review_text,
                star_rating=_coerce_int(row.get("star_rating")),
                review_date=_parse_review_date(row.get("review_date")),
                source="batch_upload",
            )
            session.add(review)
            session.flush()  # assign review.id before FK use below

            # Score by segment, exactly as POST /api/analyze does. Using the
            # whole-text predict() here would mean a review typed into the UI
            # and the same review uploaded in a CSV get different labels, and
            # the Dashboard would be aggregating two methodologies at once —
            # with the bulk half carrying the averaging bug where praise in one
            # sentence cancels a complaint in the next.
            segments = split_segments(review_text) or [review_text]
            segment_results = model.predict_many(segments)
            prediction_data = segment_results[choose_headline_segment(segment_results)]

            prediction = EmotionPrediction(
                review_id=review.id,
                model_version=settings.model_version,
                scores=prediction_data["scores"],
                predicted_labels=prediction_data["predicted_labels"],
                top_label=prediction_data["top_label"],
                top_score=prediction_data["top_score"],
                business_bucket=prediction_data["business_bucket"],
            )
            session.add(prediction)

            if i % 10 == 0:
                job.processed_rows = i
                session.commit()

        job.processed_rows = len(rows)
        job.status = "done"
        job.completed_at = datetime.utcnow()
        session.commit()

    except Exception as exc:  # noqa: BLE001 - batch task must never raise unhandled
        logger.exception("process_batch_job failed for job_id=%s", job_id)
        session.rollback()
        job = session.get(BatchJob, job_id)
        if job is not None:
            job.status = "failed"
            # str(exc) here would put raw driver/SQL/tokenizer exception text
            # into a column served by GET /api/jobs/{id} to any unauthenticated
            # caller, fingerprinting the backend. The full traceback is already
            # in the logs above, where operators can see it and users cannot.
            job.error = f"Processing failed ({type(exc).__name__}). See server logs for details."
            job.completed_at = datetime.utcnow()
            session.commit()
    finally:
        session.close()


def _coerce_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
