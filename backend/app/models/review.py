"""SQLAlchemy models for reviews and their emotion predictions.

Imported by app/api/routes/predict.py, analytics.py, reviews.py, and
app/services/batch.py (Celery task writes here). Registered on Base's
metadata, which app/main.py's lifespan uses to create tables in dev.
"""
import uuid
from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Review(Base):
    __tablename__ = "reviews"

    # Indexes cover the columns the analytics and listing queries actually
    # filter, group, and sort on. They cost nothing at demo scale, but /batch
    # exists specifically to load this table in bulk, so the sequential scans
    # these replace would start to hurt exactly when the app is being used as
    # intended.
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    review_text: Mapped[str] = mapped_column(Text, nullable=False)
    star_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(32), default="manual")  # manual | batch_upload
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    prediction: Mapped["EmotionPrediction | None"] = relationship(
        back_populates="review", uselist=False, cascade="all, delete-orphan"
    )


class EmotionPrediction(Base):
    __tablename__ = "emotion_predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    review_id: Mapped[str] = mapped_column(String(36), ForeignKey("reviews.id"), nullable=False, unique=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)

    # Full 28-label sigmoid score vector, keyed by label name.
    scores: Mapped[dict] = mapped_column(JSON, nullable=False)
    # Labels that passed their tuned per-label threshold.
    predicted_labels: Mapped[list] = mapped_column(JSON, nullable=False)
    top_label: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    top_score: Mapped[float] = mapped_column(Float, nullable=False)
    business_bucket: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    review: Mapped["Review"] = relationship(back_populates="prediction")


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | running | done | failed
    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    processed_rows: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
