"""Dashboard analytics — distribution, bucket, and trend aggregates.

Registered on the app router by app/main.py. All three endpoints aggregate
EmotionPrediction (optionally joined to Review) with SQLAlchemy 2.0 async
select()/func.count()/group_by() against the engine set up in core/db.py.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.models.review import EmotionPrediction, Review
from app.schemas.emotion import BucketDistributionItem, EmotionDistributionItem, TrendPoint

router = APIRouter()


@router.get("/analytics/distribution", response_model=list[EmotionDistributionItem])
async def emotion_distribution(
    business_bucket: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> list[EmotionDistributionItem]:
    stmt = select(EmotionPrediction.top_label, func.count().label("count"))
    if business_bucket is not None:
        stmt = stmt.where(EmotionPrediction.business_bucket == business_bucket)
    stmt = stmt.group_by(EmotionPrediction.top_label).order_by(func.count().desc())

    rows = (await db.execute(stmt)).all()
    return [EmotionDistributionItem(label=label, count=count) for label, count in rows]


@router.get("/analytics/buckets", response_model=list[BucketDistributionItem])
async def bucket_distribution(db: AsyncSession = Depends(get_db)) -> list[BucketDistributionItem]:
    stmt = (
        select(EmotionPrediction.business_bucket, func.count().label("count"))
        .where(EmotionPrediction.business_bucket.is_not(None))
        .group_by(EmotionPrediction.business_bucket)
        .order_by(func.count().desc())
    )
    rows = (await db.execute(stmt)).all()
    return [BucketDistributionItem(bucket=bucket, count=count) for bucket, count in rows]


@router.get("/analytics/trends", response_model=list[TrendPoint])
async def trends(db: AsyncSession = Depends(get_db)) -> list[TrendPoint]:
    stmt = (
        select(Review.review_date, EmotionPrediction.business_bucket, func.count().label("count"))
        .join(EmotionPrediction, EmotionPrediction.review_id == Review.id)
        .where(Review.review_date.is_not(None), EmotionPrediction.business_bucket.is_not(None))
        .group_by(Review.review_date, EmotionPrediction.business_bucket)
        .order_by(Review.review_date.asc())
    )
    rows = (await db.execute(stmt)).all()
    return [TrendPoint(date=date, bucket=bucket, count=count) for date, bucket, count in rows]
