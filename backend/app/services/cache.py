"""Shared Redis client factory.

Imported by app/services/shap_service.py for SHAP result caching, and
available to any future service that needs Redis without re-parsing
settings.redis_url.
"""
from functools import lru_cache

from redis import asyncio as redis_asyncio

from app.core.config import get_settings


@lru_cache
def get_redis() -> redis_asyncio.Redis:
    settings = get_settings()
    return redis_asyncio.from_url(settings.redis_url, decode_responses=True)
