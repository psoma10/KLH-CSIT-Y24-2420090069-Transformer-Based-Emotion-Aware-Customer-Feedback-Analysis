"""Tests for rate limiting and upload bounds.

These endpoints are unauthenticated and expensive, so the limits are the only
thing standing between one client and the whole app's CPU.
"""
import io

import pytest
from fastapi import HTTPException

from app.core.ratelimit import RateLimiter
from tests.conftest import requires_model


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host="1.2.3.4"):
        self.client = _FakeClient(host)


def test_limiter_allows_up_to_the_limit_then_rejects():
    limiter = RateLimiter(limit=3, window_seconds=60, name="/test")
    request = _FakeRequest()

    for _ in range(3):
        limiter.check(request)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check(request)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers


def test_limiter_counts_clients_independently():
    """One abusive client must not lock everyone else out."""
    limiter = RateLimiter(limit=2, window_seconds=60, name="/test")

    for _ in range(2):
        limiter.check(_FakeRequest("1.1.1.1"))

    # A different address still has its full allowance.
    limiter.check(_FakeRequest("2.2.2.2"))

    with pytest.raises(HTTPException):
        limiter.check(_FakeRequest("1.1.1.1"))


def test_limiter_window_expires():
    limiter = RateLimiter(limit=1, window_seconds=0.05, name="/test")
    request = _FakeRequest()

    limiter.check(request)
    with pytest.raises(HTTPException):
        limiter.check(request)

    import time

    time.sleep(0.06)
    limiter.check(request)  # window rolled over; must not raise


def _csv(rows, text="fine"):
    header = "review_id,product_id,review_text,star_rating,review_date\n"
    body = "".join(f"r{i},p1,{text},5,2026-01-01\n" for i in range(rows))
    return io.BytesIO((header + body).encode())


@requires_model
def test_batch_rejects_too_many_rows(client):
    from app.core.config import get_settings

    limit = get_settings().max_batch_rows
    response = client.post(
        "/api/batch", files={"file": ("big.csv", _csv(limit + 1), "text/csv")}
    )

    assert response.status_code == 413
    assert str(limit) in response.json()["detail"]


@requires_model
def test_batch_rejects_an_oversized_review_cell(client):
    from app.core.config import get_settings

    too_long = "x" * (get_settings().max_review_chars + 1)
    response = client.post(
        "/api/batch", files={"file": ("big.csv", _csv(1, text=too_long), "text/csv")}
    )

    assert response.status_code == 400


@requires_model
def test_batch_rejects_an_oversized_upload(client):
    """The read is chunked and bounded, so this must be refused rather than
    buffered whole into the API container's memory."""
    from app.core.config import get_settings

    settings = get_settings()
    padding = b"x" * (settings.max_upload_bytes + 1024)
    response = client.post("/api/batch", files={"file": ("huge.csv", io.BytesIO(padding), "text/csv")})

    assert response.status_code == 413
