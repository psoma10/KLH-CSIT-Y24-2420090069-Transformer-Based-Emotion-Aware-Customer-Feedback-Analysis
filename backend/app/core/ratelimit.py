"""A small fixed-window rate limiter for the compute-heavy endpoints.

/explain costs 6-30 seconds of CPU per uncached call and runs in a threadpool
shared with every other request, so a few dozen concurrent calls from one
client saturate the pool and take the whole app down — no authentication
stands in the way. /analyze and /predict are cheaper but still run a model.

Deliberately in-process and dependency-free: this is a demo app behind no load
balancer, so a per-process counter is honest about what it protects. A real
deployment would use Redis so the limit holds across workers, and would sit
behind a proxy that enforces limits before a request ever reaches Python.
"""
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request


class RateLimiter:
    """Allows `limit` requests per `window_seconds` per client."""

    def __init__(self, limit: int, window_seconds: float, name: str):
        self.limit = limit
        self.window_seconds = window_seconds
        self.name = name
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _client_key(self, request: Request) -> str:
        # X-Forwarded-For is trivially spoofed, so it is only meaningful behind
        # a proxy that overwrites it. Prefer the real peer address and fall
        # back to a constant, which degrades to a global limit rather than to
        # no limit at all.
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> None:
        now = time.monotonic()
        key = self._client_key(request)

        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window_seconds
            while hits and hits[0] < cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(1, int(hits[0] + self.window_seconds - now))
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit exceeded for {self.name}: "
                        f"{self.limit} requests per {int(self.window_seconds)}s."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )

            hits.append(now)

            # Without this, every client that ever connected keeps an entry
            # forever — a slow memory leak on a public endpoint.
            if len(self._hits) > 10_000:
                for stale_key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale_key]


def rate_limit(limit: int, window_seconds: float, name: str):
    """Builds a FastAPI dependency enforcing the given limit."""
    limiter = RateLimiter(limit, window_seconds, name)

    async def dependency(request: Request) -> None:
        limiter.check(request)

    return dependency
