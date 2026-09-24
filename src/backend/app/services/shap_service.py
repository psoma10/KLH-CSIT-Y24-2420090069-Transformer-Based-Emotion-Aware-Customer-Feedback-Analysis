"""Redis-cached wrapper around ml.shap_explainer.EmotionExplainer.

Imported by app/api/routes/explain.py. SHAP itself has no notion of caching
or async — this module adds both without touching ml/shap_explainer.py.
"""
import hashlib
import json
import logging
from pathlib import Path

import anyio

from app.core.config import get_settings
from app.services.cache import get_redis

# ml/ is a sibling directory, not an installed package — app/core/ml_path owns
# the host/container-safe lookup that makes the import below resolve.
import app.core.ml_path  # noqa: F401,E402

from shap_explainer import EmotionExplainer  # noqa: E402

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "shap"


class ShapService:
    """Explains a single review's emotion prediction, caching results in Redis."""

    def __init__(
        self,
        emotion_model: "EmotionModelService",
        max_evals: int = 200,
    ):
        """Shares the classifier's already-loaded weights.

        Loading a second copy from disk cost ~476MB of resident memory per
        process for nothing: SHAP only ever runs the same forward pass, under
        no_grad, and never mutates the model. Sharing the instance also makes
        it impossible for the two to drift apart on max_length, which would
        silently make /explain analyze a different amount of text than
        /analyze and report scores that disagree.
        """
        self.model_version = emotion_model.model_version
        self._explainer = EmotionExplainer(
            emotion_model.tokenizer,
            emotion_model.model,
            max_evals=max_evals,
            max_length=emotion_model.max_length,
        )

    def _cache_key(self, text: str, label: str | None) -> str:
        # max_length is part of the key because changing it changes how much of
        # the text is explained — otherwise a config change would keep serving
        # explanations computed against the old window.
        raw = f"{text}|{label or ''}|{self.model_version}|{self._explainer.max_length}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{_CACHE_PREFIX}:{digest}"

    async def explain(self, text: str, label: str | None = None) -> tuple[dict, bool]:
        """Returns (explanation, cache_hit). Runs SHAP off the event loop —

        SHAP does ~max_evals forward passes through the transformer (2-5s of
        pure CPU work), which would otherwise block every other request.
        """
        settings = get_settings()
        cache_key = self._cache_key(text, label)

        # The cache only saves repeat work — SHAP is perfectly capable of
        # running without it. Treat Redis being unreachable (or returning
        # something unparseable) as a miss rather than letting it fail the
        # request, and never let a write failure throw away a computation that
        # already cost 20-30 seconds of CPU.
        try:
            cached = await get_redis().get(cache_key)
            if cached is not None:
                return json.loads(cached), True
        except Exception:
            logger.warning("SHAP cache read failed; recomputing.", exc_info=True)

        result = await anyio.to_thread.run_sync(self._explainer.explain, text, label)

        try:
            await get_redis().set(cache_key, json.dumps(result), ex=settings.shap_cache_ttl_seconds)
        except Exception:
            logger.warning("SHAP cache write failed; returning the result uncached.", exc_info=True)

        return result, False
