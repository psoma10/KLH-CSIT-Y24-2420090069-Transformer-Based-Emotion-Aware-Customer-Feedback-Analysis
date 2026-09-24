"""FastAPI application factory and lifespan wiring.

Entry point run by uvicorn (see backend/Dockerfile CMD and README dev
instructions): `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

On startup this creates DB tables (dev-only convenience, no Alembic) and
loads the fine-tuned emotion model onto app.state.emotion_model. Model
loading is best-effort: ML training happens separately (see
ml/train_roberta.py, run on Colab) and the resulting artifacts/model-v1/
directory may not exist locally yet, so a missing/broken model must not
prevent the API from starting — it should still serve /health and any
routes that don't need the model. Routes that do need the model (predict,
explain) are responsible for checking app.state.emotion_model themselves
and raising HTTP 503 if it is None.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analytics import router as analytics_router
from app.api.routes.analyze import router as analyze_router
from app.api.routes.explain import router as explain_router
from app.api.routes.model_info import router as model_info_router
from app.api.routes.predict import router as predict_router
from app.api.routes.reviews import router as reviews_router
from app.core.config import get_settings
from app.core.db import Base, engine
from app.services.emotion_model import EmotionModelService
from app.services.shap_service import ShapService

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev-only convenience: create tables directly from ORM metadata instead
    # of running Alembic migrations. Fine for a capstone demo.
    #
    # Best-effort for the same reason model loading is: Postgres is an optional
    # dependency for the read-only demo path (/health, /model-info, /predict,
    # /explain all work without it), so an unreachable database must degrade
    # those routes rather than stop the process from booting. Routes that do
    # persist reviews surface the failure themselves at request time.
    # NOTE: create_all only creates missing tables — it does not ALTER existing
    # ones. A database created before the index definitions in
    # app/models/review.py were added will not gain them here; drop the volume
    # (`docker compose down -v`) or add the indexes by hand.
    app.state.db_ready = False
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        app.state.db_ready = True
    except Exception:
        logger.warning(
            "Could not connect to the database at startup. The API will still start, "
            "but endpoints that read or write reviews will fail until Postgres is "
            "reachable.",
            exc_info=True,
        )

    # Model loading is best-effort. Training happens separately on Colab and
    # the artifact directory may not have been copied into this environment
    # yet — that must not crash the whole API. Endpoints that require the
    # model check app.state.emotion_model at request time and return 503.
    app.state.emotion_model = None
    app.state.shap_service = None
    try:
        app.state.emotion_model = EmotionModelService(
            settings.model_dir, settings.model_version, settings.model_max_length
        )
        # Shares the classifier's weights rather than loading a second ~476MB
        # copy — see app/services/shap_service.py.
        app.state.shap_service = ShapService(
            app.state.emotion_model,
            settings.shap_max_evals,
        )
        logger.info("Emotion model and SHAP explainer loaded from %s", settings.model_dir)
    except Exception:
        logger.warning(
            "Could not load emotion model from '%s'. The API will still start, but "
            "prediction/explanation endpoints will return 503 until a valid model "
            "artifact is available there.",
            settings.model_dir,
            exc_info=True,
        )

    yield

    app.state.emotion_model = None
    app.state.shap_service = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Explainable Emotion-Aware Customer Feedback Analysis",
        version=settings.model_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predict_router, prefix=settings.api_v1_prefix, tags=["predict"])
    app.include_router(analyze_router, prefix=settings.api_v1_prefix, tags=["analyze"])
    app.include_router(explain_router, prefix=settings.api_v1_prefix, tags=["explain"])
    app.include_router(reviews_router, prefix=settings.api_v1_prefix, tags=["reviews"])
    app.include_router(analytics_router, prefix=settings.api_v1_prefix, tags=["analytics"])
    app.include_router(model_info_router, prefix=settings.api_v1_prefix, tags=["model_info"])

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": getattr(app.state, "emotion_model", None) is not None,
            "db_ready": getattr(app.state, "db_ready", False),
        }

    return app


app = create_app()
