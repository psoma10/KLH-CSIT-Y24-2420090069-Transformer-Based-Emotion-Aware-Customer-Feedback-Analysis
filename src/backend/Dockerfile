# Build context MUST be the repo root (not src/backend/), because this image
# also needs src/ml/ (emotion_labels.py, and shap_explainer.py for
# app/services/shap_service.py). docker-compose.yml sets:
#   build:
#     context: .
#     dockerfile: src/backend/Dockerfile
# Do not `docker build src/backend/` directly with this Dockerfile — src/ml/
# won't be in the build context and the COPY below will fail.
FROM python:3.11-slim

WORKDIR /app

# Runtime libs needed by torch/transformers wheels and psycopg2-binary at
# build time on slim images. curl and unzip are for scripts/fetch_model.sh —
# python:3.11-slim ships neither.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY src/backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY src/backend/app /app/app
COPY src/ml /app/ml
COPY scripts /app/scripts

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/results/ml-artifacts/model-v1

# The weights are gitignored, so nothing above brings in
# results/ml-artifacts/model-v1/. Bake them in at build time instead: baking
# beats fetching on startup because Render's disk is ephemeral, so a runtime
# fetch would re-download ~479MB on every restart and blow the health-check
# window.
#
# Pass MODEL_URL as a build arg (Render forwards service env vars to the
# build). Left unset, the build still succeeds and the image simply ships
# without weights — /health then reports model_loaded:false and /predict
# returns 503, which is the documented no-model state. A *wrong* MODEL_URL is
# a different matter and fails the build loudly, via fetch_model.sh.
ARG MODEL_URL
RUN if [ -n "$MODEL_URL" ]; then \
      MODEL_URL="$MODEL_URL" MODEL_DIR=/app/results/ml-artifacts/model-v1 \
        /app/scripts/fetch_model.sh; \
    else \
      echo "WARNING: MODEL_URL not set — image ships without weights; /predict will 503"; \
    fi

EXPOSE 8000

# Needs a shell so $PORT expands — Render assigns the port at runtime and routes
# to it; the 8000 fallback keeps `docker run` and docker-compose working locally.
# `exec` hands PID 1 to uvicorn so Render's SIGTERM on redeploy reaches the
# server instead of stopping at the shell.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
