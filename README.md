# Explainable Transformer-Based Emotion-Aware Customer Feedback Analysis

*A Multi-Label Fine-Grained Emotion Classification System with SHAP-Based Token Attribution for Business-Actionable Review Insights*

**Capstone Project — Business Analytics**

## Team

| Role | Name | ID |
|---|---|---|
| Supervisor | Dr. Swanthana Katanguri | — |
| Team Member | Pujith Krishna Soma | 2420090069 |
| Team Member | Ch G Sree Susheel | 2420090124 |
| Team Member | Sriram V | 2420090126 |
| Team Member | Yaswanth Chowdary | 2420090106 |

## Abstract

Conventional sentiment analysis tools classify customer feedback into coarse buckets — positive, negative, or neutral — which is too coarse for a business to act on and too opaque to trust. This project builds a transformer-based multi-label emotion classification framework that closes both gaps at once. A `roberta-base` model is fine-tuned on the GoEmotions taxonomy (27 emotions + neutral, 28 labels total) using multi-label (sigmoid, not softmax) classification, so a single review can carry more than one emotion simultaneously. Predictions are explained at the token level via SHAP, so a non-technical stakeholder can see not just a probability score but the specific words that drove it. The 27 fine-grained emotions roll up into 4 business-actionable buckets — **satisfaction, frustration, confusion, concern** — and each review is scored **segment by segment** rather than as a single average, so a review that opens with praise and ends with a complaint reports both instead of one misleading label. Training extends beyond GoEmotions alone via a masked multi-label loss that combines auxiliary corpora (CARER/dair-ai emotion, DailyDialog) without corrupting the label space with false negatives from their partial label coverage. The system is wrapped in a FastAPI backend and a React dashboard, making it a deployable, callable product rather than a notebook experiment. Full comparison against the reviewed literature — headline metrics, per-label deltas, and a capability matrix — is in [`docs/COMPARISON.md`](docs/COMPARISON.md).

## Current phase status

| Phase | Status |
|---|---|
| Literature survey (12 papers) + research gap identification | ✅ Done — [`docs/docx_context.md`](docs/docx_context.md) |
| Data preparation (GoEmotions, CARER, DailyDialog) | ✅ Done — [`data/`](data/) |
| Model training (baseline + multi-corpus) | ✅ Done — [`results/ml-artifacts/`](results/ml-artifacts/) |
| Evaluation + comparison against literature | ✅ Done — [`docs/COMPARISON.md`](docs/COMPARISON.md), [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) |
| SHAP explainability integration | ✅ Done — [`src/ml/shap_explainer.py`](src/ml/shap_explainer.py) |
| Backend API (FastAPI) | ✅ Done — [`src/backend/`](src/backend/) |
| Frontend dashboard (React) | ✅ Done — [`src/frontend/`](src/frontend/) |
| Deployment (Render + Vercel) | 🟡 In progress — backend live, model weights not yet hosted on Render (`/predict` returns 503 until `MODEL_URL` is set) |
| Authentication / production hardening | ⬜ Not started — see [Security posture](#security-posture) |

---

## Table of contents

- [Project structure](#project-structure)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Datasets](#datasets)
- [Quick start](#quick-start)
- [Setup (step by step)](#setup-step-by-step)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Design decisions](#design-decisions)
- [Model performance](#model-performance)
- [Tests](#tests)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Status and limitations](#status-and-limitations)

---

## Project structure

Mandatory top-level layout:

```
.
├── src/            # All source code
│   ├── backend/     # FastAPI app (API, models, services, tests)
│   ├── frontend/    # React + Vite dashboard
│   └── ml/          # Training pipeline, SHAP explainer, configs
├── docs/           # Model card, literature comparison, source papers
│   └── papers/      # PDFs of the 12 reviewed literature-survey papers
├── data/           # Datasets (raw + xlsx-viewable + preprocessed jsonl)
├── results/        # Trained model artifacts, metrics, thresholds
├── reports/        # Graded review-phase submission materials
│   ├── review-1-materials/
│   └── review-2-materials/
├── scripts/        # Deployment helper scripts (Render build, model fetch)
├── docker-compose.yml
└── README.md       # This file
```

See [Detailed source layout](#detailed-source-layout) below for what's inside `src/`.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Data Sources      │
                    │  GoEmotions (train) │
                    │  Amazon Reviews     │
                    │  (inference/demo)   │
                    └──────────┬──────────┘
                               │  data_prep.py
                               ▼
                    ┌─────────────────────┐
                    │  RoBERTa Training   │
                    │  (Colab / GPU)      │
                    │  train_roberta.py   │
                    │  evaluate.py        │
                    └──────────┬──────────┘
                               │  artifacts/model-v1/
                               │  (weights + thresholds.json + metrics)
                               ▼
        ┌──────────────────────────────────────────────┐
        │              FastAPI Backend                 │
        │  /predict   → emotion scores (whole text)    │
        │  /analyze   → per-segment scores + headline  │
        │  /explain   → SHAP token attributions        │
        │  /batch     → CSV upload → Celery job        │
        │  /analytics → distribution / buckets / trends│
        │                                              │
        │   ┌───────────┐  ┌────────┐  ┌────────────┐  │
        │   │ PostgreSQL│  │ Redis  │  │   Celery   │  │
        │   │ (reviews, │  │ (SHAP  │  │ (batch CSV │  │
        │   │  jobs)    │  │ cache) │  │  worker)   │  │
        │   └───────────┘  └────────┘  └────────────┘  │
        └──────────────────────┬───────────────────────┘
                               │ REST (JSON)
                               ▼
                    ┌─────────────────────┐
                    │  React Frontend     │
                    │  /          Analyze │
                    │  /dashboard Dashboard│
                    │  /batch     Batch   │
                    │  /model     Model   │
                    └─────────────────────┘
```

**Two independent degradation paths, both deliberate.** The backend starts even when no trained model is present and even when Postgres is unreachable. `app/main.py` attempts both during lifespan startup and logs a warning on failure rather than aborting. `/predict`, `/analyze`, and `/explain` return **HTTP 503** until a valid artifact exists at `MODEL_DIR`; routes that persist data surface the database failure at request time. `GET /health` reports `model_loaded` and `db_ready` so you can tell all four states apart.

---

## Tech stack

**ML**
- Python 3.11, HuggingFace `transformers` + `datasets`
- `roberta-base` (125M params) fine-tuned for multi-label classification
- SHAP for token-level explainability
- GoEmotions (training) + two optional auxiliary corpora — see [Datasets](#datasets)
- Amazon Reviews sample (inference/demo only, never trained on)

**Backend**
- FastAPI (async), Pydantic v2 schemas, `pydantic-settings` for config
- PostgreSQL via SQLAlchemy async (`asyncpg`) — review + batch job storage
- Redis — SHAP explanation cache
- Celery — async batch CSV processing
- In-process token-bucket rate limiter (`app/core/ratelimit.py`)

**Frontend**
- React 18 + Vite (JavaScript, not TypeScript)
- Tailwind CSS
- TanStack Query (data fetching/caching)
- Recharts (charts)
- React Router 6

**Infra**
- Docker Compose (postgres, redis, backend, celery_worker, frontend)
- Render Blueprint (`render.yaml` at repo root) — API, worker, Postgres, Redis
- Vercel (`src/frontend/vercel.json`) — static frontend
- Model training designed for Google Colab (GPU), not containerized

---

## Datasets

`model-v1` — the weights whose metrics are quoted below — was trained on **GoEmotions alone**. The two auxiliary corpora are wired up but off by default (`data.aux_datasets: []`), so the published numbers stay reproducible from the committed config; enabling them is a separate experiment.

| Dataset | Labels | Role | Source | Paper |
|---|---|---|---|---|
| GoEmotions (simplified) | 27 emotions + neutral | train / val / test | [HF](https://huggingface.co/datasets/google-research-datasets/go_emotions) | [arXiv:2005.00547](https://arxiv.org/abs/2005.00547) |
| dair-ai/emotion (CARER) | 6, single-label | auxiliary train | [HF](https://huggingface.co/datasets/dair-ai/emotion) | [D18-1404](https://aclanthology.org/D18-1404/) |
| DailyDialog | 7, single-label | auxiliary train | [HF](https://huggingface.co/datasets/roskoN/dailydialog) | [arXiv:1710.03957](https://arxiv.org/abs/1710.03957) |
| amazon_polarity | binary sentiment | inference / demo only | [HF](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | — |

Each auxiliary corpus annotates only part of the 28-label space, and a label it does not annotate is **unknown, not absent** — a `dair-ai` row tagged `sadness` says nothing about whether `gratitude` applies, because annotators were never asked. Scoring those zeros as negatives would drive every uncovered label toward zero wherever that corpus appears. Rows therefore carry a coverage mask and the loss is computed only over annotated positions. Mappings and the reasoning behind each live in [`src/ml/aux_datasets.py`](src/ml/aux_datasets.py); [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) is the source of truth for provenance and metrics.

Worth stating plainly: these corpora are tweets and dialogue, and none contains `grief`, `pride`, or `relief`. They will not fix those three 0.000-F1 labels, and they do not close the product-review domain gap.

## Quick start

Fastest path for someone who just wants to see it running, assuming a trained model already exists at `results/ml-artifacts/model-v1/`:

```bash
cd Project
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

No model yet? Follow [Setup](#setup-step-by-step) — training must happen first, and it does not run inside Docker.

---

## Setup (step by step)

There is a real dependency order here: a trained model must exist before the backend can serve predictions, so ML setup comes first.

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd Project
```

### 2. ML: train and evaluate the model

```bash
cd src/ml
pip install -r requirements.txt

# Prepare data
python data_prep.py --goemotions --out data/
python data_prep.py --amazon-sample --out data/ --n 500

# Optional: the auxiliary corpora (see Datasets). Also set
# data.aux_datasets in configs/train_config.yaml, or they are ignored.
python data_prep.py --aux --out data/

# Train (needs a GPU — recommended: upload ml/ to Google Colab and run there
# with a T4 runtime, ~30 min; a local CUDA GPU also works)
python train_roberta.py --config configs/train_config.yaml

# Tune per-label thresholds and report test-set metrics
python evaluate.py --config configs/train_config.yaml
```

Confirm `results/ml-artifacts/model-v1/` now contains the model weights, `thresholds.json`, and evaluation metrics. **`evaluate.py` is what produces `thresholds.json` — training alone does not.** The backend will not serve `/predict`, `/analyze`, or `/explain` until this directory exists.

If you trained on Colab, download `artifacts/model-v1/` and place it at `results/ml-artifacts/model-v1/` locally — that path is what both the backend `.env` default and the Docker Compose volume mount expect. The weights are ~479MB and gitignored, so they never travel with a `git clone`.

### 3. Infra: start Postgres and Redis

```bash
docker compose up -d postgres redis
```

(Or install PostgreSQL and Redis locally and point the backend at them.)

### 4. Backend

```bash
cd src/backend
cp .env.example .env   # adjust values if not using the Docker defaults
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000
```

In a separate terminal, start the Celery worker for batch CSV jobs:

```bash
celery -A app.services.celery_app worker --loglevel=info
```

Tables are created from ORM metadata on startup (no Alembic) — a deliberate simplification, noted under [Design decisions](#design-decisions).

### 5. Frontend

```bash
cd src/frontend
npm install
npm run dev   # http://localhost:5173
```

Set `VITE_API_URL` if the backend isn't at `http://localhost:8000`.

---

## Configuration

Backend settings live in `src/backend/app/core/config.py` and are read from environment variables (`src/backend/.env` in local dev, the `environment:` block in Docker Compose, `envVars:` in `render.yaml` on Render). Every value has a working default, so `.env` is only needed to override them.

### Model

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `../../results/ml-artifacts/model-v1` | Where the fine-tuned model artifact is loaded from |
| `MODEL_VERSION` | `model-v1` | Reported by `/api/model/info`; part of the SHAP cache key |
| `MODEL_MAX_LENGTH` | `128` | Tokenizer truncation length |

### Explainability

| Variable | Default | Purpose |
|---|---|---|
| `SHAP_MAX_EVALS` | `200` | SHAP evaluation budget — raise for fidelity, lower for speed |
| `SHAP_CACHE_TTL_SECONDS` | `604800` (7 days) | Redis TTL for cached explanations |

### Datastores

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/emotion_feedback` | Async Postgres DSN. `app/core/db.py` normalizes a `postgres://` scheme to the asyncpg driver, which is what Render supplies. |
| `REDIS_URL` | `redis://localhost:6379/0` | SHAP cache |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Batch job queue |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2` | Batch job results |

### Limits

| Variable | Default | Purpose |
|---|---|---|
| `MAX_UPLOAD_BYTES` | `5242880` (5MB) | Hard cap on CSV upload size, enforced while streaming |
| `MAX_BATCH_ROWS` | `5000` | Max rows accepted per batch job |
| `MAX_REVIEW_CHARS` | `5000` | Max characters per review; matches the `PredictRequest`/`AnalyzeRequest` schemas |

### App

| Variable | Default | Purpose |
|---|---|---|
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Allowed CORS origin. Compared against the browser's `Origin` header — **no trailing slash**. |
| `API_V1_PREFIX` | `/api` | Router prefix for all non-`/health` routes |

Frontend uses a single variable, `VITE_API_URL` (see `src/frontend/.env.example`).

---

## API reference

Full interactive documentation (request/response schemas, try-it-out) is auto-generated by FastAPI and available at **http://localhost:8000/docs** once the backend is running.

| Method | Path | Rate limit | Description |
|---|---|---|---|
| GET | `/health` | — | Liveness check; reports `model_loaded` and `db_ready` |
| POST | `/api/predict` | 120/min | Classify a whole review as one unit |
| POST | `/api/analyze` | 120/min | Split the review into segments, score each, return a headline segment and a `mixed` flag |
| POST | `/api/explain` | 10/min | SHAP token attributions for a review |
| GET | `/api/reviews` | — | List stored/classified reviews |
| POST | `/api/batch` | 5 per 5 min | Upload a CSV for async batch classification |
| GET | `/api/jobs/{job_id}` | — | Poll status/result of a batch job |
| GET | `/api/analytics/distribution` | — | Per-label emotion score distribution |
| GET | `/api/analytics/buckets` | — | Business bucket breakdown |
| GET | `/api/analytics/trends` | — | Emotion trends over time |
| GET | `/api/model/info` | — | Active model version, training/eval metrics |

### `/predict` vs `/analyze`

Use `/analyze` for anything user-facing. `/predict` scores the whole text in one forward pass, which averages a mixed review into a single label; `/analyze` splits on sentence boundaries **and contrastive connectives** (`but`, `however`, `although`, `though`, `except`, `unfortunately`, `whereas`), scores every segment in one batched pass, then picks a headline segment. It also records which segment represented the review, so the Dashboard counts the complaint rather than the praise that happened to score marginally higher.

Worked example — `"The fit is too good. What the fuck is this bullshit"`:

| Endpoint | Result |
|---|---|
| `/predict` | `admiration` — the negative clause is averaged away |
| `/analyze` | `admiration` 0.92 **and** `anger` 0.82, `mixed: true` — the truth |

Segmentation is bounded: fragments under 12 characters merge into the previous segment (so nothing goes unscored), and a maximum of 12 segments are produced (so `"a. " * 1000` cannot turn one request into a thousand-row batch). The logic lives in `app/services/segmentation.py`, kept free of torch and FastAPI imports so it is testable as a pure function.

### Two flags every caller should read

- **`truncated`** — returned by `/predict`, `/analyze`, and `/explain`. The model reads only the first 128 tokens, so a long review can be scored on its opening alone. This flag is the only way a caller can tell that happened.
- **`top_label_above_threshold`** — `top_label` is the raw argmax and is *always* populated, even when no label clears its own tuned cutoff and `predicted_labels` comes back empty. Check `predicted_labels` (or this flag) when you need "the model is actually confident about something."

### Batch CSV format

`POST /api/batch` requires a UTF-8 CSV containing exactly these columns (extra columns are ignored, missing ones are rejected with HTTP 400):

```csv
review_id,product_id,review_text,star_rating,review_date
r-001,p-100,"Arrived broken and support never replied.",1,2026-03-14
r-002,p-100,"Honestly better than I expected for the price.",5,2026-03-15
```

The endpoint returns a job record immediately; poll `GET /api/jobs/{job_id}` for status and results. The Celery worker must be running or jobs stay queued.

---

## Design decisions

**Multi-label BCE loss instead of softmax.** GoEmotions labels co-occur — a single comment can be both "joy" and "gratitude." Softmax forces mutually exclusive classes, so the model is trained with `BCEWithLogitsLoss` and a sigmoid over each of the 28 labels independently.

**Per-label tuned thresholds instead of a global 0.5 cutoff.** Rare emotions like grief, pride, and relief make up under 1% of the training data. A single 0.5 threshold across all labels systematically suppresses them. `evaluate.py` searches 19 cutoffs per label (0.05–0.95) on the validation split and persists the result to `thresholds.json`, which the backend loads alongside the model weights. The threshold vector is model state shipped with the artifact, not a constant in code.

**Segment-level scoring instead of document-level.** Reviews pivot mid-sentence far more often than they pivot between sentences: *"the sound is great but the battery is garbage"* is one sentence carrying two opposite verdicts, and scoring it whole reports only the first. Splitting before contrastive connectives keeps each verdict separately visible.

**SHAP results cached in Redis, keyed by content hash.** SHAP inference takes roughly 2–5 seconds per text and is deterministic for a fixed model version, so re-explaining the same text is pure waste. The backend hashes the input text plus model version and caches the attribution result, making repeat lookups near-instant.

**SHAP shares the classifier's weights.** `ShapService` wraps the already-loaded `EmotionModelService` rather than loading a second ~479MB copy of the model.

**Inference runs off the event loop.** Model forward passes are dispatched through `anyio.to_thread.run_sync`, so a slow inference cannot block the async request loop.

**27 fine-grained emotions rolled up into 4 business buckets.** A bar chart of 27 near-zero emotion scores is not actionable for a business stakeholder. Every fine-grained label maps to one of `satisfaction / frustration / confusion / concern`, so the dashboard can say "62% frustration this week" — a number someone can act on. `neutral` is deliberately excluded from all four buckets and reported separately as a coverage stat, since "neutral" is the absence of signal rather than a business outcome.

**A missing model or database degrades the API instead of crashing it.** Training happens on Colab, separate from the app, so the artifact directory is frequently absent in a fresh checkout, and Postgres is optional for the read-only demo path. Both loads are wrapped in try/except at startup; only the routes that genuinely need the missing dependency fail.

**`src/ml/emotion_labels.py` is the single source of truth for label order.** It is imported by every training script *and* by the backend's model service. `predict()` zips this hardcoded label list against raw logits and ignores the checkpoint's own `id2label` — so a checkpoint trained in a different order would attach every score to the wrong emotion and never raise an exception. A test guards this specifically.

**Tables created from ORM metadata, no Alembic.** The schema is small and the project has no production upgrade path to protect, so `Base.metadata.create_all` at startup buys working setup in one line. Note that `create_all` only creates *missing* tables — it does not `ALTER` existing ones, so a database created before the index definitions in `app/models/review.py` were added will not gain them. Drop the volume (`docker compose down -v`) or add the indexes by hand. A real deployment needs migrations.

**Auxiliary training data is masked, not concatenated.** dair-ai/emotion and DailyDialog each annotate only a subset of the 28 GoEmotions labels — a dair-ai row tagged `sadness` says nothing about whether `gratitude` applies, because its annotators were never asked. Naively concatenating the corpora and treating unannotated labels as negatives would teach the model those labels are absent whenever that corpus's text pattern appears, actively damaging the labels the extra data was meant to leave alone. Every row in `data_prep.py`'s output carries a `mask` of the label indices its source actually annotates, and `train_roberta.py`'s `MaskedTrainer` computes BCE per-element, multiplies by the mask, and averages only over annotated positions — divided by the *unmasked* count rather than by 28, so a 6-label dair-ai row's loss stays comparable in scale to a 28-label GoEmotions row's and mixed batches don't quietly down-weight the auxiliary data. Verified directly: unannotated labels receive exactly zero gradient in a synthetic test.

**Training data fetched as Parquet over plain HTTP, not through the `datasets` library.** `datasets` pulls in `lzma` internally, which fails to import on a Python built without the system `_lzma` library — a real failure mode on some local setups, not hypothetical. HuggingFace publishes a canonical Parquet export for every dataset with an open schema (verified against the published GoEmotions split sizes: 43410/5426/5427), so `data_prep.py` downloads those directly instead. This also means training data can be prepared with a bare `pyarrow`, no heavyweight dataset-loading dependency at all.

**SemEval-2018 Task 1 was evaluated as a third auxiliary corpus and rejected.** The official dataset is script-based with no Parquet export; the one community mirror that does export Parquet ships text with stopwords stripped and casing folded ("whatever decide make sure make happy"). RoBERTa is pretrained on natural text and leans on function words and casing to place tokens in context, so mixing that register into training would add distribution noise rather than emotion signal. Excluded deliberately — see `src/ml/aux_datasets.py` for the full reasoning.

**Live preview and Submit are different requests, only one of which writes to the database.** The Analyze page calls `/analyze` on every debounced keystroke to drive the live-analysis panel, and early versions of this endpoint persisted a `Review` row on every one of those calls — the Dashboard ended up full of duplicate mid-typing fragments, with `product_id`/`star_rating`/`review_date` always null since the preview never carried them. `AnalyzeRequest.persist` defaults to `false`; the preview never sets it, so it never writes. The Submit button is a separate `useMutation` that calls the same endpoint with `persist: true` plus the star rating and product actually selected in the UI, and only that call writes a row, with `review_date` stamped server-side.

---

## Model performance

Full per-label results, tuned thresholds, and limitations: **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)**. The weights are gitignored, so that file is where the numbers live independently of the artifact.

**GoEmotions test split** — Macro F1 **0.475** · Micro F1 **0.600** · Validation loss **0.085**

Both figures are in line with published `roberta-base` GoEmotions baselines (macro F1 ≈ 0.46–0.52), so the model behaves like a correctly-trained model of its size rather than an underfit one.

**Reliable labels** (these carry the demo): `gratitude` 0.919 · `amusement` 0.821 · `love` 0.814 · `admiration` 0.717 · `neutral` 0.688

**Do not trust these:** `grief`, `pride`, and `relief` all score **0.000 F1** with 6–16 test examples each — treat any appearance as noise. `nervousness` (0.118), `realization` (0.244), and `disappointment` (0.292) are closer to a coin flip than a signal. `disappointment` (precision 0.227) and `annoyance` (0.308) over-fire, which skews bucket counts toward frustration.

### Multi-corpus training experiment

`model-v1`'s numbers above are GoEmotions-only. A separate experiment tested whether mixing in [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) and [DailyDialog](https://huggingface.co/datasets/roskoN/dailydialog) — see [Datasets](#datasets) — improves anything, using two configs identical except for `data.aux_datasets` (same seed, same 4 epochs, same learning rate) so the comparison isolates the dataset change and nothing else. Run twice, independently, on different hardware:

| Run | Macro F1 | Micro F1 |
|---|---|---|
| Baseline (GoEmotions only) | 0.4747 | 0.5997 |
| Enhanced (+ dair-ai + DailyDialog) | 0.4845 | 0.6084 |
| **Delta** | **+0.0098** | **+0.0087** |

Small net gain, but the aggregate hides where it actually mattered. `nervousness` — the model's second-weakest label, 23 test examples — jumped **+0.166 to +0.321 F1** across the two runs, along with real gains on `disgust`, `optimism`, `surprise`, and `anger`. `realization` and `embarrassment` regressed in both runs by a consistent margin, a genuine trade-off rather than noise: neither auxiliary corpus covers those labels, so the model's gradient budget shifted toward the labels that gained coverage. `grief`, `pride`, and `relief` stayed at exactly 0.000 in both runs — two independent confirmations that they are data-starved rather than merely under-weighted, since neither auxiliary corpus contains any of the three.

The mechanism this depends on: each auxiliary corpus annotates only a subset of the 28 labels, so a label it does not annotate is *unannotated*, not negative. Treating those zeros as negatives would suppress every uncovered label whenever that corpus appears — the opposite of the intended effect. `src/ml/aux_datasets.py` and `src/ml/train_roberta.py`'s masked BCE loss (see [Design decisions](#design-decisions)) exist specifically to prevent that. Not used for `model-v1` — `aux_datasets: []` by default, so the headline numbers above stay reproducible from the committed config; this is a documented alternative, not the shipped model.

---

## Tests

```bash
cd src/backend
pytest tests/ -q
```

**42 tests across 4 suites** — `test_api.py` (15), `test_segmentation.py` (11), `test_emotion_model.py` (10), `test_limits.py` (6).

The suite runs against the **real** fine-tuned checkpoint and the real FastAPI app — the model is the system under test, so it is never mocked. It needs `results/ml-artifacts/model-v1/` to exist and skips itself with a clear message if it doesn't. Postgres is not required: the app is designed to start without a database, and the tests exercise that degraded path on purpose.

What it guards, beyond ordinary route behavior:

- **Label ordering.** A checkpoint trained in a different label order would silently mis-assign every score. This is the only thing that catches it.
- **Explanation honesty.** SHAP must attribute only tokens the model actually read, or the UI renders "never seen" as "didn't matter."
- **Segmentation correctness.** Mixed reviews split into the right clauses; short fragments merge rather than drop; segment count stays bounded.
- **Limits.** Upload size, row count, and review length bounds actually reject.
- **Boundary validation** and **graceful degradation** when Postgres, Redis, or `thresholds.json` are unavailable or corrupt.

Each test was checked by reverting the fix it covers and confirming it fails — a test that still passes with the feature removed is not testing anything.

---

## Deployment

The stack splits across two providers: Python on Render, static frontend on Vercel.

**Currently deployed:** frontend at [alt-labs-ten.vercel.app](https://alt-labs-ten.vercel.app), backend at `alt-labs.onrender.com`. Honest status as of this writing: the backend is live and `/health` responds, but `MODEL_URL` is not yet set on the Render service, so the image shipped without weights — `/predict` and `/analyze` return 503 until the artifact is hosted and the service is upgraded off the free plan (see the plan note below; the model alone is measured at ~675MB resident, well past the 512MB free-tier ceiling).

### Backend, worker, Postgres, Redis → Render

`render.yaml` sits at the **repo root** (Render requires this); every path in it is relative to that root. Deploy via **Render Dashboard → New → Blueprint**, pointed at the repo; Render creates all four services in one pass.

| Service | Type | Plan | Notes |
|---|---|---|---|
| `emotion-api` | web | starter | `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check `/health` |
| `emotion-worker` | worker | starter | Celery, `--concurrency=1`; loads its own copy of the model |
| `emotion-redis` | redis | starter | Broker + result backend + SHAP cache, `allkeys-lru` |
| `emotion-db` | postgres | basic-256mb | — |

**`plan: starter`, not `free`, is deliberate.** The RoBERTa weights are ~479MB before torch's runtime allocations. A 512MB free instance OOMs during model load; starter (2GB) is the smallest plan this app fits in. The worker needs the same for the same reason.

**Two values must be set by hand after the first deploy** (they are `sync: false` in the blueprint because neither exists until the services are created):

1. **`MODEL_URL`** on both `emotion-api` and `emotion-worker` — where `scripts/fetch_model.sh` downloads the ~479MB artifact from. The weights are gitignored, so they are *not* in the repo Render clones. See that script for accepted formats.
2. **`FRONTEND_ORIGIN`** on `emotion-api` — your Vercel production URL, with no trailing slash.

Build runs `scripts/render_build.sh`, which installs `src/backend/requirements.txt` (already pinned to the same torch/transformers/shap versions as `src/ml/requirements.txt`, so the training-time artifact loads identically) and then fetches the model.

### Frontend → Vercel

`src/frontend/vercel.json` sets a catch-all rewrite to `/index.html` for React Router. Set `VITE_API_URL` to the Render API URL in Vercel's environment variables.

### CORS

`FRONTEND_ORIGIN` on the API must exactly match the Vercel origin — scheme, host, and port, **no trailing slash**. A mismatch shows up as a preflight failure in the browser console and nothing in the API logs.

---

## Detailed source layout

```
src/
├── ml/                          # Training pipeline
│   ├── data_prep.py              # GoEmotions + Amazon Reviews preparation
│   ├── train_roberta.py          # Multi-label RoBERTa fine-tuning
│   ├── evaluate.py               # Per-label threshold tuning + test metrics
│   ├── shap_explainer.py         # SHAP attribution logic (shared with backend)
│   ├── emotion_labels.py         # Canonical label order + business bucket map
│   └── configs/train_config.yaml
├── backend/                     # FastAPI app
│   ├── app/
│   │   ├── main.py               # App factory, lifespan, router registration
│   │   ├── api/routes/           # predict, analyze, explain, reviews, analytics, model_info
│   │   ├── core/                 # config, db, ratelimit, ml_path
│   │   ├── models/               # SQLAlchemy ORM: Review, BatchJob
│   │   ├── schemas/              # Pydantic request/response models
│   │   └── services/             # emotion_model, shap_service, segmentation, cache, celery_app
│   └── tests/                    # 42 tests against the real checkpoint
└── frontend/                    # React + Vite app
    └── src/
        ├── pages/                # Analyze, Dashboard, Batch, Model
        ├── components/           # Charts, table, heatmap + ui/ primitives
        └── lib/                  # api client, emotion/bucket helpers, utils
```

Outside `src/`: `data/` (datasets), `results/ml-artifacts/` (trained weights + metrics, gitignored weights), `docs/` (`MODEL_CARD.md`, `COMPARISON.md`, source papers), `reports/` (review submission materials), `scripts/` (`render_build.sh`, `fetch_model.sh`), `docker-compose.yml`.

`src/ml/emotion_labels.py` is the single source of truth for label order. Changing that order silently invalidates trained checkpoints — see [Design decisions](#design-decisions).

---

## Troubleshooting

**`/api/predict` returns 503 `model not loaded`.** No valid artifact at `MODEL_DIR`. Check `GET /health` — if `model_loaded` is `false`, confirm `results/ml-artifacts/model-v1/` exists and contains the weights plus `thresholds.json`. Backend logs record the load failure with a stack trace at startup.

**Predictions work but nothing appears in the Dashboard.** Check `db_ready` on `/health`. The app starts without Postgres by design; analysis succeeds and persistence fails silently in that state.

**Batch jobs stay `pending` forever.** The Celery worker isn't running or can't reach Redis. Start it with `celery -A app.services.celery_app worker --loglevel=info` from `backend/`, and verify `CELERY_BROKER_URL` points at a reachable Redis.

**CORS errors in the browser console.** `FRONTEND_ORIGIN` must exactly match the origin the frontend is served from — scheme, host, and port, with no trailing slash. The default allows only `http://localhost:5173`.

**First `/api/explain` call is slow.** Expected — SHAP takes ~2–5 seconds per uncached text. The result is cached in Redis for 7 days, so repeat calls on the same text return quickly. Lower `SHAP_MAX_EVALS` to trade fidelity for speed.

**HTTP 429.** Rate limits are per endpoint: `/explain` 10/min, `/predict` and `/analyze` 120/min, `/batch` 5 per 5 min. The limiter is in-process, so limits apply per worker, not globally.

**Render deploy OOMs during startup.** The service is on the free plan. Model load needs ~2GB; use `starter` or higher.

**New indexes don't appear after a schema change.** `create_all` only creates missing tables, never alters existing ones. Run `docker compose down -v` to drop the volume, or add the indexes by hand.

---

## Security posture

**This app has no authentication.** Every endpoint is open to anyone who can reach it, which is fine for a local demo and not fine on a public address. Read this before exposing it:

- **No auth on any route.** Add authentication before putting this anywhere real.
- **Rate limits are in-process**, so they are per worker rather than global. A real deployment needs Redis-backed limits and a proxy in front.
- **CSV uploads are capped** at 5MB, 5000 rows, and 5000 characters per review. Uploads are read in 1MB chunks and rejected mid-stream on overflow, because an unbounded `await file.read()` would pull the whole file into memory before anything could reject it. Each row also costs a model inference in the worker, so these bounds are about worker time as much as memory.
- **Postgres and Redis are bound to `127.0.0.1`** in `docker-compose.yml`. Published on `0.0.0.0` they are a database with default credentials and a cache with none. On Render, Redis has an empty `ipAllowList` so only in-network services reach it.
- **`/api/model/info` serves evaluation metrics unauthenticated.** Harmless here, deliberate to note.

---

## Status and limitations

This is a capstone deliverable, not a production system. Known gaps, stated plainly:

- **No authentication.** Every endpoint is open. Rate limits and upload caps are an interim bound against trivial resource exhaustion, not an access control.
- **No database migrations.** Schema changes require dropping and recreating tables.
- **Trained on GoEmotions (Reddit comments), applied to product reviews.** This is a domain shift. The Amazon Reviews sample is used for inference and demonstration only, never for fine-tuning, so accuracy on review text is expected to be lower than the reported GoEmotions test metrics — but this is a reasoned expectation, **not a measurement**. No labeled product-review emotion set exists to check it against.
- **Three labels score 0.000 F1** (`grief`, `pride`, `relief`). They are almost certainly data-starved, but that has not been *tested*: training used unweighted `BCEWithLogitsLoss`, under which predicting "absent" for a rare label is nearly free, so imbalance and genuine unlearnability are currently confounded. `use_pos_weight: true` in `configs/train_config.yaml` enables per-label positive weighting to separate the two. The ablation ships but has not been run.
- **Metrics come from a single seed** and are not re-run for variance — which matters most for the labels with single-digit support.
- **SHAP faithfulness is assumed, not measured.** No human evaluation of whether the attributions match human judgment of what drove the emotion.
- **Only the first 128 tokens are read.** Reported via the `truncated` flag, but a long review whose complaint sits in the last paragraph can be scored on its polite opening alone.
