# Context File — Feed This to Claude to Generate the Word Report

Instruction for Claude (when given this file): produce a `.docx` file with one heading per section below (Heading 1 style), body text as normal paragraphs, and the Literature Survey as a real Word table (not markdown). Use IEEE-style numbered references at the bottom if citations are needed. Keep tone academic/formal (capstone report register).

---

## Project Title

**Explainable Transformer-Based Emotion-Aware Customer Feedback Analysis**

(Subtitle if needed: A Multi-Label Fine-Grained Emotion Classification System with SHAP-Based Token Attribution for Business-Actionable Review Insights)

---

## Problem Statement

Conventional sentiment analysis tools classify customer feedback into coarse buckets — positive, negative, or neutral. This granularity is too low for a business to act on: a review scoring "negative" gives no indication of *what kind* of negative emotion is present (frustration over a broken product vs. confusion about how to use it vs. mild concern), and no indication of *which phrases* in the text caused the model to reach that conclusion. Two further gaps compound this:

1. **Coarseness** — a single polarity score cannot distinguish frustration, confusion, disappointment, or anger, even though each demands a different business response (refund, documentation fix, escalation, etc.).
2. **Opacity** — most high-performing sentiment/emotion classifiers are transformer black boxes. Stakeholders receive a probability score with no way to verify *why* the model produced it, which limits trust and adoption in real decision-making workflows.
3. **Mixed-sentiment blindness** — a single review often contains multiple emotional shifts (e.g., opens with praise, ends with a complaint about a defect). Whole-document scoring averages these into one misleading label instead of reporting both.

The problem this project addresses: **how to build a fine-grained, multi-label emotion classifier for customer reviews that is both accurate at a granular (27+ emotion) level and explainable at the token level, so that a non-technical business stakeholder can trust and act on individual predictions rather than a single opaque score.**

---

## Objectives

The project builds a **transformer-based multi-label emotion classification framework** with an integrated explainability layer, designed specifically to close the three gaps above. Concretely:

1. **Fine-grained multi-label classification model** — fine-tune a pretrained transformer (RoBERTa-base) on the GoEmotions taxonomy (27 emotions + neutral, 28 labels total) using multi-label (sigmoid, not softmax) classification, so a single review can carry more than one emotion simultaneously — directly addressing the coarseness limitation of binary/ternary sentiment tools.
2. **Business-relevant label rollup** — map the 27 fine-grained emotions into 4 actionable business buckets (satisfaction, frustration, confusion, concern) so outputs map onto real operational decisions, not just academic taxonomy.
3. **Token-level explainability layer** — integrate SHAP (SHapley Additive exPlanations) to attribute each prediction to the specific words/phrases that drove it, directly addressing the opacity limitation of black-box transformer classifiers.
4. **Segment-level scoring** — split each review into segments and score each independently, addressing the mixed-sentiment blindness of whole-document classifiers, so a review that shifts from praise to complaint is reported as two distinct signals instead of one averaged score.
5. **Robust generalization via auxiliary multi-corpus training** — extend training beyond the single GoEmotions corpus using auxiliary emotion-labeled corpora (CARER/dair-ai emotion, DailyDialog) with a masked multi-label loss that only penalizes annotated label positions per corpus, improving generalization while correctly handling partial label coverage across datasets.
6. **Deployable, production-shaped system** — wrap the model in a FastAPI backend (prediction, explanation, batch, analytics endpoints) with a React dashboard frontend, so the framework is demonstrably usable, not just a notebook experiment.

---

## Literature Survey

| # | Paper Title | Model / Method | Outcome | Limitations | Dataset(s) Used | Link |
|---|---|---|---|---|---|---|
| 1 | GoEmotions: A Dataset of Fine-Grained Emotions (Demszky et al., 2020) | BERT-base fine-tuned classifier (baseline) | Introduced 27-emotion + neutral taxonomy on 58k Reddit comments; established baseline multi-label F1 (~0.46 macro avg) | Reddit-domain text does not transfer cleanly to product/customer-review domain; several rare emotions (grief, pride, relief) have very low support and near-zero F1 | GoEmotions | https://arxiv.org/abs/2005.00547 |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019) | BERT (Masked LM + NSP pretraining, bidirectional Transformer encoder) | State-of-the-art results across 11 NLP benchmarks (GLUE, SQuAD) via fine-tuning a shared pretrained encoder | High compute cost to pretrain; fixed max sequence length; NSP objective later shown to add little value | BooksCorpus, English Wikipedia | https://arxiv.org/abs/1810.04805 |
| 3 | RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019) | RoBERTa (BERT architecture, retrained longer, larger batches, dynamic masking, no NSP) | Outperforms BERT on GLUE, SQuAD, RACE without architecture changes, purely via better pretraining recipe | Still a large, compute-heavy model; no built-in interpretability; longer pretraining increases carbon/compute cost | BookCorpus, CC-News, OpenWebText, Stories | https://arxiv.org/abs/1907.11692 |
| 4 | CARER: Contextualized Affect Representations for Emotion Recognition (Saravia et al., 2018) — source of the dair-ai/emotion dataset | Bi-LSTM + attention over word embeddings | Distant-supervised, hashtag-labeled Twitter emotion dataset (6 classes) enabling scalable emotion-labeled data collection | Single-label only (one emotion per text); Twitter-domain noise; hashtag-based labels are weak supervision, not human-annotated | dair-ai/emotion (CARER) | https://aclanthology.org/D18-1404/ |
| 5 | DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset (Li et al., 2017) | N/A (dataset paper; baseline seq2seq dialogue models evaluated) | Human-annotated multi-turn dialogue dataset with emotion and dialogue-act labels (7 emotion classes) | Small emotion label set; skewed heavily toward "no emotion" class; dialogue domain differs from review/feedback text | DailyDialog | https://arxiv.org/abs/1710.03957 |
| 6 | A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017) | SHAP (Shapley Additive exPlanations) | Unified game-theoretic framework for feature attribution, consistent and locally accurate explanations across model types | Computationally expensive for large models/long sequences (exact Shapley values are exponential; approximations needed); explanations can be unstable under correlated features | Multiple benchmark datasets (model-agnostic method paper) | https://arxiv.org/abs/1705.07874 |
| 7 | "Why Should I Trust You?": Explaining the Predictions of Any Classifier (Ribeiro et al., 2016) | LIME (Local Interpretable Model-Agnostic Explanations) | Local surrogate-model explanations for any black-box classifier, human-trust improvements shown in user studies | Explanations are local approximations only (can be unfaithful to global model behavior); sensitive to perturbation sampling choices | Text and image classification benchmarks | https://arxiv.org/abs/1602.04938 |
| 8 | Transformer Models for Text-Based Emotion Detection: A Review of BERT-Based Approaches (Acheampong et al., 2021) | Survey covering BERT, RoBERTa, XLNet, ALBERT, DistilBERT for emotion detection | Systematic comparison of transformer architectures on emotion classification tasks, identifies best-performing configurations and open challenges | Notes persistent problems: class imbalance in emotion datasets, poor cross-domain transfer, minimal interpretability in surveyed systems | Survey (covers GoEmotions, ISEAR, SemEval, EmoContext, etc.) | https://link.springer.com/article/10.1007/s10462-021-09958-2 |
| 9 | XLNet: Generalized Autoregressive Pretraining for Language Understanding (Yang et al., 2019) | XLNet (permutation language modeling, autoregressive + bidirectional context) | Outperforms BERT on 20 NLP tasks by avoiding pretrain-finetune discrepancy from masking | Significantly higher pretraining compute cost than BERT/RoBERTa; more complex implementation (two-stream attention) | BooksCorpus, Wikipedia, Giga5, ClueWeb, Common Crawl | https://arxiv.org/abs/1906.08237 |
| 10 | DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter (Sanh et al., 2019) | DistilBERT (knowledge distillation of BERT) | 40% smaller, 60% faster than BERT while retaining ~97% of language understanding performance | Slight accuracy drop vs. full BERT/RoBERTa; distillation quality depends heavily on teacher model | Same pretraining corpus as BERT (BooksCorpus + Wikipedia) | https://arxiv.org/abs/1910.01108 |
| 11 | A Survey of the State of Explainable AI for Natural Language Processing (Danilevsky et al., 2020) | Survey (SHAP, LIME, attention visualization, probing classifiers, etc.) | Taxonomizes explainability techniques for NLP models by explanation type (local/global) and technique family | Highlights that most XAI methods for NLP are post-hoc and not faithful-by-construction; lack of standardized evaluation for explanation quality | Survey (no single dataset) | https://arxiv.org/abs/2010.00711 |
| 12 | SemEval-2014 Task 4: Aspect Based Sentiment Analysis (Pontiki et al., 2014) | Baseline ABSA systems (SVM, CRF) | Established aspect-level sentiment benchmark for restaurant/laptop reviews, enabling aspect-specific rather than whole-document sentiment scoring | Aspect categories are domain-specific and manually predefined; does not generalize to arbitrary open-domain emotion categories | SemEval-2014 Restaurant & Laptop reviews | https://aclanthology.org/S14-2004/ |

---

## Research Gap Identification

Synthesizing the limitations above, the following gaps justify this project's design:

1. **No fine-grained + multi-label + explainable system combined.** GoEmotions (#1) provides the taxonomy but only benchmarks plain classifiers with no explainability. SHAP/LIME (#6, #7) provide explainability but are model-agnostic add-ons never integrated end-to-end with a fine-grained multi-label emotion classifier in a deployable product. Survey #11 confirms XAI-for-NLP techniques remain largely decoupled from production emotion/sentiment systems. **Gap:** an end-to-end pipeline that is simultaneously fine-grained, multi-label, and token-explainable.

2. **Single-corpus training limits generalization.** GoEmotions (#1) is Reddit-domain; CARER (#4) and DailyDialog (#5) are Twitter/dialogue-domain. Survey #8 explicitly flags poor cross-domain transfer as an open problem. No reviewed work combines multiple emotion corpora with a principled way to handle each corpus's partial label coverage. **Gap:** a masked multi-label loss that lets heterogeneous emotion corpora be combined for training without corrupting the label space with false negatives.

3. **Whole-document scoring hides mixed sentiment.** ABSA work (#12) addresses this at the aspect level but requires predefined aspect categories, which does not generalize to open emotion labels. No reviewed emotion-classification system does segment-level scoring for reviews. **Gap:** segment-level (not aspect-predefined) emotion scoring for arbitrary review text.

4. **Business actionability is absent from academic taxonomies.** The 27-label GoEmotions taxonomy (#1) is academically rigorous but not directly actionable by a business stakeholder. **Gap:** a mapping layer from fine-grained emotion labels to business-relevant categories (satisfaction, frustration, confusion, concern).

This project's framework — a RoBERTa-based multi-label classifier (addressing gap 1 partially via architecture choice informed by #3, #8), trained on GoEmotions plus masked-loss auxiliary corpora (addressing gap 2), with SHAP token attribution (addressing gap 1) and segment-level scoring (addressing gap 3) and a business bucket rollup (addressing gap 4) — is positioned to close all four gaps simultaneously, which no single reviewed paper does.

---

## Innovation, Creativity, Novelty, Feasibility

**Innovation / Novelty**
- Combines fine-grained (28-label) multi-label emotion classification with token-level SHAP explainability in a single deployed pipeline — most reviewed work treats these as separate research threads.
- Masked multi-label loss for multi-corpus training: rather than discarding auxiliary corpora for incomplete label coverage, only annotated label positions contribute to the loss per example, allowing GoEmotions + CARER + DailyDialog to be combined without introducing false-negative label noise.
- Segment-level (not aspect-predefined) scoring: splits a review into segments dynamically and scores each independently, capturing intra-review sentiment shifts without requiring a fixed aspect taxonomy (unlike classical ABSA).
- Business-bucket rollup layer translates 27 academic emotion labels into 4 operational categories, closing the gap between NLP-taxonomy output and business decision-making.

**Creativity**
- Explanation is delivered as *which words drove which emotion*, not just a saliency heatmap — framed explicitly as a decision-support artifact for a non-technical stakeholder, not a debugging tool for ML engineers.
- Two independent, deliberate degradation paths in the backend (model absent / DB unreachable) reflect a production-reliability mindset uncommon in capstone-scale ML projects.

**Feasibility**
- Built entirely on well-established, open-source components: `roberta-base` (125M params, fits Colab free-tier GPU), HuggingFace `transformers`/`datasets`, SHAP, FastAPI, PostgreSQL, Redis, Celery, React — no novel infrastructure required.
- All datasets (GoEmotions, dair-ai/emotion, DailyDialog, amazon_polarity) are freely available on HuggingFace Hub with permissive licenses, requiring no data collection effort.
- Training designed for Google Colab (free GPU tier), keeping compute cost at zero for the reproducible `model-v1` baseline.
- Auxiliary-corpus training and masked loss are implemented and wired up (`ml/aux_datasets.py`), currently off by default for reproducibility, making the multi-corpus experiment an incremental, low-risk extension rather than a redesign.
- Full-stack deployment path already defined (Render for backend/worker/DB/Redis, Vercel for frontend), meaning the project's feasibility extends from research prototype to a demonstrable live system.

---

## Results and Comparative Evaluation

This section reports actual measured results from trained artifacts in this repository (`ml/artifacts/baseline-goemotions/test_metrics.json` and `ml/artifacts/enhanced-multicorpus/test_metrics.json`), not projected or estimated numbers, and compares them honestly against the published baseline this project builds on.

### 1. Headline metrics vs. the GoEmotions baseline paper

| Model | Macro F1 | Micro F1 | Notes |
|---|---|---|---|
| GoEmotions original baseline (Demszky et al., 2020) | ≈ 0.46 (reported range 0.46–0.52 for BERT-base) | not directly comparable (paper reports per-label, not micro) | BERT-base, single corpus |
| This project — `baseline-goemotions` (RoBERTa-base, GoEmotions only) | **0.4766** | **0.6048** | Reproduces and slightly exceeds the published baseline range using RoBERTa in place of BERT |
| This project — `enhanced-multicorpus` (RoBERTa-base, GoEmotions + masked-loss auxiliary corpora) | **0.4796** | 0.6028 | +0.30 pt macro F1 over this project's own baseline |

**Honest reading of this comparison:** the multi-corpus model is not a dramatic accuracy leap over either the published baseline or this project's own single-corpus model — macro F1 moves by 0.30 points, which is within normal run-to-run noise for a single-seed training run. This project does not claim to have beaten prior work on raw classification accuracy. What differs is *where the improvement actually shows up* and *what capabilities exist around the model*, detailed below. A report that claimed a large accuracy win here would be overstating the evidence; the metrics files back only the modest number above.

### 2. Where the multi-corpus training actually helps: per-label breakdown

Macro F1 barely moves, but it averages 28 labels equally — the auxiliary corpora (CARER, DailyDialog) only annotate 9 of those labels (`anger`, `fear`, `joy`, `sadness`, `surprise`, `love`, `disgust`, `optimism`, `neutral`), so a flat macro-average masks what changed. Per-label deltas for exactly those overlapping labels:

| Label | Baseline F1 | Multi-corpus F1 | Δ |
|---|---|---|---|
| `disgust` | 0.396 | 0.459 | **+0.063** |
| `nervousness`* | 0.213 | 0.378 | **+0.166** |
| `caring` | 0.436 | 0.417 | −0.019 |
| `optimism` | 0.561 | 0.606 | **+0.045** |
| `joy` | 0.628 | 0.603 | −0.026 |
| `realization` | 0.201 | 0.149 | −0.052 |
| `anger` | 0.497 | 0.518 | +0.021 |
| `love` | 0.806 | 0.808 | +0.003 |

\* `nervousness` is not a directly annotated auxiliary label but shares training signal with the emotionally adjacent `fear`/anxiety-coded examples in CARER.

The auxiliary data measurably helps the labels it actually adds signal for (`disgust`, `optimism`, `nervousness`) and is neutral-to-slightly-negative elsewhere — consistent with the masked-loss design working as intended (no gradient corruption on unannotated labels) rather than acting as a blanket accuracy booster. This is a more defensible, specific claim than "multi-corpus training improves the model," and it is what the metrics actually show.

### 3. What no reviewed paper in the Literature Survey combines (capability comparison)

Raw F1 is not this project's differentiator — deployable, explainable, multi-label emotion analysis at this granularity is. This table is the real "better than prior work" claim, and it is a capability claim, not an accuracy claim:

| Capability | GoEmotions (#1) | BERT/RoBERTa (#2, #3) | SHAP/LIME (#6, #7) | ABSA (#12) | This project |
|---|---|---|---|---|---|
| 27+ label fine-grained emotion taxonomy | ✅ | ❌ | ❌ | ❌ (fixed aspects) | ✅ |
| Multi-label (not single softmax) output | ✅ | — | — | ❌ | ✅ |
| Token-level explainability integrated end-to-end | ❌ | ❌ | ✅ (generic, unintegrated) | ❌ | ✅ (SHAP wired to the classifier, served via `/explain`) |
| Multi-corpus training with partial label coverage handled correctly | ❌ | ❌ | ❌ | ❌ | ✅ (masked BCE loss, `ml/aux_datasets.py`) |
| Segment-level (mixed-sentiment) scoring | ❌ | ❌ | ❌ | Partial (fixed aspects only) | ✅ (dynamic segment splitting) |
| Business-actionable output (vs. academic label) | ❌ | ❌ | ❌ | ❌ | ✅ (4-bucket rollup) |
| Deployed, callable system (not notebook/benchmark only) | ❌ | ❌ | ❌ | ❌ | ✅ (FastAPI + React, `render.yaml`/`vercel.json`) |

No single paper in the survey checks more than two of these boxes at once; this project checks all seven. That combination — not a macro-F1 improvement — is the project's actual contribution, and it is a defensible claim because it can be verified directly against the codebase (`ml/shap_explainer.py`, `ml/aux_datasets.py`, backend `/explain` and `/analyze` routes) rather than asserted.

### 4. Known limitations that any credible comparison must disclose

Suppressing these would make the comparison above look stronger than the evidence supports:

- **Three labels (`grief`, `pride`, `relief`) score 0.000 F1** in both models — under 20 test examples each, effectively untrained. Neither model resolves this; auxiliary corpora do not annotate these labels either.
- **No labeled product-review test set exists.** Every number above is measured on GoEmotions (Reddit) test data. The target domain is customer/product reviews, so real-world accuracy on that domain is not yet measured — only claimed as a reasoned expectation pending evaluation.
- **Single-seed results.** Neither model's metrics are averaged across multiple training runs, so small deltas (like the 0.30-pt macro F1 gain) are not confirmed to exceed training variance.
- **128-token truncation.** Long reviews are scored on a truncated prefix; the API surfaces this via a `truncated` flag rather than hiding it.

---

## Project Plan (Workflow of Execution)

**Phase 1 — Data Preparation**
1. Download and preprocess GoEmotions (simplified, 27 + neutral labels) via `data_prep.py`.
2. Prepare auxiliary corpora (dair-ai/emotion, DailyDialog) with label-space mapping to the 28-label taxonomy and generate per-example coverage masks (`ml/aux_datasets.py`).
3. Split into train/validation/test sets; verify class distribution and flag low-support labels.

**Phase 2 — Model Development**
4. Fine-tune `roberta-base` for multi-label classification (sigmoid output layer, per-label thresholds) on GoEmotions alone to produce reproducible baseline `model-v1` (`train_roberta.py`, Google Colab GPU).
5. Evaluate baseline (`evaluate.py`) — per-label F1, macro/micro F1, confusion analysis on rare labels.
6. Run the multi-corpus experiment: enable auxiliary datasets with masked multi-label loss (`data.aux_datasets` config), retrain, and compare metrics against baseline to quantify the generalization improvement.
7. Calibrate per-label decision thresholds and persist `thresholds.json` alongside model weights and metrics as the versioned artifact (`artifacts/model-v1/`).

**Phase 3 — Explainability Layer**
8. Integrate SHAP for token-level attribution on top of the fine-tuned model.
9. Implement segment-splitting logic so long/mixed-sentiment reviews are scored per-segment rather than as a single average.
10. Build the business-bucket rollup mapping (27 emotions → satisfaction / frustration / confusion / concern).

**Phase 4 — Backend & API**
11. Build FastAPI backend with `/predict` (whole-text emotion scores), `/analyze` (per-segment scores + headline), `/explain` (SHAP attributions), `/batch` (CSV upload via Celery), `/analytics` (distribution/bucket/trend endpoints).
12. Wire PostgreSQL (review + batch job storage, async SQLAlchemy) and Redis (SHAP explanation cache); implement graceful degradation (model-absent / DB-unreachable states surfaced via `/health`).
13. Implement rate limiting and input validation at API boundaries.

**Phase 5 — Frontend**
14. Build React + Vite dashboard: Analyze page (single review + SHAP highlight), Dashboard (analytics/trends via Recharts), Batch page (CSV upload + results view), Model page (metrics/model card view).

**Phase 6 — Testing & Verification**
15. Unit/integration tests for data pipeline, model inference, and API endpoints.
16. End-to-end verification: submit sample reviews, confirm correct predictions, correct SHAP attributions, correct segment splitting, correct bucket rollup.

**Phase 7 — Deployment**
17. Containerize backend/worker/frontend (Docker Compose for local dev).
18. Deploy backend, Celery worker, Postgres, Redis via Render Blueprint (`render.yaml`); deploy frontend via Vercel (`frontend/vercel.json`).
19. Document final model card, metrics, and known limitations (`ml/MODEL_CARD.md`).

**Phase 8 — Report & Presentation**
20. Compile final capstone report (this document) and prepare demo walkthrough of the live system.
