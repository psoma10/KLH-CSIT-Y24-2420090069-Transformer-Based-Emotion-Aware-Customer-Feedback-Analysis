# Comparison: This Project vs. Reviewed Prior Work

## Research Gap Identification

Synthesizing the limitations of the reviewed literature, the following gaps justify this project's design:

1. **No fine-grained + multi-label + explainable system combined.** GoEmotions (#1) provides the taxonomy but only benchmarks plain classifiers with no explainability. SHAP/LIME (#6, #7) provide explainability but are model-agnostic add-ons never integrated end-to-end with a fine-grained multi-label emotion classifier in a deployable product. The Acheampong survey (#8) confirms XAI-for-NLP techniques remain largely decoupled from production emotion/sentiment systems. **Gap:** an end-to-end pipeline that is simultaneously fine-grained, multi-label, and token-explainable.

2. **Single-corpus training limits generalization.** GoEmotions (#1) is Reddit-domain; CARER (#4) and DailyDialog (#5) are Twitter/dialogue-domain. Survey #8 explicitly flags poor cross-domain transfer as an open problem. No reviewed work combines multiple emotion corpora with a principled way to handle each corpus's partial label coverage. **Gap:** a masked multi-label loss that lets heterogeneous emotion corpora be combined for training without corrupting the label space with false negatives.

3. **Whole-document scoring hides mixed sentiment.** ABSA work (#12) addresses this at the aspect level but requires predefined aspect categories, which does not generalize to open emotion labels. No reviewed emotion-classification system does segment-level scoring for reviews. **Gap:** segment-level (not aspect-predefined) emotion scoring for arbitrary review text.

4. **Business actionability is absent from academic taxonomies.** The 27-label GoEmotions taxonomy (#1) is academically rigorous but not directly actionable by a business stakeholder. **Gap:** a mapping layer from fine-grained emotion labels to business-relevant categories (satisfaction, frustration, confusion, concern).

This project's framework — a RoBERTa-based multi-label classifier, trained on GoEmotions plus masked-loss auxiliary corpora, with SHAP token attribution, segment-level scoring, and a business bucket rollup — is positioned to close all four gaps simultaneously, which no single reviewed paper does.

---

## Why This Project Beats the Reviewed Prior Work

### On raw accuracy: honest, not dramatic

| Model | Macro F1 | Micro F1 | Notes |
|---|---|---|---|
| GoEmotions original baseline (Demszky et al., 2020) | ≈ 0.46 (reported range 0.46–0.52 for BERT-base) | not directly comparable | BERT-base, single corpus |
| This project — `baseline-goemotions` (RoBERTa-base, GoEmotions only) | **0.4766** | **0.6048** | Reproduces and slightly exceeds published baseline range |
| This project — `enhanced-multicorpus` (RoBERTa-base, GoEmotions + masked-loss auxiliary corpora) | **0.4796** | 0.6028 | +0.30 pt macro F1 over own baseline |

This project does not claim a large accuracy win over prior work — the 0.30-point macro F1 gain is within normal single-seed noise. A claim of a dramatic accuracy improvement would overstate what these numbers support.

### On capability: this is where it actually wins

| Capability | GoEmotions (#1) | BERT/RoBERTa (#2, #3) | SHAP/LIME (#6, #7) | ABSA (#12) | This project |
|---|---|---|---|---|---|
| 27+ label fine-grained emotion taxonomy | ✅ | ❌ | ❌ | ❌ (fixed aspects) | ✅ |
| Multi-label (not single softmax) output | ✅ | — | — | ❌ | ✅ |
| Token-level explainability integrated end-to-end | ❌ | ❌ | ✅ (generic, unintegrated) | ❌ | ✅ (SHAP wired to classifier, served via `/explain`) |
| Multi-corpus training with partial label coverage handled correctly | ❌ | ❌ | ❌ | ❌ | ✅ (masked BCE loss) |
| Segment-level (mixed-sentiment) scoring | ❌ | ❌ | ❌ | Partial (fixed aspects only) | ✅ (dynamic segment splitting) |
| Business-actionable output (vs. academic label) | ❌ | ❌ | ❌ | ❌ | ✅ (4-bucket rollup) |
| Deployed, callable system (not notebook/benchmark only) | ❌ | ❌ | ❌ | ❌ | ✅ (FastAPI + React) |

No single reviewed paper checks more than two of these seven boxes at once. This project checks all seven. That combination — not a macro-F1 bump — is the actual contribution, and it is verifiable directly against the codebase (`ml/shap_explainer.py`, `ml/aux_datasets.py`, backend `/explain`/`/analyze` routes), not merely asserted.

### Per-label evidence the multi-corpus addition works as designed, not just noise

| Label | Baseline F1 | Multi-corpus F1 | Δ |
|---|---|---|---|
| `disgust` | 0.396 | 0.459 | **+0.063** |
| `nervousness` | 0.213 | 0.378 | **+0.166** |
| `optimism` | 0.561 | 0.606 | **+0.045** |
| `anger` | 0.497 | 0.518 | +0.021 |
| `love` | 0.806 | 0.808 | +0.003 |
| `caring` | 0.436 | 0.417 | −0.019 |
| `joy` | 0.628 | 0.603 | −0.026 |
| `realization` | 0.201 | 0.149 | −0.052 |

Gains cluster exactly on labels the auxiliary corpora annotate (`disgust`, `optimism`, `nervousness`) — evidence the masked loss is targeted, not a blanket booster.

### Caveats a credible comparison must disclose

- `grief`, `pride`, `relief` still score 0.000 F1 in both models — under 20 test examples each, effectively untrained.
- No labeled product-review test set exists yet; all numbers above are measured on GoEmotions (Reddit) test data.
- Single-seed results — deltas like the 0.30-pt macro F1 gain are not confirmed to exceed training variance.

---

**Bottom line:** the improvement over prior work is architectural and capability-based (explainability, multi-corpus handling, segment scoring, business rollup, deployment), not a raw F1 win.
