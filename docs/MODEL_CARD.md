# Model card — `model-v1`

Fine-tuned RoBERTa multi-label emotion classifier used by the API and the
Celery batch worker. The weights themselves are not in git (≈480MB, see
`.gitignore`); this file records what was trained and how well it scored so
the numbers survive independently of the artifact.

## What it is

| | |
|---|---|
| Base model | `roberta-base` |
| Task | Multi-label emotion classification (sigmoid, not softmax) |
| Labels | 28 — the 27 GoEmotions emotions plus `neutral` |
| Training data | GoEmotions |
| Epochs | 4 |
| Max sequence length | 128 tokens |
| Artifact location | `ml/artifacts/model-v1/` |

Because it is multi-label, a review can carry several emotions at once, and
each label has its own decision threshold tuned on the validation split
(`thresholds.json`) rather than a single shared cutoff.

## Headline results (GoEmotions test split)

| Metric | Score |
|---|---|
| Macro F1 | **0.475** |
| Micro F1 | **0.600** |
| Validation loss | 0.085 |

Macro F1 weights every emotion equally, so it is dragged down by the rare
ones; micro F1 weights by frequency. Both are in line with published
`roberta-base` GoEmotions baselines (macro F1 ≈ 0.46–0.52), so this behaves
like a correctly-trained model of its size rather than an underfit one.

## Per-label performance

Sorted by F1. `Threshold` is the tuned per-label cutoff actually used at
inference time.

| Label | F1 | Precision | Recall | Support | Threshold |
|---|---|---|---|---|---|
| `gratitude` | 0.919 | 0.954 | 0.886 | 352 | 0.55 |
| `amusement` | 0.821 | 0.746 | 0.913 | 264 | 0.30 |
| `love` | 0.814 | 0.762 | 0.874 | 238 | 0.35 |
| `admiration` | 0.717 | 0.713 | 0.720 | 504 | 0.50 |
| `remorse` | 0.707 | 0.610 | 0.839 | 56 | 0.40 |
| `neutral` | 0.688 | 0.598 | 0.809 | 1787 | 0.20 |
| `joy` | 0.625 | 0.580 | 0.677 | 161 | 0.30 |
| `fear` | 0.620 | 0.532 | 0.744 | 78 | 0.20 |
| `curiosity` | 0.580 | 0.455 | 0.799 | 284 | 0.20 |
| `optimism` | 0.571 | 0.624 | 0.527 | 186 | 0.35 |
| `surprise` | 0.556 | 0.552 | 0.560 | 141 | 0.35 |
| `desire` | 0.526 | 0.580 | 0.482 | 83 | 0.20 |
| `sadness` | 0.516 | 0.448 | 0.609 | 156 | 0.25 |
| `anger` | 0.495 | 0.490 | 0.500 | 198 | 0.40 |
| `embarrassment` | 0.491 | 0.812 | 0.351 | 37 | 0.20 |
| `excitement` | 0.486 | 0.468 | 0.505 | 103 | 0.25 |
| `disapproval` | 0.436 | 0.357 | 0.558 | 267 | 0.20 |
| `approval` | 0.428 | 0.408 | 0.450 | 351 | 0.25 |
| `caring` | 0.428 | 0.575 | 0.341 | 135 | 0.50 |
| `confusion` | 0.418 | 0.324 | 0.588 | 153 | 0.20 |
| `disgust` | 0.405 | 0.756 | 0.276 | 123 | 0.50 |
| `annoyance` | 0.392 | 0.308 | 0.537 | 320 | 0.20 |
| `disappointment` | 0.292 | 0.227 | 0.411 | 151 | 0.15 |
| `realization` | 0.244 | 0.382 | 0.179 | 145 | 0.20 |
| `nervousness` | 0.118 | 0.107 | 0.130 | 23 | 0.10 |
| `grief` | 0.000 | 0.000 | 0.000 | 6 | 0.05 |
| `pride` | 0.000 | 0.000 | 0.000 | 16 | 0.05 |
| `relief` | 0.000 | 0.000 | 0.000 | 11 | 0.05 |
## Known limitations

Worth reading before trusting an individual prediction.

- **Three labels score 0.000 F1** — `grief` (6 test examples), `pride` (16),
  and `relief` (11). The model never successfully predicts them. With that
  little support they are effectively untrained, and their tuned thresholds
  (0.05) reflect the optimizer trying to salvage recall that isn't there.
  Treat any of these three appearing in output as noise.
- **`nervousness` (F1 0.118) is barely better.** Below roughly 0.30 F1 —
  which also covers `realization` and `disappointment` — predictions are
  closer to a coin flip than a signal.
- **The reliable labels are the frequent, lexically distinctive ones**:
  `gratitude` (0.919), `amusement` (0.821), `love` (0.814), `admiration`
  (0.717). These carry the demo.
- **Low-precision labels over-fire.** `disappointment` (precision 0.227) and
  `annoyance` (0.308) flag roughly three texts for every one they get right,
  so bucket counts skew toward frustration.
- **Only the first 128 tokens are read.** Longer reviews are truncated, and
  the API reports this with a `truncated` flag on both `/predict` and
  `/explain` responses — a long review whose complaint sits in the last
  paragraph can be scored on its polite opening alone.
- **Domain shift.** Trained on Reddit comments, applied here to product
  reviews. GoEmotions phrasing is shorter and more conversational than a
  typical review, so scores on long-form review text are less trustworthy
  than these test-set numbers suggest.
- **`top_label` ignores thresholds.** It is the argmax, so it is always
  populated even when no label clears its own cutoff and
  `predicted_labels` comes back empty. Check `predicted_labels` when you
  need "the model is actually confident about something".

## Not yet tried

The three 0.000-F1 labels are described above as data-starved, and with 6-16
test examples each that is very likely true — but it has not been *tested*.
Training used a plain unweighted `BCEWithLogitsLoss`, under which predicting
"absent" for a rare label is nearly free, so imbalance and genuine
unlearnability are currently confounded.

`use_pos_weight: true` in `configs/train_config.yaml` enables per-label
positive weighting (`neg/pos`, capped) to separate the two. If `grief`,
`pride`, and `relief` stay at 0.000 with weighting on, "not enough data" is an
evidence-backed claim rather than an assumption; if they move, the current
numbers were leaving something on the table. Expect macro F1 to move in both
directions — rare-label recall up, frequent-label precision down.

Also untested: performance on the actual target domain. Every number here is
from GoEmotions (Reddit comments); no labeled product-review set exists, so
the domain-shift caveat above is a reasoned expectation, not a measurement.
Metrics come from a single seed and are not re-run for variance, which matters
most for the labels with single-digit support.

## Datasets

`model-v1` was trained on **GoEmotions alone**. Everything quoted above comes
from that single corpus; the auxiliary sources below are wired up but were not
used for these weights, and `aux_datasets` is empty by default so these numbers
stay reproducible from the committed config.

| Dataset | Labels | Role | Link |
|---|---|---|---|
| GoEmotions (simplified) | 27 emotions + neutral | training, val, test | [HF](https://huggingface.co/datasets/google-research-datasets/go_emotions) |
| dair-ai/emotion (CARER) | 6, single-label | auxiliary train | [HF](https://huggingface.co/datasets/dair-ai/emotion) |
| DailyDialog | 7, single-label | auxiliary train | [HF](https://huggingface.co/datasets/roskoN/dailydialog) |
| amazon_polarity | binary sentiment | demo/inference only, never training | [HF](https://huggingface.co/datasets/fancyzhx/amazon_polarity) |

Citations: GoEmotions [arXiv:2005.00547](https://arxiv.org/abs/2005.00547) ·
CARER [D18-1404](https://aclanthology.org/D18-1404/) ·
DailyDialog [arXiv:1710.03957](https://arxiv.org/abs/1710.03957)

Each auxiliary corpus annotates only a subset of the 28 labels, and a label it
does not annotate is *unknown*, not negative — a `dair-ai` row tagged `sadness`
says nothing about whether `gratitude` applies, because annotators were never
asked. Rows therefore carry a coverage mask and the loss is computed only over
masked positions, so unannotated labels contribute no gradient. `ml/
aux_datasets.py` holds the mappings and the reasoning behind each; source
labels with no GoEmotions counterpart are dropped rather than forced onto a
near-miss. SemEval-2018 Task 1 was evaluated as a third source and excluded:
its only Parquet mirror ships text with stopwords stripped and casing folded,
a register mismatch that would add noise rather than signal.

Note what this can and cannot fix: these corpora add `anger`, `fear`, `joy`,
`sadness`, `surprise`, `love`, `disgust`, `optimism` and `neutral`. None
contains `grief`, `pride`, or `relief`, so the three 0.000-F1 labels are not
addressed by more data of this kind — `use_pos_weight` remains the lever there.
All are tweets or dialogue, so none closes the product-review domain gap either.

## Reproducing

```bash
cd ml
python data_prep.py --goemotions --out data/
python train_roberta.py --config configs/train_config.yaml
python evaluate.py --config configs/train_config.yaml   # writes thresholds.json + test_metrics.json
```

To train on the auxiliary corpora as well — a different experiment, not a
reproduction of the numbers above:

```bash
python data_prep.py --goemotions --aux --out data/
# then set data.aux_datasets in configs/train_config.yaml, e.g.
#   aux_datasets: [dair_emotion, daily_dialog]
python train_roberta.py --config configs/train_config.yaml
python evaluate.py --config configs/train_config.yaml
```

Auxiliary data joins the **train split only**; validation and test stay pure
GoEmotions so macro F1 means the same thing across configs and is always
measured on fully-annotated rows.

`evaluate.py` is what produces the tuned thresholds and the metrics quoted
above; training alone does not.
