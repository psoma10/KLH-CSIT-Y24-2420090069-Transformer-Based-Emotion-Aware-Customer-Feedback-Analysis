# datasets/raw

Untouched, original-format copies of the three training corpora, as published
on HuggingFace Hub — pulled straight from the source repos, no preprocessing
applied. This exists alongside `ml/data/` (this project's cleaned/mapped
jsonl, actually used by `train_roberta.py`) so the source data is inspectable
independently of this project's transformations.

## go_emotions/

From [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions).

| File | Split | Rows | Config |
|---|---|---|---|
| `train-00000-of-00001.parquet` | train | 43,410 | `simplified` (27 emotions + neutral, label IDs) |
| `validation-00000-of-00001.parquet` | validation | 5,426 | `simplified` |
| `test-00000-of-00001.parquet` | test | 5,427 | `simplified` |
| `raw-train-00000-of-00001.parquet` | train | 211,225 | `raw` (unaggregated, per-annotator rows, 37 columns incl. Reddit metadata) |

Paper: Demszky et al., 2020 — [arXiv:2005.00547](https://arxiv.org/abs/2005.00547)

## dair_ai_emotion/

From [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (`split` config).

| File | Split | Rows |
|---|---|---|
| `train-00000-of-00001.parquet` | train | 16,000 |
| `validation-00000-of-00001.parquet` | validation | 2,000 |
| `test-00000-of-00001.parquet` | test | 2,000 |

6 single-label emotion classes, Twitter text, hashtag-weak-supervised.
Paper (CARER): Saravia et al., 2018 — [D18-1404](https://aclanthology.org/D18-1404/)

## dailydialog/

From [`roskoN/dailydialog`](https://huggingface.co/datasets/roskoN/dailydialog).
Original zips kept alongside their extracted contents.

| Folder | Split | Files |
|---|---|---|
| `train/` | train | `dialogues_train.txt`, `dialogues_act_train.txt`, `dialogues_emotion_train.txt` |
| `validation/` | validation | same pattern |
| `test/` | test | same pattern |

Multi-turn dialogue, human-annotated, 7 emotion classes (incl. no-emotion).
Paper: Li et al., 2017 — [arXiv:1710.03957](https://arxiv.org/abs/1710.03957)

## Relationship to `ml/data/`

`ml/data/*.jsonl` is this project's preprocessed form — label-space mapped to
the shared 28-label taxonomy, GoEmotions `simplified` config only (not `raw`),
with dair-ai/emotion and DailyDialog carrying coverage masks for the masked
multi-label loss (see `ml/aux_datasets.py`). That is what `train_roberta.py`
actually reads. This `datasets/raw/` folder is reference material — the
untouched source, kept so provenance is checkable without re-downloading.

Not re-fetched here: `amazon_polarity`, used only for the non-training demo
CSV (`data_prep.py --amazon-sample`), not a training corpus.
