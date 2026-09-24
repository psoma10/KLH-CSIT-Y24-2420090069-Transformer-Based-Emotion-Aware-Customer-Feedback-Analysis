"""Prepares training data (GoEmotions + auxiliary corpora) and demo data.

Usage:
    python data_prep.py --goemotions --out data/
    python data_prep.py --aux --out data/                 # all auxiliary corpora
    python data_prep.py --aux dair_emotion --out data/    # just one
    python data_prep.py --amazon-sample --out data/ --n 500

Every corpus is downloaded from HuggingFace as published — nothing here is
hand-authored or locally edited.

Data is fetched as Parquet over HTTP rather than via the `datasets` library.
That drops a heavy dependency from data prep, and `datasets` cannot even be
imported on a Python built without `_lzma`, which is how this project's
interpreter happens to be built. The Parquet exports are HuggingFace's own
canonical conversions, so the rows are identical to what `load_dataset` yields.

Output is JSONL, one object per line:
    {"text": str, "labels": [int], "mask": [int], "source": str}

`labels` are indices into GOEMOTIONS_LABELS. `mask` lists the label indices the
row's source actually annotates — see aux_datasets.py for why that distinction
decides whether auxiliary data helps or hurts.
"""
import argparse
import io
import json
import urllib.request
from pathlib import Path

from emotion_labels import NUM_LABELS

PARQUET_API = "https://huggingface.co/api/datasets/{hf_id}/parquet"


def _parquet_urls(hf_id: str, config: str, split: str) -> list[str]:
    """Resolves a dataset split to its Parquet shard URLs."""
    with urllib.request.urlopen(PARQUET_API.format(hf_id=hf_id), timeout=60) as r:
        index = json.load(r)
    if config not in index:
        raise ValueError(f"{hf_id}: no config {config!r}; has {list(index)}")
    if split not in index[config]:
        raise ValueError(
            f"{hf_id}/{config}: no split {split!r}; has {list(index[config])}"
        )
    return index[config][split]


def _read_split(hf_id: str, config: str, split: str) -> list[dict]:
    """Downloads and concatenates every Parquet shard for one split."""
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for url in _parquet_urls(hf_id, config, split):
        with urllib.request.urlopen(url, timeout=300) as r:
            payload = r.read()
        rows.extend(pq.read_table(io.BytesIO(payload)).to_pylist())
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def prepare_goemotions(out_dir: Path) -> None:
    """Writes the GoEmotions splits — the 28-label backbone.

    `labels` already arrives as a list of ints per example (the corpus is
    multi-label: a comment can be both "joy" and "gratitude"), so no remapping
    is needed. The mask covers all 28 because GoEmotions annotates all 28;
    written explicitly so every row in the mixed corpus has one shape.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    full_mask = list(range(NUM_LABELS))

    for split in ("train", "validation", "test"):
        raw = _read_split("google-research-datasets/go_emotions", "simplified", split)
        rows = [
            {
                "text": r["text"],
                "labels": [int(i) for i in r["labels"]],
                "mask": full_mask,
                "source": "goemotions",
            }
            for r in raw
            if (r["text"] or "").strip()
        ]
        bad = [r for r in rows if any(i >= NUM_LABELS for i in r["labels"])]
        if bad:
            raise ValueError(
                f"GoEmotions returned a label index >= {NUM_LABELS}; "
                "emotion_labels.py is stale against the upstream corpus."
            )
        _write_jsonl(out_dir / f"goemotions_{split}.jsonl", rows)


def _rows_single(raw: list[dict], spec) -> list[dict]:
    """One integer label per row (dair-ai/emotion)."""
    out = []
    for r in raw:
        text = (r[spec.text_column] or "").strip()
        idx = spec.target_index(spec.source_label_names[int(r["label"])])
        if text and idx is not None:
            out.append({"text": text, "labels": [idx]})
    return out


def _rows_dialogue(raw: list[dict], spec) -> list[dict]:
    """Nested utterance/emotion lists, exploded to one row per utterance.

    Dropping ~80% of no_emotion utterances is deliberate: DailyDialog is
    overwhelmingly neutral, and keeping them all would flood the mixed corpus
    with `neutral` and bias the model toward predicting nothing at all. Some
    are kept because `neutral` is a real label worth supporting.
    """
    out = []
    neutral_src = spec.source_label_names[0]
    for r in raw:
        utterances = r[spec.text_column] or []
        emotions = r["emotions"] or []
        for i, (utt, emo) in enumerate(zip(utterances, emotions)):
            text = (utt or "").strip()
            if not text:
                continue
            name = spec.source_label_names[int(emo)]
            # Deterministic 1-in-5 sample of the neutral majority — no RNG, so
            # the corpus is byte-identical on every run.
            if name == neutral_src and i % 5 != 0:
                continue
            idx = spec.target_index(name)
            if idx is not None:
                out.append({"text": text, "labels": [idx]})
    return out


def prepare_aux(out_dir: Path, keys: list[str]) -> None:
    """Writes auxiliary corpora into the GoEmotions label space.

    Each row carries `mask`: the label indices this corpus can speak to. Labels
    outside it are unannotated, not negative, and train_roberta drops them from
    the loss.
    """
    from aux_datasets import AUX_DATASETS

    out_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        spec = AUX_DATASETS[key]
        mask = spec.covered_indices

        raw = _read_split(spec.hf_id, spec.hf_config, "train")
        missing = {spec.text_column} - set(raw[0])
        if missing:
            raise ValueError(
                f"{key}: expected column {spec.text_column!r}, got {list(raw[0])}. "
                "The upstream schema changed and this mapping is stale."
            )

        builder = {"single": _rows_single, "dialogue": _rows_dialogue}[spec.layout]
        rows = builder(raw, spec)
        if spec.max_rows:
            rows = rows[: spec.max_rows]

        for row in rows:
            row["mask"] = mask
            row["source"] = key
        _write_jsonl(out_dir / f"{key}_train.jsonl", rows)


def prepare_amazon_sample(out_dir: Path, n: int, seed: int = 42) -> None:
    """Pulls a sample of Amazon reviews for the demo/dashboard only.

    Never used for training. Columns match what backend/app/services/reviews.py
    expects on CSV upload, so the demo exercises the same code path as a real
    customer upload.
    """
    import csv
    import random

    raw = _read_split("fancyzhx/amazon_polarity", "amazon_polarity", "test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(raw)), min(n, len(raw)))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "amazon_reviews_sample.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["review_id", "product_id", "review_text", "star_rating", "review_date"]
        )
        for i, idx in enumerate(indices):
            row = raw[idx]
            text = f"{row['title']}. {row['content']}".strip()
            # amazon_polarity carries only binary polarity; a plausible star
            # rating is synthesized purely for dashboard filter UX.
            star = 5 if row["label"] == 1 else 2
            writer.writerow(
                [f"demo-{i:05d}", f"prod-{idx % 200:04d}", text, star, "2026-01-01"]
            )
    print(f"wrote {path} ({len(indices)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goemotions", action="store_true")
    parser.add_argument("--amazon-sample", action="store_true")
    parser.add_argument(
        "--aux",
        nargs="*",
        metavar="KEY",
        help="auxiliary corpora to prepare; no value means all of them",
    )
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    if args.goemotions:
        prepare_goemotions(args.out)
    if args.amazon_sample:
        prepare_amazon_sample(args.out, args.n)
    if args.aux is not None:
        from aux_datasets import AUX_DATASETS

        keys = args.aux or list(AUX_DATASETS)
        unknown = [k for k in keys if k not in AUX_DATASETS]
        if unknown:
            parser.error(f"unknown --aux {unknown}; choose from {list(AUX_DATASETS)}")
        prepare_aux(args.out, keys)
    if not (args.goemotions or args.amazon_sample or args.aux is not None):
        parser.error("pass --goemotions, --amazon-sample and/or --aux")
