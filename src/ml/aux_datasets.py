"""Auxiliary emotion corpora mapped into the GoEmotions 28-label space.

GoEmotions is the backbone: it is the only source annotating all 28 labels, so
it alone defines the label space. Every corpus here annotates a *subset*, which
is the whole difficulty. If a dair-ai row is tagged `sadness`, that says nothing
about whether `gratitude` applies — annotators were never asked. Training on
that zero as a negative would teach the model those labels are absent whenever
this corpus appears, damaging exactly the labels the extra data was meant to
leave alone.

So each entry declares which GoEmotions labels its scheme can speak to.
data_prep writes that per row as a mask, and train_roberta computes BCE only
over masked positions, so unmasked labels contribute no gradient.

Everything is fetched as Parquet over plain HTTP rather than through the
`datasets` library. Two reasons: it removes a heavy dependency from data prep,
and `datasets` cannot be imported on a Python built without `_lzma`, which is
how this project's interpreter happens to be built.

Datasets are used exactly as published — no hand-authored or locally-edited
data anywhere in the training path.

Sources
  GoEmotions       https://huggingface.co/datasets/google-research-datasets/go_emotions
                   Demszky et al. 2020, https://arxiv.org/abs/2005.00547
  dair-ai/emotion  https://huggingface.co/datasets/dair-ai/emotion
                   Saravia et al. 2018 (CARER), https://aclanthology.org/D18-1404/
  DailyDialog      https://huggingface.co/datasets/roskoN/dailydialog
                   Li et al. 2017, https://arxiv.org/abs/1710.03957

Why not SemEval-2018 Task 1 (E-c), the obvious fourth source: the original
`SemEvalWorkshop/sem_eval_2018_task_1` is a script-based dataset with no
Parquet export, and the one mirror that does export Parquet
(`vibhorag101/..._cleaned_labels`) ships text with stopwords removed and casing
folded — "whatever decide make sure make happy". RoBERTa is pretrained on
natural text and relies on function words and casing, so mixing that register
with raw GoEmotions would add distribution noise rather than signal. It is
excluded deliberately, not overlooked.

Why not DailyDialog via `li2017dailydialog/daily_dialog`: same script-dataset
problem. `roskoN/dailydialog` is the same corpus published as data-only, and is
cited to the original paper above.
"""
from __future__ import annotations

from dataclasses import dataclass

from emotion_labels import GOEMOTIONS_LABELS


@dataclass(frozen=True)
class AuxDataset:
    """One auxiliary corpus and how its labels land in the 28-label space."""

    key: str
    hf_id: str
    hf_config: str
    text_column: str
    # Source label name -> GoEmotions label name. Source labels missing from
    # this mapping have no GoEmotions equivalent and are dropped rather than
    # forced onto a near-miss, which would inject the noise the extra data is
    # meant to reduce.
    label_map: dict[str, str]
    # Ordered source label names, for corpora whose label column is an integer
    # index with no names attached (DailyDialog's `emotions`).
    source_label_names: tuple[str, ...]
    # "single" – one label per row, from an int index.
    # "boolean_columns" – one boolean column per label.
    # "dialogue" – nested utterance/emotion lists, exploded to one row each.
    layout: str
    # Cap on rows kept per split. DailyDialog is ~100k utterances and about
    # four-fifths carry no emotion; uncapped it would swamp GoEmotions and turn
    # the run into a `neutral` classifier.
    max_rows: int | None = None

    @property
    def covered_indices(self) -> list[int]:
        """Indices this corpus can speak to — the per-row mask.

        Only mapped targets. A corpus that never annotates `pride` must not
        push `pride` toward zero.
        """
        return sorted({GOEMOTIONS_LABELS.index(v) for v in self.label_map.values()})

    def target_index(self, source_label: str) -> int | None:
        mapped = self.label_map.get(source_label)
        return None if mapped is None else GOEMOTIONS_LABELS.index(mapped)


# dair-ai/emotion (CARER) — 16k English tweets, six single-label classes.
# Column names verified against the published Parquet: ["text", "label"]. All
# six source labels exist verbatim in GoEmotions, so nothing here is a
# judgement call.
DAIR_EMOTION = AuxDataset(
    key="dair_emotion",
    hf_id="dair-ai/emotion",
    hf_config="split",
    text_column="text",
    label_map={
        "sadness": "sadness",
        "joy": "joy",
        "love": "love",
        "anger": "anger",
        "fear": "fear",
        "surprise": "surprise",
    },
    source_label_names=("sadness", "joy", "love", "anger", "fear", "surprise"),
    layout="single",
)

# DailyDialog — 11k dialogues, exploded to one row per utterance. Spoken
# register, closer to customer-support text than Reddit is, and the only
# auxiliary source carrying an explicit no-emotion class that maps to
# `neutral`. Emotion ints follow the corpus's published order, starting at
# no_emotion=0.
DAILY_DIALOG = AuxDataset(
    key="daily_dialog",
    hf_id="roskoN/dailydialog",
    hf_config="full",
    text_column="utterances",
    label_map={
        "no_emotion": "neutral",
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "happiness": "joy",
        "sadness": "sadness",
        "surprise": "surprise",
    },
    source_label_names=(
        "no_emotion",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ),
    layout="dialogue",
    max_rows=20_000,
)

AUX_DATASETS: dict[str, AuxDataset] = {d.key: d for d in (DAIR_EMOTION, DAILY_DIALOG)}


def validate_mappings() -> None:
    """Fails loudly on a mapping target that is not a real GoEmotions label.

    Cheap, and the alternative is an IndexError deep inside training or — far
    worse — a silently wrong index that quietly trains the wrong label.
    """
    for name, ds in AUX_DATASETS.items():
        for src, tgt in ds.label_map.items():
            if tgt not in GOEMOTIONS_LABELS:
                raise ValueError(
                    f"{name}: maps {src!r} to {tgt!r}, not a GoEmotions label"
                )
            if src not in ds.source_label_names:
                raise ValueError(
                    f"{name}: maps {src!r}, absent from source_label_names"
                )
        if not ds.covered_indices:
            raise ValueError(f"{name}: maps to no GoEmotions labels")


validate_mappings()
