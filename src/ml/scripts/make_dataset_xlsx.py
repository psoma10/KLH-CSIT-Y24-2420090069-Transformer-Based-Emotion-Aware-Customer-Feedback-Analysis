"""One-off: convert datasets/raw/* into human-viewable .xlsx files.

Decodes integer label IDs to emotion names so a reviewer can open the file
in Excel and read it, rather than staring at label=3. Run from ml/:

    python scripts/make_dataset_xlsx.py
"""

from pathlib import Path

import pandas as pd

from emotion_labels import GOEMOTIONS_LABELS
from aux_datasets import DAIR_EMOTION, DAILY_DIALOG

RAW = Path("datasets/raw")
OUT = Path("datasets/xlsx")
OUT.mkdir(parents=True, exist_ok=True)

EXCEL_ROW_LIMIT = 1_048_576 - 1  # header row


def decode_goemotions_labels(label_ids) -> str:
    return ", ".join(GOEMOTIONS_LABELS[i] for i in label_ids)


def write_go_emotions() -> None:
    with pd.ExcelWriter(OUT / "go_emotions.xlsx", engine="openpyxl") as xl:
        for split in ("train", "validation", "test"):
            df = pd.read_parquet(RAW / "go_emotions" / f"{split}-00000-of-00001.parquet")
            df = df.copy()
            df["emotions"] = df["labels"].apply(decode_goemotions_labels)
            df = df[["id", "text", "emotions", "labels"]].rename(
                columns={"labels": "label_ids"}
            )
            df.to_excel(xl, sheet_name=split, index=False)

    # raw/ config: 211k rows, per-annotator, no aggregated label ids to decode
    raw_path = RAW / "go_emotions" / "raw-train-00000-of-00001.parquet"
    df = pd.read_parquet(raw_path)
    if len(df) > EXCEL_ROW_LIMIT:
        df = df.iloc[:EXCEL_ROW_LIMIT]
    df.to_excel(OUT / "go_emotions_raw_annotator_level.xlsx", index=False, engine="openpyxl")
    print("wrote go_emotions.xlsx, go_emotions_raw_annotator_level.xlsx")


def write_dair_emotion() -> None:
    names = DAIR_EMOTION.source_label_names  # index position == label int
    with pd.ExcelWriter(OUT / "dair_ai_emotion.xlsx", engine="openpyxl") as xl:
        for split in ("train", "validation", "test"):
            df = pd.read_parquet(RAW / "dair_ai_emotion" / f"{split}-00000-of-00001.parquet")
            df = df.copy()
            df["emotion"] = df["label"].apply(lambda i: names[i])
            df = df[["text", "emotion", "label"]].rename(columns={"label": "label_id"})
            df.to_excel(xl, sheet_name=split, index=False)
    print("wrote dair_ai_emotion.xlsx")


def _parse_dialogues_file(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    dialogues = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        utterances = [u.strip() for u in line.split("__eou__") if u.strip()]
        dialogues.append(utterances)
    return dialogues


def _parse_int_labels_file(path: Path) -> list[list[int]]:
    text = path.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(x) for x in line.split()])
    return rows


def write_dailydialog() -> None:
    names = DAILY_DIALOG.source_label_names  # index position == emotion int
    with pd.ExcelWriter(OUT / "dailydialog.xlsx", engine="openpyxl") as xl:
        for split in ("train", "validation", "test"):
            split_dir = RAW / "dailydialog" / split
            dialogues = _parse_dialogues_file(split_dir / f"dialogues_{split}.txt")
            emotions = _parse_int_labels_file(split_dir / f"dialogues_emotion_{split}.txt")
            acts = _parse_int_labels_file(split_dir / f"dialogues_act_{split}.txt")

            rows = []
            for d_idx, (utts, emos, act_ids) in enumerate(zip(dialogues, emotions, acts)):
                for u_idx, (utt, emo_id, act_id) in enumerate(zip(utts, emos, act_ids)):
                    rows.append(
                        {
                            "dialogue_id": d_idx,
                            "turn": u_idx,
                            "utterance": utt,
                            "emotion": names[emo_id],
                            "emotion_id": emo_id,
                            "dialogue_act_id": act_id,
                        }
                    )
            df = pd.DataFrame(rows)
            if len(df) > EXCEL_ROW_LIMIT:
                df = df.iloc[:EXCEL_ROW_LIMIT]
            df.to_excel(xl, sheet_name=split, index=False)
    print("wrote dailydialog.xlsx")


if __name__ == "__main__":
    write_go_emotions()
    write_dair_emotion()
    write_dailydialog()
    print("done ->", OUT)
