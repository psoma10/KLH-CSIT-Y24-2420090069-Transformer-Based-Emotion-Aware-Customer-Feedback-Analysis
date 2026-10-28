"""Convert results/comparison/{metrics_summary,per_label_f1}.csv to a static JSON
consumed by the Model Comparison page. Numbers are copied verbatim (no rounding).

Run from anywhere:  python src/frontend/scripts/build_comparison_data.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results" / "comparison"
OUT = Path(__file__).resolve().parents[1] / "src" / "data" / "model_comparison.json"

# id -> (display name, group)
MODELS = {
    "tfidf_lr": ("TF-IDF + Logistic Regression", "baseline"),
    "tfidf_svm": ("TF-IDF + Linear SVM", "baseline"),
    "tfidf_nb": ("TF-IDF + Naive Bayes", "baseline"),
    "tfidf_rf": ("TF-IDF + Random Forest", "baseline"),
    "ml_ensemble": ("TF-IDF ML Ensemble (LR+SVM+RF)", "baseline"),
    "textcnn": ("TextCNN", "baseline"),
    "bilstm": ("BiLSTM", "baseline"),
    "cnn_bilstm": ("CNN-BiLSTM-Attention", "baseline"),
    "bert_plain": ("BERT-base (fine-tuned)", "baseline"),
    "roberta_plain": ("RoBERTa-base (fine-tuned)", "baseline"),
    "roberta_plain+ml": ("RoBERTa + TF-IDF fusion", "hybrid"),
    "roberta_bilstm_attn": ("RoBERTa-BiLSTM-Attention", "hybrid"),
    "roberta_bilstm_attn+ml": ("PROPOSED: RoBERTa-BiLSTM-Attn + TF-IDF", "proposed"),
}


def num(s):
    return float(s) if s not in ("", None) else None


def main():
    with open(SRC / "metrics_summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    names, stats = rows[0][1:], rows[1][1:]
    models = []
    for r in rows[3:]:
        if not r or r[0] not in MODELS:
            continue
        metrics = {}
        for name, stat, val in zip(names, stats, r[1:]):
            metrics.setdefault(name, {})[stat] = num(val)
        for m in metrics.values():
            m.setdefault("std", None)
        name, group = MODELS[r[0]]
        models.append({"id": r[0], "name": name, "group": group, "metrics": metrics})

    with open(SRC / "per_label_f1.csv", newline="", encoding="utf-8") as f:
        pl = list(csv.DictReader(f))
    per_label = [
        {"label": d["label"], "support": int(float(d["support"])),
         "f1": {m: num(d[m]) for m in MODELS if m in d}}
        for d in pl
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"dataset": "GoEmotions test split (5,427 comments, 28 labels)",
                    "models": models, "perLabel": per_label}, indent=1),
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(models)} models, {len(per_label)} labels)")


if __name__ == "__main__":
    main()
