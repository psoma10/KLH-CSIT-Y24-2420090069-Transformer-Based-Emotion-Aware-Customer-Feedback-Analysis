"""Scores every saved model with one metric suite and writes the comparison.

    python compare.py [--bootstrap 300]

Reads   results/predictions/<model>__seed<k>.npz
Writes  results/comparison/
          metrics_per_run.csv       every metric, every (model, seed)
          metrics_summary.csv       mean and std over seeds, every metric
          per_label_f1.csv          per-label F1 (mean over seeds)
          cost.csv                  parameters / train time / inference speed
          significance.csv          paired bootstrap: proposed model vs each other
          confidence_intervals.csv  bootstrap 95% CIs for headline metrics
          fig_*.png                 headline bars, per-label deltas, PR curves
          RESULTS.md                the tables, generated (do not hand-edit)

Protocol, identical for every model: thresholds are tuned per label on the
validation split and applied unchanged to test. Multi-seed models (the
transformers) are reported as mean ± std over seeds; significance and CIs use
the first seed of each model so every model is resampled on the same test rows.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from common import COMPARE_DIR, PRED_DIR, GOEMOTIONS_LABELS, load_predictions, load_split, tune_thresholds
from metrics import HEADLINE, LOWER_IS_BETTER, _confusion, _kappa, _mcc, _safe_div, apply_thresholds, evaluate_all, per_label_table

PROPOSED = "roberta_bilstm_attn+ml"

# (display name, family) in the order they appear in every table.
MODELS = {
    "tfidf_lr": ("TF-IDF + Logistic Regression", "Classical ML"),
    "tfidf_svm": ("TF-IDF + Linear SVM", "Classical ML"),
    "tfidf_nb": ("TF-IDF + Naive Bayes", "Classical ML"),
    "tfidf_rf": ("TF-IDF + Random Forest", "Classical ML"),
    "ml_ensemble": ("TF-IDF ML Ensemble (LR+SVM+RF)", "Hybrid ML"),
    "textcnn": ("TextCNN", "Deep learning"),
    "bilstm": ("BiLSTM", "Deep learning"),
    "cnn_bilstm": ("CNN-BiLSTM-Attention", "Hybrid DL"),
    "bert_plain": ("BERT-base (fine-tuned)", "Transformer"),
    "roberta_plain": ("RoBERTa-base (fine-tuned)", "Transformer"),
    "roberta_plain+ml": ("RoBERTa + TF-IDF ML fusion", "Transformer + ML"),
    "roberta_bilstm_attn": ("RoBERTa-BiLSTM-Attention", "Hybrid DL (Transformer + BiLSTM)"),
    "roberta_bilstm_attn+ml": ("PROPOSED: RoBERTa-BiLSTM-Attn + TF-IDF ML", "Proposed hybrid"),
}
BOOT_METRICS = ["f1_macro", "f1_micro", "mcc_macro", "auc_roc_macro", "pr_auc_macro"]


def load_runs() -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for p in sorted(PRED_DIR.glob("*__seed*.npz")):
        name, seed = re.match(r"(.+)__seed(\d+)\.npz", p.name).groups()
        runs[name][int(seed)] = load_predictions(p)
    return runs


# ------------------------------------------------------------------ bootstrap
def _boot_one(w, Y, probs, preds, pos_cols):
    """All bootstrap metrics for one resample, expressed as row weights `w`."""
    y = Y > 0.5
    p = preds > 0
    W = w[:, None]
    tp, fp = (W * (y & p)).sum(0), (W * (~y & p)).sum(0)
    fn, tn = (W * (y & ~p)).sum(0), (W * (~y & ~p)).sum(0)
    TP, FP, FN, TN = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    out = {
        "f1_macro": float(_safe_div(2 * tp, 2 * tp + fp + fn).mean()),
        "f1_micro": float(2 * TP / max(2 * TP + FP + FN, 1)),
        "mcc_macro": float(_mcc(tp, fp, fn, tn).mean()),
    }
    out["auc_roc_macro"] = float(np.mean([roc_auc_score(Y[:, j], probs[:, j], sample_weight=w) for j in pos_cols]))
    out["pr_auc_macro"] = float(np.mean([average_precision_score(Y[:, j], probs[:, j], sample_weight=w) for j in pos_cols]))
    return out


def bootstrap(Y, probs, thresholds, weights, n_jobs=-1) -> pd.DataFrame:
    preds = apply_thresholds(probs, thresholds)
    pos_cols = [j for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    rows = Parallel(n_jobs=n_jobs)(delayed(_boot_one)(w, Y, probs, preds, pos_cols) for w in weights)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- report
def fmt(mean, std=None, digits=4):
    return f"{mean:.{digits}f}" if std is None or np.isnan(std) or std == 0 else f"{mean:.{digits}f} ± {std:.{digits}f}"


def md_table(df: pd.DataFrame, index=False) -> str:
    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + ["—" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_marks(summary: pd.DataFrame, metric: str) -> str | None:
    col = summary[(metric, "mean")]
    return col.idxmin() if metric in LOWER_IS_BETTER else col.idxmax()


def main(n_boot: int) -> None:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    _, Yva = load_split("validation")
    _, Yte = load_split("test")
    runs = load_runs()
    order = [m for m in MODELS if m in runs]
    missing = [m for m in MODELS if m not in runs]
    if missing:
        print(f"[warn] no predictions for: {', '.join(missing)}")

    # ---- per-run metrics ------------------------------------------------
    records, per_label, thresholds = [], defaultdict(list), {}
    for name in order:
        for seed, r in sorted(runs[name].items()):
            th = tune_thresholds(r["val"], Yva)
            thresholds[(name, seed)] = th
            m = evaluate_all(Yte, r["test"], th)
            records.append({"model": name, "seed": seed, **m})
            per_label[name].append(pd.DataFrame(per_label_table(Yte, r["test"], th)).set_index("label"))
    per_run = pd.DataFrame(records)
    per_run.to_csv(COMPARE_DIR / "metrics_per_run.csv", index=False)

    metric_cols = [c for c in per_run.columns if c not in ("model", "seed")]
    grp = per_run.groupby("model")[metric_cols].agg(["mean", "std"]).reindex(order)
    grp.to_csv(COMPARE_DIR / "metrics_summary.csv")
    n_seeds = per_run.groupby("model")["seed"].nunique().reindex(order)

    pl = pd.concat({n: pd.concat(v).groupby(level=0)["f1"].mean() for n, v in per_label.items()}, axis=1)
    pl = pl.reindex(GOEMOTIONS_LABELS)[order]
    support = per_label[order[0]][0]["support"]
    pl.insert(0, "support", support)
    pl.to_csv(COMPARE_DIR / "per_label_f1.csv")

    # ---- cost -------------------------------------------------------------
    cost = []
    for name in order:
        metas = [r["meta"] for r in runs[name].values()]
        avg = lambda k: np.nanmean([m.get(k) if m.get(k) is not None else np.nan for m in metas])
        cost.append({"model": name, "params_M": avg("n_params") / 1e6, "train_min": avg("train_seconds") / 60,
                     "infer_ms_per_1k_texts": avg("infer_ms_per_1k") / 1e3 if not np.isnan(avg("infer_ms_per_1k")) else np.nan})
    cost = pd.DataFrame(cost).set_index("model")
    cost.to_csv(COMPARE_DIR / "cost.csv")

    # ---- bootstrap CIs + paired significance --------------------------------
    rng = np.random.default_rng(0)
    weights = [np.bincount(rng.integers(0, len(Yte), len(Yte)), minlength=len(Yte)).astype(float) for _ in range(n_boot)]
    boots = {}
    for name in order:
        seed = min(runs[name])
        print(f"bootstrapping {name} (seed {seed}) ...", flush=True)
        boots[name] = bootstrap(Yte, runs[name][seed]["test"], thresholds[(name, seed)], weights)
    ci_rows = []
    for name in order:
        row = {"model": name}
        for k in BOOT_METRICS:
            lo, hi = np.percentile(boots[name][k], [2.5, 97.5])
            row[k] = f"{boots[name][k].mean():.4f} [{lo:.4f}, {hi:.4f}]"
        ci_rows.append(row)
    ci = pd.DataFrame(ci_rows).set_index("model")
    ci.to_csv(COMPARE_DIR / "confidence_intervals.csv")

    sig_rows = []
    if PROPOSED in boots:
        for name in order:
            if name == PROPOSED:
                continue
            for k in BOOT_METRICS:
                d = boots[PROPOSED][k] - boots[name][k]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                sig_rows.append({"vs": name, "metric": k, "delta_mean": d.mean(), "ci_lo": lo, "ci_hi": hi,
                                 "p_value": max(p, 1 / n_boot), "significant_95": bool(lo > 0 or hi < 0)})
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(COMPARE_DIR / "significance.csv", index=False)

    make_figures(order, grp, runs, Yte, thresholds, pl)
    write_report(order, grp, n_seeds, cost, ci, sig, pl, n_boot)
    print(f"\nWrote results to {COMPARE_DIR}")


# -------------------------------------------------------------------- figures
COLORS = {"Classical ML": "#8c8c8c", "Hybrid ML": "#5b7fa6", "Deep learning": "#c7a04a", "Hybrid DL": "#c7842f",
          "Transformer": "#4c956c", "Transformer + ML": "#2f7d63", "Hybrid DL (Transformer + BiLSTM)": "#1f6f8b",
          "Proposed hybrid": "#c0392b"}


def make_figures(order, grp, runs, Yte, thresholds, pl):
    metrics = ["f1_macro", "f1_micro", "auc_roc_macro", "pr_auc_macro", "mcc_macro"]
    labels = ["Macro F1", "Micro F1", "Macro AUC-ROC", "Macro PR-AUC", "Macro MCC"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(19, 5.6), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, mtr, lab in zip(axes, metrics, labels):
        vals = [grp.loc[m, (mtr, "mean")] for m in order]
        errs = [np.nan_to_num(grp.loc[m, (mtr, "std")]) for m in order]
        ax.barh(y, vals, xerr=errs, color=[COLORS[MODELS[m][1]] for m in order], capsize=2, height=0.72)
        for yy, v in zip(y, vals):
            ax.text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=8)
        ax.set_title(lab, fontsize=11)
        ax.set_xlim(min(vals) * 0.85, max(vals) * 1.09)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODELS[m][0].replace("PROPOSED: ", "★ ") for m in order], fontsize=9)
    fig.suptitle("GoEmotions test set — all models, identical protocol (thresholds tuned on validation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(COMPARE_DIR / "fig_headline_metrics.png", dpi=160)
    plt.close(fig)

    if PROPOSED in pl and "roberta_plain" in pl:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=False)
        for ax, ref in zip(axes, ["roberta_plain", "ml_ensemble"]):
            if ref not in pl:
                continue
            d = (pl[PROPOSED] - pl[ref]).sort_values()
            ax.barh(d.index, d.values, color=["#c0392b" if v < 0 else "#2f7d63" for v in d.values])
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"Per-label F1: proposed − {MODELS[ref][0]}", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(COMPARE_DIR / "fig_per_label_delta.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in ["tfidf_lr", "ml_ensemble", "bilstm", "roberta_plain", PROPOSED]:
        if name not in runs:
            continue
        seed = min(runs[name])
        pr, rc, _ = precision_recall_curve((Yte > 0.5).ravel(), runs[name][seed]["test"].ravel())
        ap = average_precision_score((Yte > 0.5).ravel(), runs[name][seed]["test"].ravel())
        ax.plot(rc, pr, lw=2.4 if name == PROPOSED else 1.4, color=COLORS[MODELS[name][1]],
                label=f"{MODELS[name][0].replace('PROPOSED: ', '★ ')} (AP {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Micro-averaged precision–recall, test set")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(COMPARE_DIR / "fig_pr_curves.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------- report
GROUPS = {
    "Threshold-based (thresholds tuned on validation)": [
        ("f1_macro", "Macro F1"), ("f1_micro", "Micro F1"), ("f1_weighted", "Weighted F1"), ("f1_samples", "Samples F1"),
        ("precision_macro", "Macro Precision"), ("recall_macro", "Macro Recall / Sensitivity"),
        ("specificity_macro", "Macro Specificity"), ("balanced_accuracy_macro", "Balanced Acc."),
        ("mcc_macro", "Macro MCC"), ("mcc_micro", "Micro MCC"), ("kappa_macro", "Macro Cohen's κ"),
        ("subset_accuracy", "Exact-match Acc."), ("top1_accuracy", "Top-1 Acc."),
        ("jaccard_samples", "Jaccard (samples)"), ("hamming_loss", "Hamming loss ↓"),
    ],
    "Threshold-free ranking & probability quality": [
        ("auc_roc_macro", "Macro AUC-ROC"), ("auc_roc_micro", "Micro AUC-ROC"),
        ("pr_auc_macro", "Macro PR-AUC"), ("pr_auc_micro", "Micro PR-AUC"),
        ("label_ranking_ap", "Label-ranking AP"), ("coverage_error", "Coverage error ↓"),
        ("ranking_loss", "Ranking loss ↓"), ("log_loss", "Log loss ↓"),
        ("brier_score", "Brier ↓"), ("ece", "ECE ↓"),
    ],
    "Business-bucket level (4 buckets the dashboard reports)": [
        ("bucket_macro_f1", "Bucket macro F1"), ("bucket_macro_auc_roc", "Bucket macro AUC-ROC"),
    ],
}


def metric_table(order, grp, n_seeds, keys) -> str:
    rows = {}
    for m in order:
        rows[MODELS[m][0]] = {}
    for key, lab in keys:
        best = best_marks(grp, key)
        for m in order:
            mean, std = grp.loc[m, (key, "mean")], grp.loc[m, (key, "std")]
            digits = 3 if key == "coverage_error" else 4
            cell = fmt(mean, std if n_seeds[m] > 1 else None, digits)
            rows[MODELS[m][0]][lab] = f"**{cell}**" if m == best else cell
    return md_table(pd.DataFrame(rows).T.rename_axis("Model"), index=True)


def write_report(order, grp, n_seeds, cost, ci, sig, pl, n_boot):
    multi = [f"{MODELS[m][0].split(' (')[0]}={n_seeds[m]}" for m in order if n_seeds[m] > 1]
    seeds_note = ", ".join(multi) if multi else "none"
    L = ["# Comparative results (auto-generated by `src/ml/compare.py`)", "",
         "GoEmotions test split, 5,427 comments, 28 labels. Every model is scored by the same code with per-label "
         "thresholds tuned on the validation split. **Bold** = best in column. ↓ = lower is better. "
         "`±` = std over seeds, shown only for models trained with more than one seed "
         f"(seeds per model: {seeds_note}).", ""]
    for title, keys in GROUPS.items():
        L += [f"## {title}", "", metric_table(order, grp, n_seeds, keys), ""]

    L += ["## Cost", "", md_table(cost.rename(index=lambda m: MODELS[m][0]).round(2).rename_axis("Model"), index=True), ""]
    L += [f"## Bootstrap 95% confidence intervals ({n_boot} resamples of the test set)", "",
          md_table(ci.rename(index=lambda m: MODELS[m][0]), index=True), ""]
    if len(sig):
        L += [f"## Is the proposed model significantly better? (paired bootstrap, {n_boot} resamples)", "",
              "Δ = proposed − other. Significant = 95% CI of Δ excludes 0.", ""]
        t = sig.copy()
        t["vs"] = t["vs"].map(lambda m: MODELS[m][0])
        t["Δ [95% CI]"] = t.apply(lambda r: f"{r.delta_mean:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", axis=1)
        t["sig."] = t["significant_95"].map({True: "yes", False: "no"})
        pt = t.pivot(index="vs", columns="metric", values="Δ [95% CI]").reindex(
            [MODELS[m][0] for m in order if m != PROPOSED])[BOOT_METRICS]
        L += [md_table(pt.rename_axis("Proposed vs."), index=True), ""]
    L += ["## Per-label F1 (mean over seeds)", "", md_table(pl.round(3).rename(columns=lambda c: MODELS.get(c, (c,))[0]).rename_axis("Label"), index=True), ""]
    (COMPARE_DIR / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=300)
    main(ap.parse_args().bootstrap)
