"""Full multi-label evaluation suite.

`evaluate_all(Y, probs, thresholds)` returns one flat dict of ~35 metrics.
Multi-label conventions, stated once so the table headers are unambiguous:

  * "macro" = unweighted mean over the labels that have both classes present in
    the split (AUC / PR-AUC are undefined otherwise; every label has test
    positives here, so nothing is actually dropped).
  * "micro" = pool every (sample, label) decision into one binary problem.
  * Specificity / MCC / Cohen's kappa are binary-per-label quantities, so the
    macro version is the mean of the 28 per-label values; the micro version is
    computed on the pooled decisions.
  * Sensitivity is recall, reported under both names because reviewers ask for
    both.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    coverage_error,
    jaccard_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    roc_auc_score,
)

from emotion_labels import BUSINESS_BUCKETS, GOEMOTIONS_LABELS

EPS = 1e-7
_LABEL_IDX = {l: i for i, l in enumerate(GOEMOTIONS_LABELS)}


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs >= thresholds[None, :]).astype(np.int8)


def _confusion(Y: np.ndarray, pred: np.ndarray):
    y = Y > 0.5
    p = pred > 0
    tp = (y & p).sum(0).astype(np.float64)
    fp = (~y & p).sum(0).astype(np.float64)
    fn = (y & ~p).sum(0).astype(np.float64)
    tn = (~y & ~p).sum(0).astype(np.float64)
    return tp, fp, fn, tn


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=np.float64)), where=np.asarray(b) != 0)


def _mcc(tp, fp, fn, tn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _safe_div(num, den)


def _kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = _safe_div(tp + tn, n)
    pe = _safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), n * n)
    return _safe_div(po - pe, 1 - pe)


def _valid_cols(Y: np.ndarray) -> np.ndarray:
    pos = (Y > 0.5).sum(0)
    return (pos > 0) & (pos < len(Y))


def _per_label_auc(Y, probs, fn):
    out = np.full(Y.shape[1], np.nan)
    for j in np.flatnonzero(_valid_cols(Y)):
        out[j] = fn(Y[:, j], probs[:, j])
    return out


def expected_calibration_error(Y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    """ECE over all (sample, label) probabilities, equal-width bins."""
    p = probs.ravel()
    y = (Y > 0.5).ravel().astype(np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def per_label_table(Y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    pred = apply_thresholds(probs, thresholds)
    tp, fp, fn, tn = _confusion(Y, pred)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    spec = _safe_div(tn, tn + fp)
    mcc = _mcc(tp, fp, fn, tn)
    kap = _kappa(tp, fp, fn, tn)
    auc = _per_label_auc(Y, probs, roc_auc_score)
    ap = _per_label_auc(Y, probs, average_precision_score)
    return [
        {
            "label": GOEMOTIONS_LABELS[j], "support": int(tp[j] + fn[j]),
            "threshold": float(thresholds[j]), "precision": prec[j], "recall": rec[j],
            "f1": f1[j], "specificity": spec[j], "mcc": mcc[j], "kappa": kap[j],
            "auc_roc": auc[j], "pr_auc": ap[j],
        }
        for j in range(Y.shape[1])
    ]


def bucket_metrics(Y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    """Metrics at the 4 business buckets the dashboard actually reports.

    A bucket is positive when any of its emotions is; its score is the max of
    its members' probabilities, its decision the OR of its members' decisions.
    """
    pred = apply_thresholds(probs, thresholds)
    yb, pb, db = [], [], []
    for members in BUSINESS_BUCKETS.values():
        idx = [_LABEL_IDX[m] for m in members]
        yb.append((Y[:, idx] > 0.5).any(1))
        pb.append(probs[:, idx].max(1))
        db.append(pred[:, idx].any(1))
    yb, pb, db = (np.stack(a, 1) for a in (yb, pb, db))
    tp, fp, fn, _ = _confusion(yb.astype(float), db.astype(int))
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    auc = _per_label_auc(yb.astype(float), pb, roc_auc_score)
    return {"bucket_macro_f1": float(np.nanmean(f1)), "bucket_macro_auc_roc": float(np.nanmean(auc))}


def evaluate_all(Y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    """All headline metrics for one model on one split. Thresholds come from validation."""
    Y = (Y > 0.5).astype(np.float64)
    probs = np.clip(probs.astype(np.float64), EPS, 1 - EPS)
    pred = apply_thresholds(probs, thresholds)
    n, L = Y.shape
    tp, fp, fn, tn = _confusion(Y, pred)
    support = tp + fn

    # ---- threshold-based ----
    prec, rec = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    spec = _safe_div(tn, tn + fp)
    TP, FP, FN, TN = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    micro_p, micro_r = TP / max(TP + FP, 1), TP / max(TP + FN, 1)

    inter = (Y * pred).sum(1)
    union = ((Y + pred) > 0).sum(1)
    samples_f1 = _safe_div(2 * inter, Y.sum(1) + pred.sum(1))
    samples_p = _safe_div(inter, pred.sum(1))
    samples_r = _safe_div(inter, Y.sum(1))
    w = support / support.sum()

    valid = _valid_cols(Y)
    auc = _per_label_auc(Y, probs, roc_auc_score)
    ap = _per_label_auc(Y, probs, average_precision_score)

    m = {
        # accuracy family
        "accuracy_elementwise": float((tp + tn).sum() / (n * L)),
        "subset_accuracy": float((pred == Y).all(1).mean()),
        "top1_accuracy": float(Y[np.arange(n), probs.argmax(1)].mean()),
        "balanced_accuracy_macro": float(np.mean((rec + spec) / 2)),
        "hamming_loss": float((pred != Y).mean()),
        "jaccard_samples": float(np.mean(_safe_div(inter, union))),
        "jaccard_macro": float(jaccard_score(Y, pred, average="macro", zero_division=0)),
        # precision / recall / F1
        "precision_macro": float(prec.mean()), "recall_macro": float(rec.mean()), "f1_macro": float(f1.mean()),
        "precision_micro": float(micro_p), "recall_micro": float(micro_r),
        "f1_micro": float(2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12)),
        "precision_weighted": float((prec * w).sum()), "recall_weighted": float((rec * w).sum()),
        "f1_weighted": float((f1 * w).sum()),
        "precision_samples": float(samples_p.mean()), "recall_samples": float(samples_r.mean()),
        "f1_samples": float(samples_f1.mean()),
        # sensitivity / specificity
        "sensitivity_macro": float(rec.mean()),
        "specificity_macro": float(spec.mean()),
        "specificity_micro": float(TN / max(TN + FP, 1)),
        # chance-corrected agreement
        "mcc_macro": float(_mcc(tp, fp, fn, tn).mean()),
        "mcc_micro": float(_mcc(TP, FP, FN, TN)),
        "kappa_macro": float(_kappa(tp, fp, fn, tn).mean()),
        "kappa_micro": float(_kappa(TP, FP, FN, TN)),
        # threshold-free ranking quality
        "auc_roc_macro": float(np.nanmean(auc)),
        "auc_roc_micro": float(roc_auc_score(Y.ravel(), probs.ravel())),
        "auc_roc_weighted": float(np.nansum(auc[valid] * w[valid]) / w[valid].sum()),
        "pr_auc_macro": float(np.nanmean(ap)),
        "pr_auc_micro": float(average_precision_score(Y.ravel(), probs.ravel())),
        "pr_auc_weighted": float(np.nansum(ap[valid] * w[valid]) / w[valid].sum()),
        "label_ranking_ap": float(label_ranking_average_precision_score(Y, probs)),
        "coverage_error": float(coverage_error(Y, probs)),
        "ranking_loss": float(label_ranking_loss(Y, probs)),
        # probabilistic quality
        "log_loss": float(-np.mean(Y * np.log(probs) + (1 - Y) * np.log(1 - probs))),
        "brier_score": float(np.mean((probs - Y) ** 2)),
        "ece": expected_calibration_error(Y, probs),
    }
    m.update(bucket_metrics(Y, probs, thresholds))
    return m


# Which direction is "better" — drives bolding and the ranking in compare.py.
LOWER_IS_BETTER = {"hamming_loss", "coverage_error", "ranking_loss", "log_loss", "brier_score", "ece"}

# The subset of metrics shown in the headline table and bootstrapped for CIs.
HEADLINE = [
    "f1_macro", "f1_micro", "f1_weighted", "f1_samples",
    "precision_macro", "recall_macro", "specificity_macro",
    "auc_roc_macro", "pr_auc_macro", "mcc_macro", "kappa_macro",
    "subset_accuracy", "hamming_loss", "log_loss",
]
