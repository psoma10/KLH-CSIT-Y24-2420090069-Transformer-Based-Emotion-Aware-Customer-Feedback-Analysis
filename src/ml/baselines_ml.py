"""Classical ML baselines and the hybrid ML ensemble, all on TF-IDF features.

    python baselines_ml.py            # all models
    python baselines_ml.py --only tfidf_lr tfidf_svm

Models (each one-vs-rest over the 28 GoEmotions labels):
  tfidf_lr        Logistic Regression
  tfidf_svm       Linear SVM, Platt-calibrated so it emits probabilities
  tfidf_nb        Multinomial Naive Bayes
  tfidf_rf        Random Forest (native multi-label)
  ml_ensemble     HYBRID ML: soft-voting over a validation-selected subset of the four

Hyper-parameters are picked on the validation split (macro-F1 at tuned
thresholds), never on test. Test probabilities are written once, at the end.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from common import PRED_DIR, Timer, load_predictions, load_split, save_predictions, set_seed, tune_thresholds
from metrics import apply_thresholds, _confusion, _safe_div

SEED = 42


def build_features(train, val, test):
    """Word 1-2grams + character 2-5grams. Character n-grams matter here: the
    text is Reddit, full of misspellings, elongations ("sooooo") and emoticons
    that a word vocabulary throws away."""
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150_000, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=250_000)
    Xtr = hstack([word.fit_transform(train), char.fit_transform(train)]).tocsr()
    Xva = hstack([word.transform(val), char.transform(val)]).tocsr()
    Xte = hstack([word.transform(test), char.transform(test)]).tocsr()
    # NB needs non-negative counts-like features; word-only block is enough for it
    nw = word.transform(train).shape[1]
    return Xtr, Xva, Xte, nw


def macro_f1_at_tuned(val_probs, Yva) -> float:
    th = tune_thresholds(val_probs, Yva)
    tp, fp, fn, _ = _confusion(Yva, apply_thresholds(val_probs, th))
    return float(_safe_div(2 * tp, 2 * tp + fp + fn).mean())


def _native_proba(clf, X):
    """RandomForest fits all labels in one forest; predict_proba gives one array per label."""
    return np.stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(X)) for p in clf.predict_proba(X)], axis=1)


def fit_ovr(make_estimator, Xtr, Ytr, Xva, Xte, native=False):
    if native:
        clf = make_estimator().fit(Xtr, Ytr.astype(int))
        return clf, _native_proba(clf, Xva), _native_proba(clf, Xte)
    clf = OneVsRestClassifier(make_estimator(), n_jobs=-1).fit(Xtr, Ytr.astype(int))
    return clf, clf.predict_proba(Xva), clf.predict_proba(Xte)


def select_and_fit(name, grid, make, Xtr, Ytr, Xva, Yva, Xte, native=False):
    """Small validation-set grid search; returns the winning model's probabilities."""
    best = None
    for params in grid:
        clf, pv, pt = fit_ovr(lambda p=params: make(**p), Xtr, Ytr, Xva, Xte, native)
        score = macro_f1_at_tuned(pv, Yva)
        print(f"  {name} {params} -> val macro-F1 {score:.4f}")
        if best is None or score > best[0]:
            best = (score, params, pv, pt)
    return best


def main(only: set[str] | None) -> None:
    set_seed(SEED)
    tr_x, Ytr = load_split("train")
    va_x, Yva = load_split("validation")
    te_x, _ = load_split("test")

    with Timer() as t_feat:
        Xtr, Xva, Xte, nw = build_features(tr_x, va_x, te_x)
    print(f"features: {Xtr.shape[1]:,} dims (word block {nw:,}) in {t_feat.seconds:.1f}s")
    Xtr_w, Xva_w, Xte_w = Xtr[:, :nw], Xva[:, :nw], Xte[:, :nw]

    specs = {
        "tfidf_lr": dict(
            grid=[{"C": c} for c in (2.0, 8.0)],
            make=lambda C: LogisticRegression(C=C, solver="liblinear", max_iter=200),
            X=(Xtr, Xva, Xte),
        ),
        "tfidf_svm": dict(
            grid=[{"C": c} for c in (0.1, 0.4)],
            make=lambda C: CalibratedClassifierCV(LinearSVC(C=C), method="sigmoid", cv=3),
            X=(Xtr, Xva, Xte),
        ),
        "tfidf_nb": dict(
            grid=[{"alpha": a} for a in (0.05, 0.3)],
            make=lambda alpha: MultinomialNB(alpha=alpha),
            X=(Xtr_w, Xva_w, Xte_w),
        ),
        "tfidf_rf": dict(
            grid=[{"n": 200}],
            make=lambda n: RandomForestClassifier(
                n_estimators=n, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=SEED
            ),
            X=(Xtr_w, Xva_w, Xte_w),
            native=True,
        ),
    }

    outputs: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    for name, s in specs.items():
        if only and name not in only:
            continue
        print(f"== {name}")
        with Timer() as t:
            score, params, pv, pt = select_and_fit(name, s["grid"], s["make"], s["X"][0], Ytr, s["X"][1], Yva, s["X"][2], s.get("native", False))
        n_params = int(Xtr.shape[1] * 28) if name in ("tfidf_lr", "tfidf_svm") else None
        meta = {"family": "Classical ML", "hyperparams": params, "val_macro_f1": score,
                "train_seconds": t.seconds, "n_params": n_params}
        outputs[name] = (pv, pt, meta)
        if not only or name in only:
            save_predictions(name, SEED, pv, pt, meta)

    if not only or "ml_ensemble" in only:
        build_ensemble(Yva)


ENSEMBLE_CANDIDATES = ["tfidf_lr", "tfidf_svm", "tfidf_nb", "tfidf_rf"]


def build_ensemble(Yva) -> None:
    """HYBRID ML: soft-voting (mean probability) over a quality-filtered set of TF-IDF models.

    Quality floor: a member is dropped when its validation macro-PR-AUC is below
    90% of the best member's. That removes Naive Bayes, whose probabilities are
    badly calibrated (val macro-F1 0.30) and which measurably lowers every
    ensemble it joins. Threshold-free PR-AUC is used for the floor rather than
    tuned-threshold F1 because it is far less noisy on the rare labels.
    (Tried instead: greedy forward selection on val F1 — it collapses to the SVM
    alone, i.e. not an ensemble at all, since LR/SVM share a feature view.)
    Built from saved member predictions, so `--only ml_ensemble` re-runs it alone.
    """
    members = {n: load_predictions(PRED_DIR / f"{n}__seed{SEED}.npz") for n in ENSEMBLE_CANDIDATES}
    solo_ap = {n: macro_ap(Yva, m["val"]) for n, m in members.items()}
    floor = 0.9 * max(solo_ap.values())
    chosen = [n for n in members if solo_ap[n] >= floor]
    pv = np.mean([members[k]["val"] for k in chosen], axis=0)
    pt = np.mean([members[k]["test"] for k in chosen], axis=0)
    val_f1 = macro_f1_at_tuned(pv, Yva)
    meta = {"family": "Hybrid ML", "members": chosen, "dropped": [n for n in members if n not in chosen],
            "member_val_pr_auc": solo_ap, "combiner": "soft voting (mean probability), 0.9x best-member PR-AUC floor",
            "val_macro_f1": val_f1, "n_params": None,
            "train_seconds": sum(members[k]["meta"].get("train_seconds", 0) for k in chosen)}
    print(f"== ml_ensemble members {chosen} (dropped {meta['dropped']}) -> val macro-F1 {val_f1:.4f}")
    save_predictions("ml_ensemble", SEED, pv, pt, meta)


def macro_ap(Y, P) -> float:
    return float(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    main(set(a.only) if a.only else None)
