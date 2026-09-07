"""Repertoire- and sequence-level metrics.

Two things earlier evaluations did not do, and which these functions add:

1. Report **ROC-AUC / PR-AUC**, not only accuracy. With a 44%/56% class split an
   accuracy of 0.61 is barely above the majority baseline and says little about
   ranking quality.
2. Evaluate on **held-out repertoires**. Splitting sequences out of the training
   repertoires leaves the repertoire's own topic proportions fitted on the labels,
   so those numbers are optimistic by construction.
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


__all__ = [
    "classify_repertoires",
    "cross_validated_report",
    "precision_at_k",
    "signal_enrichment",
]


def _make_classifier(kind: str):
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, max_depth=3, class_weight="balanced", random_state=0
        )
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    raise ValueError(f"Unsupported classifier {kind}")


def classify_repertoires(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    classifier: str = "random_forest",
) -> dict[str, float]:
    """Fit on the training repertoires, report on the held-out ones."""
    model = _make_classifier(classifier)
    model.fit(features_train, labels_train)
    scores_test = model.predict_proba(features_test)[:, 1]
    scores_train = model.predict_proba(features_train)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(labels_test, scores_test)),
        "pr_auc": float(average_precision_score(labels_test, scores_test)),
        "accuracy": float(accuracy_score(labels_test, scores_test >= 0.5)),
        "f1": float(f1_score(labels_test, scores_test >= 0.5)),
        "train_roc_auc": float(roc_auc_score(labels_train, scores_train)),
        "majority_baseline_accuracy": float(
            max(labels_test.mean(), 1 - labels_test.mean())
        ),
        "n_train": int(len(labels_train)),
        "n_test": int(len(labels_test)),
    }


def cross_validated_report(
    features: np.ndarray,
    labels: np.ndarray,
    classifier: str = "random_forest",
    n_splits: int = 5,
    random_state: int = 0,
) -> dict[str, float]:
    """Stratified K-fold over repertoires, matching the paper's 5-fold protocol."""
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    per_fold = []
    for train_index, test_index in splitter.split(features, labels):
        per_fold.append(
            classify_repertoires(
                features[train_index],
                labels[train_index],
                features[test_index],
                labels[test_index],
                classifier=classifier,
            )
        )
    keys = ["roc_auc", "pr_auc", "accuracy", "f1"]
    report = {}
    for key in keys:
        values = np.array([fold[key] for fold in per_fold])
        report[f"{key}_mean"] = float(values.mean())
        report[f"{key}_std"] = float(values.std())
    report["n_splits"] = n_splits
    return report


def precision_at_k(
    scores: np.ndarray,
    is_signal: np.ndarray,
    k_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precision among the top-k sequences ranked by signal intensity.

    Mirrors the paper's Fig. 2A: ``k_max`` defaults to the number of signal-positive
    sequences, so the curve runs over ``k / k_max`` in [0, 1] and the witness rate is
    the random-guessing control.

    Returns ``(k_values, precisions)``.
    """
    order = np.argsort(-np.asarray(scores))
    ranked = np.asarray(is_signal, dtype=bool)[order]
    k_max = int(ranked.sum()) if k_max is None else int(k_max)
    if k_max < 1:
        return np.array([]), np.array([])
    k_values = np.arange(1, k_max + 1)
    return k_values, np.cumsum(ranked[:k_max]) / k_values


def signal_enrichment(
    scores: np.ndarray,
    is_signal: np.ndarray,
    top_fractions: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
) -> dict[str, float]:
    """Fold-enrichment of known signal sequences in the top-scoring fractions.

    An enrichment of 1.0 is chance; the witness rate itself is the baseline
    precision, so this is the scale-free version of ``precision_at_k``.
    """
    scores = np.asarray(scores)
    is_signal = np.asarray(is_signal, dtype=bool)
    witness_rate = is_signal.mean()
    order = np.argsort(-scores)
    ranked = is_signal[order]
    report = {"witness_rate": float(witness_rate), "n_signal": int(is_signal.sum())}
    if witness_rate <= 0:
        return report
    report["roc_auc"] = float(roc_auc_score(is_signal, scores))
    for fraction in top_fractions:
        k = max(int(round(fraction * len(scores))), 1)
        precision = float(ranked[:k].mean())
        report[f"precision@{fraction:g}"] = precision
        report[f"enrichment@{fraction:g}"] = precision / float(witness_rate)
    return report
