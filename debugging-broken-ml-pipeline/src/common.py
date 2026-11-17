"""Shared helpers: splits, metrics, and the small pieces both the broken
and the fixed pipeline use. Keeping them here means the two pipelines
differ only in the (deliberate) flaws, not in plumbing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Honest metrics for a binary classifier. ROC-AUC is None when the
    test set holds a single class (it is undefined then)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # The broken run's test set holds a single class on purpose (flaw 7),
    # which makes sklearn warn about it. The warning is expected noise
    # here, so it is silenced inside this helper only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def sorted_split(df: pd.DataFrame, target: str, test_size: float = 0.2):
    """The broken split: sort by the label, then cut a contiguous block
    off the front with no shuffling. The test set ends up (nearly) one
    class only, and the same customer lands in train and test."""
    ordered = df.sort_values(target).reset_index(drop=True)
    n_test = int(len(ordered) * test_size)
    test = ordered.iloc[:n_test]
    train = ordered.iloc[n_test:]
    return train, test


def group_stratified_split(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    n_splits: int = 5,
    seed: int = 42,
):
    """The correct split: StratifiedGroupKFold — test keeps the same
    class balance as the whole data AND no customer appears in both
    train and test. Fold 0 is used as the test set."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(
        sgkf.split(df, df[target], groups=df[group_col])
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def group_overlap(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> int:
    """How many group ids appear in BOTH train and test (must be 0)."""
    return len(set(train[group_col]) & set(test[group_col]))
