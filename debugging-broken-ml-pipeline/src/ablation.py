"""Ablation: fix one flaw at a time and watch the numbers move.

The broken pipeline's headline accuracy (~1.00) is fake for TWO
reasons at once — the leaky feature AND the one-class test set — so a
single before/after comparison hides which flaw did what. This script
runs five intermediate variants (all with the same KNN model, so only
the flaw fixes differ) plus the fully fixed pipeline, and records the
honest metrics of each. The results feed the ablation table in
REPORT.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.common import compute_metrics, group_stratified_split, sorted_split
from src.data import FEATURES_ALL, FEATURES_LEGIT, TARGET, load_dataset
from src.fixed_pipeline import run_fixed


def run_knn_variant(
    df,
    *,
    use_leaky: bool,
    split: str,
    scaler_scope: str,
    apply_scaling: bool,
) -> dict:
    """One KNN run with the flaw switches set explicitly.

    split: "sorted" (broken) or "group" (fixed).
    scaler_scope: "full" (fit before split) or "train" (fit after split).
    apply_scaling: whether the model actually sees the scaled features.
    """
    features = list(FEATURES_ALL) if use_leaky else list(FEATURES_LEGIT)

    if split == "sorted":
        train_df, test_df = sorted_split(df, TARGET)
    else:
        from src.data import GROUP_COL

        train_df, test_df = group_stratified_split(df, TARGET, GROUP_COL)

    X_train_raw = train_df[features].to_numpy(dtype=float)
    X_test_raw = test_df[features].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    if scaler_scope == "full":
        scaler = StandardScaler().fit(df[features].to_numpy(dtype=float))
    else:
        scaler = StandardScaler().fit(X_train_raw)

    X_train = scaler.transform(X_train_raw) if apply_scaling else X_train_raw
    X_test = scaler.transform(X_test_raw) if apply_scaling else X_test_raw

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_proba)


def run_ablation(df=None) -> list[dict]:
    """Run every variant and return a list of {variant, metrics} rows."""
    if df is None:
        df = load_dataset()

    variants = [
        (
            "Broken (all 7 flaws)",
            dict(use_leaky=True, split="sorted", scaler_scope="full", apply_scaling=False),
        ),
        (
            "Fix ordering only (leak still in)",
            dict(use_leaky=True, split="group", scaler_scope="full", apply_scaling=False),
        ),
        (
            "Fix leakage only (test still one class)",
            dict(use_leaky=False, split="sorted", scaler_scope="full", apply_scaling=False),
        ),
        (
            "Fix leakage + ordering",
            dict(use_leaky=False, split="group", scaler_scope="full", apply_scaling=False),
        ),
        (
            "+ scale properly (train-only, applied)",
            dict(use_leaky=False, split="group", scaler_scope="train", apply_scaling=True),
        ),
    ]
    rows = []
    for name, kwargs in variants:
        metrics = run_knn_variant(df, **kwargs)
        rows.append({"variant": name, "metrics": metrics})

    fixed = run_fixed(df)
    rows.append({"variant": "Fully fixed (LR + balanced + scaled)", "metrics": fixed["metrics"]})
    return rows


def print_ablation(rows: list[dict]) -> None:
    print(f"{'Variant':<42}{'Acc':>7}{'BalAcc':>8}{'F1':>7}{'ROC-AUC':>9}")
    print("-" * 73)
    for row in rows:
        m = row["metrics"]
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "  n/a "
        print(
            f"{row['variant']:<42}{m['accuracy']:>7.4f}{m['balanced_accuracy']:>8.4f}"
            f"{m['f1']:>7.4f}{roc:>9}"
        )


if __name__ == "__main__":
    print_ablation(run_ablation())
