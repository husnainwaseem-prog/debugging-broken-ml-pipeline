"""Programmatic flaw detector.

For each of the 7 known flaws it checks the evidence in a pipeline
run result (broken or fixed) and says whether the flaw is present.
Run it with ``python -m src.cli detect`` — it prints a side-by-side
table for the broken vs the fixed pipeline.
"""

from __future__ import annotations

import numpy as np

from src.broken_pipeline import run_broken
from src.data import LEAKY_FEATURE, TARGET, load_dataset
from src.fixed_pipeline import run_fixed

# Index of the big-scale column ("income", 0-10,000) inside the feature
# list. Raw income has a std in the thousands; scaled income ~1.0.
INCOME_INDEX = 1


def _income_std(result: dict) -> float:
    """Std of the 'income' column in what the model ACTUALLY saw:
    the raw features for the broken run, the scaler output for the
    fixed run (its scaler lives inside the Pipeline)."""
    X = result["X_train_used"]
    if result["scaled"]:
        X = result["scaler"].transform(X)
    return float(np.std(X[:, INCOME_INDEX]))


def detect_flaws(result: dict, df=None) -> list[dict]:
    """Check all 7 flaws against one pipeline run. Returns a list of
    dicts: {id, name, present, evidence}."""
    if df is None:
        df = load_dataset()
    overall_pos_rate = float(df[TARGET].mean())
    test_pos_rate = float(np.mean(result["y_test"]))
    checks = []

    checks.append(
        {
            "id": 1,
            "name": "Target leakage (leaky feature used as input)",
            "present": bool(result["uses_leaky_feature"]),
            "evidence": f"features used: {result['features']}",
        }
    )

    checks.append(
        {
            "id": 2,
            "name": "Scaler fit on the FULL data before the split",
            "present": result["scaler_n_samples_seen"] == result["n_rows_total"],
            "evidence": (
                f"scaler saw {result['scaler_n_samples_seen']} rows "
                f"of {result['n_rows_total']} total "
                f"(train size would be {len(result['y_train'])})"
            ),
        }
    )

    stratified_ok = abs(test_pos_rate - overall_pos_rate) <= 0.05
    checks.append(
        {
            "id": 3,
            "name": "Wrong split: not stratified, groups ignored",
            "present": (result["group_overlap"] > 0) or (not stratified_ok),
            "evidence": (
                f"customers in both train and test: {result['group_overlap']}; "
                f"test positive rate {test_pos_rate:.3f} vs overall "
                f"{overall_pos_rate:.3f}"
            ),
        }
    )

    income_std = _income_std(result)
    checks.append(
        {
            "id": 4,
            "name": "Unscaled features fed to a distance-based model",
            "present": (not result["scaled"]) and income_std > 100,
            "evidence": (
                f"model input 'income' std = {income_std:.1f} "
                f"(raw = thousands, scaled = ~1.0); scaled={result['scaled']}"
            ),
        }
    )

    checks.append(
        {
            "id": 5,
            "name": "Label imbalance ignored (no class_weight / balancing)",
            "present": result["class_weight"] is None,
            "evidence": f"class_weight = {result['class_weight']!r}",
        }
    )

    checks.append(
        {
            "id": 6,
            "name": "Wrong metric: accuracy reported as the only metric",
            "present": result["reported_metrics"] == ["accuracy"],
            "evidence": f"reported metrics: {result['reported_metrics']}",
        }
    )

    single_class_test = len(np.unique(result["y_test"])) == 1
    checks.append(
        {
            "id": 7,
            "name": "Data-ordering bug: sorted by label, split without shuffling",
            "present": single_class_test
            or result["split_method"] == "sorted_no_shuffle",
            "evidence": (
                f"split method: {result['split_method']}; "
                f"test set classes: {sorted(np.unique(result['y_test']).tolist())}"
            ),
        }
    )

    return checks


def print_report(broken: dict, fixed: dict, df=None) -> None:
    """Print a side-by-side flaw table for both pipelines."""
    if df is None:
        df = load_dataset()
    broken_checks = {c["id"]: c for c in detect_flaws(broken, df)}
    fixed_checks = {c["id"]: c for c in detect_flaws(fixed, df)}
    print(f"{'Flaw':<66}{'Broken':<10}{'Fixed':<10}")
    print("-" * 86)
    for flaw_id in sorted(broken_checks):
        b = broken_checks[flaw_id]
        f = fixed_checks[flaw_id]
        mark = lambda present: "PRESENT " if present else "absent  "
        print(f"{flaw_id}. {b['name']:<62}{mark(b['present']):<10}{mark(f['present']):<10}")
    print("-" * 86)
    n_broken = sum(c["present"] for c in broken_checks.values())
    n_fixed = sum(c["present"] for c in fixed_checks.values())
    print(f"Flaws present: broken {n_broken}/7, fixed {n_fixed}/7")
    print()
    for flaw_id in sorted(broken_checks):
        b = broken_checks[flaw_id]
        print(f"[{flaw_id}] {b['name']}")
        print(f"    broken evidence: {b['evidence']}")
        print(f"    fixed  evidence: {fixed_checks[flaw_id]['evidence']}")


if __name__ == "__main__":
    df = load_dataset()
    print_report(run_broken(df), run_fixed(df), df)
