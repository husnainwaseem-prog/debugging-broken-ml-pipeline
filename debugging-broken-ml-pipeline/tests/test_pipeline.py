"""Tests for the broken vs fixed pipelines.

Each test pins down one fact the report claims: the flaw exists in the
broken pipeline and is gone in the fixed one, and the headline numbers
are what the code actually produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.ablation import run_ablation, run_knn_variant
from src.broken_pipeline import run_broken
from src.common import group_stratified_split, sorted_split
from src.data import (
    FEATURES_LEGIT,
    GROUP_COL,
    LEAKY_FEATURE,
    TARGET,
    generate_dataset,
    load_dataset,
)
from src.detect import detect_flaws
from src.fixed_pipeline import run_fixed


@pytest.fixture(scope="module")
def df():
    return load_dataset()


@pytest.fixture(scope="module")
def broken(df):
    return run_broken(df)


@pytest.fixture(scope="module")
def fixed(df):
    return run_fixed(df)


# --- dataset ------------------------------------------------------------

def test_dataset_shape_and_imbalance(df):
    assert len(df) == 2000
    assert set(df[TARGET].unique()) == {0, 1}
    pos_rate = df[TARGET].mean()
    assert 0.10 <= pos_rate <= 0.20, f"positive rate {pos_rate:.3f} not ~85/15"


def test_leaky_feature_is_a_near_perfect_proxy(df):
    corr = df[LEAKY_FEATURE].corr(df[TARGET])
    assert corr > 0.9, f"leaky feature correlation {corr:.3f} is too weak to inflate results"


def test_each_customer_appears_multiple_times(df):
    counts = df[GROUP_COL].value_counts()
    assert counts.min() == 5 and counts.max() == 5


def test_dataset_is_reproducible():
    a = generate_dataset(seed=42)
    b = generate_dataset(seed=42)
    pd.testing.assert_frame_equal(a, b)


# --- flaw 1: leakage ------------------------------------------------------

def test_broken_uses_leaky_feature(broken):
    assert broken["uses_leaky_feature"] is True
    assert LEAKY_FEATURE in broken["features"]


def test_fixed_drops_leaky_feature(fixed):
    assert fixed["uses_leaky_feature"] is False
    assert LEAKY_FEATURE not in fixed["features"]
    assert fixed["features"] == FEATURES_LEGIT


# --- flaw 2: scaler fit on full data --------------------------------------

def test_broken_scaler_fit_on_full_data(broken):
    assert broken["scaler_n_samples_seen"] == broken["n_rows_total"] == 2000


def test_fixed_scaler_fit_on_train_only(fixed):
    assert fixed["scaler_n_samples_seen"] == len(fixed["y_train"])
    assert fixed["scaler_n_samples_seen"] < fixed["n_rows_total"]


# --- flaw 3: wrong split ---------------------------------------------------

def test_broken_split_has_group_overlap(broken):
    assert broken["group_overlap"] > 0


def test_fixed_split_has_no_group_overlap(fixed):
    assert fixed["group_overlap"] == 0


def test_fixed_split_is_stratified(df, fixed):
    overall = df[TARGET].mean()
    test_rate = fixed["y_test"].mean()
    assert abs(test_rate - overall) <= 0.05


# --- flaw 4: unscaled features ----------------------------------------------

def test_broken_model_input_is_unscaled(broken):
    income_std = float(np.std(broken["X_train_used"][:, 1]))
    assert broken["scaled"] is False
    assert income_std > 100, f"income std {income_std:.1f} looks scaled"


def test_fixed_model_input_is_scaled(fixed):
    scaler = fixed["scaler"]
    scaled = scaler.transform(fixed["X_train_used"])
    assert fixed["scaled"] is True
    assert abs(float(np.std(scaled[:, 1])) - 1.0) < 0.05


# --- flaw 5: imbalance ignored ------------------------------------------------

def test_broken_ignores_imbalance(broken):
    assert broken["class_weight"] is None


def test_fixed_handles_imbalance(fixed):
    assert fixed["class_weight"] == "balanced"


# --- flaw 6: wrong metric -------------------------------------------------------

def test_broken_reports_only_accuracy(broken):
    assert broken["reported_metrics"] == ["accuracy"]


def test_fixed_reports_full_metrics(fixed):
    for name in ("accuracy", "balanced_accuracy", "f1", "roc_auc", "confusion_matrix"):
        assert name in fixed["reported_metrics"]


# --- flaw 7: ordering bug ----------------------------------------------------------

def test_broken_test_set_is_single_class(broken):
    assert sorted(np.unique(broken["y_test"]).tolist()) == [0]


def test_fixed_test_set_has_both_classes(fixed):
    assert set(np.unique(fixed["y_test"]).tolist()) == {0, 1}


# --- headline numbers -----------------------------------------------------------------

def test_broken_accuracy_is_inflated(broken):
    assert broken["reported_accuracy"] > 0.95


def test_broken_honest_f1_is_zero(broken):
    # The test set holds no churners, so the honest F1 is 0 even
    # though the reported accuracy is ~1.00.
    assert broken["metrics"]["f1"] == 0.0


def test_fixed_accuracy_below_broken(broken, fixed):
    assert fixed["metrics"]["accuracy"] < broken["reported_accuracy"]


def test_fixed_metrics_are_honest_and_nontrivial(fixed):
    m = fixed["metrics"]
    assert m["f1"] > 0.3, f"fixed F1 {m['f1']:.3f} too low"
    assert m["roc_auc"] > 0.7, f"fixed ROC-AUC {m['roc_auc']:.3f} too low"
    assert m["balanced_accuracy"] > 0.6
    cm = np.asarray(m["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() == len(fixed["y_test"])


def test_fixed_is_a_sklearn_pipeline_with_scaler(fixed):
    assert isinstance(fixed["model"], Pipeline)
    assert "scaler" in fixed["model"].named_steps


# --- detector ---------------------------------------------------------------------------

def test_detector_finds_all_flaws_in_broken(df, broken):
    found = {c["id"]: c["present"] for c in detect_flaws(broken, df)}
    assert len(found) == 7
    assert all(found.values()), f"detector missed flaws: {found}"


def test_detector_finds_no_flaws_in_fixed(df, fixed):
    found = {c["id"]: c["present"] for c in detect_flaws(fixed, df)}
    assert not any(found.values()), f"detector false positives: {found}"


# --- ablation --------------------------------------------------------------------------------

def test_ablation_fixing_ordering_exposes_the_leak(df):
    rows = {r["variant"]: r["metrics"] for r in run_ablation(df)}
    broken = rows["Broken (all 7 flaws)"]
    fix_ordering = rows["Fix ordering only (leak still in)"]
    # Accuracy stays perfect — the leak alone is enough for 1.00, which
    # is exactly what proves the leak is doing all the work.
    assert fix_ordering["accuracy"] == pytest.approx(broken["accuracy"])
    assert fix_ordering["accuracy"] > 0.90
    # But the honest metrics become measurable for the first time, and
    # they expose the leak: F1 0.00 -> 1.00, ROC-AUC undefined -> 1.00.
    assert broken["f1"] == 0.0
    assert fix_ordering["f1"] == pytest.approx(1.0, abs=0.001)
    assert fix_ordering["roc_auc"] == pytest.approx(1.0, abs=0.001)


def test_ablation_fixing_leakage_only_leaves_f1_at_zero(df):
    rows = {r["variant"]: r["metrics"] for r in run_ablation(df)}
    fix_leak = rows["Fix leakage only (test still one class)"]
    # Removing the leak changes nothing while the test set is still one
    # class: F1 stays 0 and accuracy stays inflated by the bad split.
    assert fix_leak["f1"] == 0.0
    assert fix_leak["accuracy"] > 0.95


def test_ablation_variants_run(df):
    metrics = run_knn_variant(
        df, use_leaky=False, split="group", scaler_scope="train", apply_scaling=True
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["roc_auc"] is not None
