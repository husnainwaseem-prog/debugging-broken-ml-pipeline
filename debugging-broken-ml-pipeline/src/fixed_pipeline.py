"""THE FIXED PIPELINE — the same task, done correctly.

Every flaw in ``broken_pipeline.py`` is fixed here:

1. The leaky ``post_outcome_score`` is dropped — it is only known
   after the outcome, so it can never be an input.
2. The scaler lives INSIDE a sklearn Pipeline, so it is fit on the
   training rows only, after the split.
3. The split is group-aware AND stratified (StratifiedGroupKFold):
   no customer appears in both train and test, and the test set keeps
   the 85/15 class balance.
4. Features are scaled before the model sees them, so the 0-1 column
   counts as much as the 0-10,000 column.
5. The imbalance is handled with ``class_weight="balanced"``.
6. Five honest metrics are reported — accuracy, balanced accuracy,
   F1, ROC-AUC and the confusion matrix — not accuracy alone.
7. The rows are never sorted by label; the split shuffles (with the
   group/stratification constraints above).

The reported numbers are lower than the broken pipeline's. That is
the point: they are honest.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import compute_metrics, group_overlap, group_stratified_split
from src.data import FEATURES_LEGIT, GROUP_COL, LEAKY_FEATURE, TARGET, load_dataset


def run_fixed(df=None) -> dict:
    """Run the fixed pipeline end to end. Returns the same shape of
    result dict as ``run_broken`` so the detector and tests can compare
    the two runs directly."""
    if df is None:
        df = load_dataset()

    # FIX 1 — drop the leaky feature. The model only sees information
    # that exists before the outcome.
    features = list(FEATURES_LEGIT)
    assert LEAKY_FEATURE not in features

    # FIX 3 + FIX 7 — group-aware, stratified split; no sorting by label.
    train_df, test_df = group_stratified_split(df, TARGET, GROUP_COL)
    X_train = train_df[features].to_numpy(dtype=float)
    X_test = test_df[features].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    # FIX 2 + FIX 4 — the scaler is a step inside the Pipeline, so it is
    # fit on the training rows only, and the model always sees scaled
    # features.
    # FIX 5 — class_weight="balanced" compensates for the 85/15 split.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    scaler = model.named_steps["scaler"]

    result = {
        "name": "fixed",
        "features": features,
        "uses_leaky_feature": LEAKY_FEATURE in features,
        "scaler": scaler,
        "scaler_n_samples_seen": int(np.asarray(scaler.n_samples_seen_).ravel()[0]),
        "n_rows_total": len(df),
        "split_method": "stratified_group_kfold",
        "stratified": True,
        "group_overlap": group_overlap(train_df, test_df, GROUP_COL),
        "scaled": True,
        "class_weight": "balanced",
        "X_train_used": X_train,
        "X_test_used": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "y_pred": y_pred,
        # FIX 6 — report the full honest picture, not accuracy alone.
        "reported_metrics": [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "roc_auc",
            "confusion_matrix",
        ],
        "reported_accuracy": metrics["accuracy"],
        "metrics": metrics,
    }
    return result


if __name__ == "__main__":
    res = run_fixed()
    print("Fixed pipeline — honest metrics:")
    for key, value in res["metrics"].items():
        print(f"  {key}: {value}")
