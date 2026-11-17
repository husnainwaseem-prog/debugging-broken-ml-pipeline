"""THE BROKEN PIPELINE — the original code, flaws and all.

Do not "fix" this file: it is the exhibit. It contains 7 deliberate
flaws, each marked with a ``FLAW n`` comment. The fixed version lives
in ``fixed_pipeline.py``; ``detect.py`` checks for each flaw
programmatically; REPORT.md explains every one.

What it does: trains a k-nearest-neighbours churn classifier and
reports a single number — accuracy — that looks fantastic (about 1.00)
and is almost entirely fake.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.common import compute_metrics, group_overlap, sorted_split
from src.data import FEATURES_ALL, GROUP_COL, LEAKY_FEATURE, TARGET, load_dataset


def run_broken(df=None, test_size: float = 0.2) -> dict:
    """Run the broken pipeline end to end. Returns everything the
    detector, the tests and the report need to inspect."""
    if df is None:
        df = load_dataset()

    # FLAW 1 — TARGET LEAKAGE: the post-outcome score is used as an
    # input feature. It is only known AFTER churn happens, so the model
    # is peeking at the answer.
    features = list(FEATURES_ALL)
    X = df[features].to_numpy(dtype=float)
    y = df[TARGET].to_numpy()

    # FLAW 2 — SCALER FIT ON THE FULL DATA: the scaler learns the mean
    # and spread from all 2,000 rows, including the test set, before
    # any split happens. Test information leaks into training.
    scaler = StandardScaler()
    scaler.fit(X)

    # FLAW 7 — DATA-ORDERING BUG: the rows are sorted by the label and
    # split without shuffling, so the test set is (nearly) one class
    # only. Combined with FLAW 3 this also puts the same customer in
    # train and test.
    # FLAW 3 — WRONG SPLIT: no stratification on imbalanced data and
    # no respect for the customer groups.
    train_df, test_df = sorted_split(df, TARGET, test_size=test_size)
    # Rebuild X/y for the split rows (positions match the sorted order).
    ordered = df.sort_values(TARGET).reset_index(drop=True)
    n_test = int(len(ordered) * test_size)
    X_train_raw = ordered[features].to_numpy(dtype=float)[n_test:]
    X_test_raw = ordered[features].to_numpy(dtype=float)[:n_test]
    y_train = ordered[TARGET].to_numpy()[n_test:]
    y_test = ordered[TARGET].to_numpy()[:n_test]

    # FLAW 4 — UNSCALED FEATURES TO A DISTANCE MODEL: the scaler above
    # was fit but never applied — KNN is trained on the RAW features,
    # so the 0-10,000 columns drown out the 0-1 column completely.
    # FLAW 5 — IMBALANCE IGNORED: no class_weight, no resampling, no
    # stratification. The model is free to just predict the majority
    # class (and, on this test set, that is enough to look perfect).
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_raw, y_train)
    y_pred = model.predict(X_test_raw)
    y_proba = model.predict_proba(X_test_raw)[:, 1]

    honest = compute_metrics(y_test, y_pred, y_proba)

    result = {
        "name": "broken",
        "features": features,
        "uses_leaky_feature": LEAKY_FEATURE in features,
        "scaler": scaler,
        "scaler_n_samples_seen": int(np.asarray(scaler.n_samples_seen_).ravel()[0]),
        "n_rows_total": len(df),
        "split_method": "sorted_no_shuffle",
        "stratified": False,
        "group_overlap": group_overlap(train_df, test_df, GROUP_COL),
        "scaled": False,  # model input is the raw, unscaled X
        "class_weight": None,
        "X_train_used": X_train_raw,
        "X_test_used": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "y_pred": y_pred,
        # FLAW 6 — WRONG METRIC: accuracy is the ONLY number reported,
        # on an 85/15 imbalanced problem where predicting "no churn"
        # for everyone already scores ~0.85.
        "reported_metrics": ["accuracy"],
        "reported_accuracy": honest["accuracy"],
        "metrics": honest,
    }
    return result


if __name__ == "__main__":
    res = run_broken()
    print(f"Broken pipeline — reported accuracy: {res['reported_accuracy']:.4f}")
    print("Honest metrics on the same run:")
    for key, value in res["metrics"].items():
        print(f"  {key}: {value}")
