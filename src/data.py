"""Synthetic dataset for the broken-pipeline debugging project.

Generates a small customer-churn-style table with everything a broken
pipeline needs to trip over:

* binary target, imbalanced ~85/15
* features on wildly different scales (0-1 vs 0-10,000)
* a time/order column (``day``)
* a group id (``customer_id`` — each customer appears 5 times)
* one LEAKY feature (``post_outcome_score``) derived from the target:
  a score that is only known AFTER the outcome, so it must never be
  used as an input.

The file is generated with a fixed seed (42) so every run reproduces
the same rows. The CSV is committed to the repo; regenerate it any
time with ``python -m src.cli generate``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

N_ROWS = 2000
N_CUSTOMERS = 400
ROWS_PER_CUSTOMER = 5
SEED = 42

LEAKY_FEATURE = "post_outcome_score"
TARGET = "churned"
GROUP_COL = "customer_id"

FEATURES_ALL = [
    "age",
    "income",
    "score_small",
    "visits",
    "post_outcome_score",
]
# Everything except the leaky feature: the only features a model may use.
FEATURES_LEGIT = [f for f in FEATURES_ALL if f != LEAKY_FEATURE]


def generate_dataset(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    """Build the synthetic dataset. Returns a DataFrame with 2000 rows."""
    rng = np.random.default_rng(seed)

    # Each customer appears exactly 5 times (group structure).
    customer_id = np.repeat(np.arange(N_CUSTOMERS), ROWS_PER_CUSTOMER)
    rng.shuffle(customer_id)
    day = rng.permutation(n_rows)  # time/order column

    # Features on wildly different scales.
    age = rng.integers(18, 90, size=n_rows).astype(float)
    income = rng.lognormal(mean=8.0, sigma=0.7, size=n_rows)
    income = np.clip(income, 0, 10_000)
    score_small = rng.uniform(0.0, 1.0, size=n_rows)
    visits = rng.poisson(lam=6.0, size=n_rows).astype(float)

    # True churn signal: depends on the four legit features only.
    logit = (
        -5.95
        + 0.020 * age
        + 0.00035 * income
        + 1.8 * score_small
        + 0.10 * visits
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(n_rows) < prob).astype(int)

    # LEAKY feature: a post-outcome score, only known after churn happens.
    # Near-perfect proxy for the target — using it is cheating.
    post_outcome_score = np.where(
        target == 1,
        rng.normal(8500, 400, size=n_rows),
        rng.normal(600, 400, size=n_rows),
    )
    post_outcome_score = np.clip(post_outcome_score, 0, 10_000)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "day": day,
            "age": age,
            "income": np.round(income, 2),
            "score_small": np.round(score_small, 4),
            "visits": visits,
            "post_outcome_score": np.round(post_outcome_score, 2),
            "churned": target,
        }
    )
    return df


def default_data_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "data" / "customers.csv"


def save_dataset(path: pathlib.Path | str | None = None, seed: int = SEED) -> pathlib.Path:
    path = pathlib.Path(path) if path else default_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(seed=seed)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: pathlib.Path | str | None = None) -> pd.DataFrame:
    path = pathlib.Path(path) if path else default_data_path()
    return pd.read_csv(path)
