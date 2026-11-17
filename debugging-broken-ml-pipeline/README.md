# Debugging a Broken ML Pipeline

A churn classifier pipeline with **7 deliberate flaws** — find every one,
fix it, and show how the reported performance changes. The broken pipeline
reports **accuracy 1.00**. The honest number is F1 **0.00** on the same run.
The fixed pipeline reports accuracy 0.68, F1 0.38, ROC-AUC 0.74 — lower,
and real.

## What each part does

| File | What it does | Why it exists |
| --- | --- | --- |
| `src/data.py` | Generates the synthetic dataset (seed 42) and loads it | No downloads; the same 2,000 rows every run |
| `src/broken_pipeline.py` | The original pipeline, flaws and all | The exhibit — do not "fix" this file |
| `src/fixed_pipeline.py` | The same task, done correctly | The reference solution |
| `src/common.py` | Shared splits and honest metric computation | So the two pipelines differ only in the flaws |
| `src/detect.py` | Checks for each flaw programmatically | Proof the flaws are real, not just claimed |
| `src/ablation.py` | Fixes one flaw at a time and records the numbers | Shows which flaw did what to the score |
| `src/plots.py` | Before/after bar chart | One picture for the demo |
| `src/cli.py` | Command line: `generate`, `run-broken`, `run-fixed`, `detect`, `evaluate` | How you run everything |
| `tests/test_pipeline.py` | 29 tests | Each flaw exists in broken, is gone in fixed |
| `data/customers.csv` | The generated dataset (2,000 rows, committed) | Runs without regenerating; recreate with `generate` |
| `results/` | `metrics.json`, `ablation.json`, `plots/before_after.png` | The measured before/after numbers |
| `REPORT.md` | The required document: each issue, why it matters, how it was detected | The write-up for the submission |

## The dataset

2,000 rows of synthetic customer churn (seed 42), imbalanced 85/15
(311 churned / 1,689 not). Columns:

- `customer_id` — group id; each customer appears 5 times
- `day` — time/order column
- `age` (18–90), `income` (0–10,000), `score_small` (0–1), `visits` (0–20)
- `post_outcome_score` — **leaky**: a score only known after churn happens
  (correlation 0.99 with the target). Never an input.
- `churned` — the target

## The 7 flaws (summary — full detail in REPORT.md)

1. **Target leakage** — the post-outcome score is used as an input.
2. **Scaler fit on the full data** before the split (2,000 rows, test included).
3. **Wrong split** — no stratification, customer groups ignored (265 customers
   in both train and test).
4. **Unscaled features to a distance model** — the scaler is fit but never
   applied; KNN sees raw 0–10,000 columns.
5. **Imbalance ignored** — no `class_weight`, no resampling.
6. **Wrong metric** — accuracy is the only number reported, on 85/15 data.
7. **Data-ordering bug** — rows sorted by label, split without shuffling, so
   the test set is 100% one class.

## How to run it

```bash
python -m venv .venv
# Windows PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.cli run-broken    # reported accuracy 1.00, honest F1 0.00
python -m src.cli run-fixed     # accuracy 0.68, F1 0.38, ROC-AUC 0.74
python -m src.cli detect        # flaw table: broken 7/7, fixed 0/7
python -m src.cli evaluate      # both + ablation, writes results/
python -m pytest tests/ -q      # 29 tests
```

`data/customers.csv` is committed, so the project runs as-is. To regenerate
it from the seed: `python -m src.cli generate`.

## Results (measured, seed 42)

| | Accuracy | Balanced acc. | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Broken (reported) | **1.0000** | — | — | — |
| Broken (honest, same run) | 1.0000 | 1.0000 | **0.0000** | undefined |
| Fixed | 0.6775 | 0.6537 | **0.3768** | **0.7397** |

The fixed model's accuracy (0.68) is below the naive "predict no churn for
everyone" baseline (0.85) — because `class_weight="balanced"` trades raw
accuracy for actually catching churners (recall 0.62). Accuracy alone
cannot tell a good imbalanced-data model from a useless one; F1 and
ROC-AUC can. Full numbers and the flaw-by-flaw ablation are in REPORT.md.
