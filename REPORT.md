# Debugging a Broken ML Pipeline — Report

**Task:** a pipeline with at least 6 deliberate flaws. Find every flaw, fix
it, and show how the reported performance changes.

**Data:** 2,000 synthetic customer rows (seed 42), binary churn target,
imbalanced 85/15 (311 churned / 1,689 not). Features on wildly different
scales (`score_small` 0–1 vs `income` 0–10,000), a time column (`day`), a
group id (`customer_id`, 5 rows each), and one leaky feature
(`post_outcome_score`, correlation 0.99 with the target).

**Headline result:** the broken pipeline reports accuracy **1.0000**. On the
same run its honest F1 is **0.0000**. The fixed pipeline reports accuracy
**0.6775**, balanced accuracy **0.6537**, F1 **0.3768**, ROC-AUC **0.7397**.

## Before / after

| | Accuracy | Balanced acc. | F1 | ROC-AUC | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| Broken (as reported) | **1.0000** | — | — | — | — |
| Broken (honest, same run) | 1.0000 | 1.0000 | **0.0000** | undefined | [[400, 0], [0, 0]] |
| Fixed | 0.6775 | 0.6537 | **0.3768** | **0.7397** | [[232, 105], [24, 39]] |

Read the broken row carefully: accuracy 1.00 **and** F1 0.00 at the same
time. The test set contains zero churners, so a model that predicts "no
churn" for everyone scores perfectly and learns nothing. Even balanced
accuracy reads 1.00 — every metric computed on a one-class test set is
meaningless, which is why the test set itself is the first thing to check.

## The 7 flaws

### 1. Target leakage — the post-outcome score is used as an input

- **What it is:** `post_outcome_score` is a score assigned *after* the
  customer churns (or doesn't). It is in the feature list, so the model
  trains and predicts on information that does not exist at prediction
  time. Its correlation with the target is 0.99.
- **Why it matters:** the model is peeking at the answer. In production
  the column would not be available, so the deployed model would be far
  worse than the reported number — and nobody would know until it shipped.
- **How detected:** correlation check in the tests
  (`test_leaky_feature_is_a_near_perfect_proxy`, r = 0.9909); the detector
  checks whether the feature is in the model's input list.
- **What the fix changed:** dropping it alone does not fix the headline
  number (see the ablation — the one-class test set keeps accuracy
  inflated), but it is the flaw that makes the *honest* numbers fake: with
  the split fixed and the leak still in, accuracy is still 1.00 with F1
  1.00. The leak alone is enough for a perfect score.

### 2. Scaler fit on the full data before the split

- **What it is:** `StandardScaler` is fit on all 2,000 rows — including the
  400 test rows — before any split happens.
- **Why it matters:** the test set's mean and spread leak into training.
  The effect is small here (a scaler is a weak leak) but the pattern is
  the dangerous one: any preprocessing fit before the split contaminates
  the evaluation. With a stronger preprocessor (e.g. target encoding, or
  imputation learned from the target) the same mistake silently inflates
  results by a lot.
- **How detected:** the detector reads `scaler.n_samples_seen_`: broken
  2,000 (= all rows), fixed 1,600 (= train rows only). Tests
  `test_broken_scaler_fit_on_full_data` /
  `test_fixed_scaler_fit_on_train_only`.
- **What the fix changed:** the scaler moved inside a sklearn `Pipeline`,
  so it is fit on the training rows only, after the split. On its own this
  moves the numbers by almost nothing — which is itself the lesson: a leak
  can be real and still small, and only the honest comparison shows it.

### 3. Wrong split — not stratified, customer groups ignored

- **What it is:** the split is a plain contiguous cut with no
  stratification on the 85/15 target and no respect for `customer_id`.
- **Why it matters:** two failures at once. Without stratification an
  imbalanced split can starve the test set of the minority class (here it
  did — see flaw 7). Without group awareness, the same customer appears in
  both train and test (265 customers overlap), so the model is partly
  evaluated on people it already saw — memorisation scores as
  generalisation.
- **How detected:** the detector counts group overlap (broken 265, fixed
  0) and compares the test positive rate to the overall rate (broken 0.000
  vs 0.155, fixed 0.158 vs 0.155).
- **What the fix changed:** `StratifiedGroupKFold` — the test set keeps
  the 85/15 balance and shares zero customers with train. This is the fix
  that makes every other metric trustworthy.

### 4. Unscaled features fed to a distance-based model

- **What it is:** the scaler is fit (flaw 2) but never applied — KNN is
  trained on the raw features. `income` (0–10,000) has a standard
  deviation of ~2,400 in the model input; `score_small` (0–1) has one of
  ~0.3. Distances are decided by `income` almost alone.
- **Why it matters:** for any distance-based model (KNN, SVM, k-means),
  unscaled features mean the biggest-numbered column wins, regardless of
  how informative it is. A genuinely useful 0–1 feature is effectively
  deleted from the model.
- **How detected:** the detector measures the standard deviation of the
  `income` column in what the model actually saw: broken 2406.0 (raw),
  fixed 1.0 (scaled). Tests `test_broken_model_input_is_unscaled` /
  `test_fixed_model_input_is_scaled`.
- **What the fix changed:** scaling alone barely moved KNN (ablation:
  F1 0.2353 → 0.2366). Honest finding, not a failure of the fix: in this
  data the big-scale column (`income`) happens to be informative, so
  letting it dominate was not very costly. The fix still matters — it
  makes the model behave as designed instead of by accident — but the
  numbers say the real gains came from the split, the leak removal and
  the balanced model, not from scaling.

### 5. Label imbalance ignored

- **What it is:** no `class_weight`, no resampling, no stratification.
  With 85% of rows in one class, a model can score 0.85 accuracy by
  predicting "no churn" for everyone and learning nothing.
- **Why it matters:** accuracy rewards the lazy answer. On imbalanced
  data the model you ship is the one that never catches the rare event —
  the churners, the fraud cases, the sick patients — which is usually the
  whole point of the model.
- **How detected:** the detector checks `class_weight` (broken `None`,
  fixed `"balanced"`); the fixed run's confusion matrix shows the trade:
  39 of 63 churners caught (recall 0.62) at the cost of 105 false alarms.
- **What the fix changed:** `class_weight="balanced"` in the logistic
  regression. Accuracy *fell* (0.84 → 0.68 against the honest KNN
  comparison) while F1 rose (0.24 → 0.38) and ROC-AUC rose (0.67 → 0.74).
  The fixed accuracy is even below the naive all-negative baseline
  (0.85) — which is expected and correct: the baseline's F1 is 0.00. A
  model that catches churners is worth more than a model that pads
  accuracy by ignoring them.

### 6. Wrong metric — accuracy reported as the only metric

- **What it is:** the broken pipeline prints one number, accuracy, and
  nothing else — on an 85/15 problem.
- **Why it matters:** accuracy cannot distinguish a good imbalanced-data
  model from a useless one. The broken run proves it: accuracy 1.00, F1
  0.00, on the same predictions. A single headline metric is not a
  summary; it is a hiding place.
- **How detected:** the detector checks the reported-metrics list
  (broken `["accuracy"]`, fixed five metrics). The honest metrics for the
  broken run are computed in the same script — the gap between the
  reported number and the honest ones *is* the detection.
- **What the fix changed:** the fixed pipeline reports accuracy, balanced
  accuracy, F1, ROC-AUC and the confusion matrix together. No single one
  of them tells the story; together they do.

### 7. Data-ordering bug — sorted by label, split without shuffling

- **What it is:** the rows are sorted by the target and cut into a
  contiguous 80/20 split with no shuffling. The test set is the first 400
  rows of the sorted frame — 400 non-churners, zero churners.
- **Why it matters:** it is the flaw that makes all the others invisible.
  A one-class test set makes accuracy, balanced accuracy and F1 all
  meaningless at once (accuracy 1.00, F1 0.00), and it hides the leak
  (flaw 1) because the leak never gets a fair test either. Ordering bugs
  are silent: no error, no warning, just a number that looks great.
- **How detected:** the detector checks the test set's classes (broken
  `[0]`, fixed `[0, 1]`) and the split method. The test
  `test_broken_test_set_is_single_class` pins it down.
- **What the fix changed:** no sorting by label anywhere; the split
  shuffles under the group/stratification constraints of flaw 3. Fixing
  this flaw alone (ablation row 2) changed accuracy not at all — it stayed
  1.00 — because the leak was still in. That is the point of the ablation:
  one fix at a time shows which flaw was actually carrying the fake score.

## Ablation — fixing one flaw at a time

Same KNN model throughout (except the last row), so only the flaw fixes
differ. All numbers measured on the same data, seed 42.

| Variant | Accuracy | Balanced acc. | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Broken (all 7 flaws) | 1.0000 | 1.0000 | 0.0000 | undefined |
| Fix ordering only (leak still in) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Fix leakage only (test still one class) | 0.9575 | 0.9575 | 0.0000 | undefined |
| Fix leakage + ordering | 0.8375 | 0.5616 | 0.2353 | 0.6734 |
| + scale properly (train-only, applied) | 0.8225 | 0.5591 | 0.2366 | 0.6369 |
| Fully fixed (logistic regression + balanced + scaled) | 0.6775 | 0.6537 | 0.3768 | 0.7397 |

What the ablation shows:

- **The fake 1.00 had two independent causes.** Fixing the ordering alone
  left accuracy at 1.00 (the leak is that strong — F1 1.00 proves it).
  Fixing the leak alone left accuracy at 0.96 (the one-class test set is
  that forgiving — F1 still 0.00). Only fixing *both* moved the honest
  numbers into a believable range.
- **Scaling barely mattered here** (F1 0.235 → 0.237). Reported honestly
  rather than hidden: the big-scale feature was informative, so the
  unscaled model was accidentally fine.
- **The balanced model is where the honest gain is:** F1 0.24 → 0.38,
  ROC-AUC 0.67 → 0.74, at the cost of raw accuracy — the correct trade on
  imbalanced data.

## How each flaw was detected (summary)

| # | Flaw | Detection method |
| --- | --- | --- |
| 1 | Target leakage | Feature-target correlation (0.99); input-list check |
| 2 | Scaler fit on full data | `scaler.n_samples_seen_` (2,000 vs 1,600) |
| 3 | Wrong split | Group overlap count (265 vs 0); test vs overall positive rate |
| 4 | Unscaled features | Std of `income` in the model input (2406 vs 1.0) |
| 5 | Imbalance ignored | `class_weight` is `None`; confusion matrix shows no churners caught |
| 6 | Wrong metric | Reported-metrics list; honest metrics computed on the same run |
| 7 | Ordering bug | Test-set classes `[0]`; split method recorded in the run |

Run the checks yourself: `python -m src.cli detect` (broken 7/7, fixed 0/7),
`python -m pytest tests/ -q` (29 tests, all passing).
