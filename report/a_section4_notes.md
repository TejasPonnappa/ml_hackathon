# Section 4 notes — Bayesian hyperparameter optimisation (Person A)

Sources: `src/ml/bayesopt_lightgbm.py`, `results/a_bayesopt_trials.csv`, `results/a_bayesopt_best.json`,
`results/a_bayesopt_history.png`, `results/a_bayesopt_comparison.csv`, `results/a_cv_lightgbm.json`,
`results/a_cv_lightgbm_tuned.json`, `results/a_bench_inference.csv`.
Command: `python -m src.ml.bayesopt_lightgbm --trials 20` (run in the terminal by Tejas).

## Model chosen for optimisation and why
LightGBM had the highest CV observation-level Top-1 (0.8865) and, among the three tree models that were
statistically tied on accuracy (RF, XGBoost, LightGBM; CV macro F1 within 0.001, fold std ≈ 0.018),
the lowest CV log loss (0.0926). It was also the fastest tree model to train (≈10 s full fit vs 26 s
XGBoost, 162 s RF), which makes many optimisation trials affordable. See `results/a_model_comparison.csv`.

## 1–2. Hyperparameters and ranges searched
| Hyperparameter | Range / distribution | Untuned value | Reason for range |
|---|---|---|---|
| learning_rate | 0.02 – 0.2, log-uniform | 0.05 | step size; log scale because its effect is multiplicative |
| n_estimators | 150 – 800, step 50 | 400 | must trade off against learning rate |
| num_leaves | 15 – 255, log | 63 | main complexity knob of leaf-wise trees (under- to over-fitting) |
| min_child_samples | 10 – 300, log | 50 | minimum rows per leaf; regularises noisy GPS labels |
| subsample | 0.5 – 1.0 | 0.8 | row bagging per round |
| colsample_bytree | 0.5 – 1.0 | 0.8 | feature bagging per tree |
| reg_lambda | 0.001 – 10, log | 1.0 | L2 penalty on leaf weights |
| reg_alpha | 0.001 – 10, log | (0) | L1 penalty on leaf weights |
Fixed: objective = binary (log loss), subsample_freq = 1, seed 42.

## 3. Objective
Minimise the **mean validation log loss over the same 5-fold GroupKFold-by-trip split used for every model**,
on the train partition only (78 trips after removing the two anomalous ones). Log loss was chosen because the
HMM decoder consumes the classifier probabilities, so probability quality matters, and because it is a smooth
objective. Macro F1 and Top-1 were recorded for every trial as well. Student-test and evaluator-test data were
not used in the search.

Optimiser: Optuna TPE sampler (Bayesian, Tree-structured Parzen Estimator; 5 random start-up trials, seed 42).
Trial 0 was the untuned configuration, so the search had to beat it. A MedianPruner stopped trials whose running
mean fold loss was worse than the median of earlier trials at the same fold.

## 4. Number of trials and compute cost
20 trials: 6 completed all 5 folds, 14 were pruned (after 2–3 folds). Total search time 868 s (14.5 min) on an
8-thread laptop CPU; a completed trial took 42–143 s. Note: values of pruned trials in the CSV are means over only
2–3 folds and are not comparable with 5-fold values.

## 5. Best hyperparameters (trial 4, CV log loss 0.09220)
learning_rate 0.0572, n_estimators 650, num_leaves 26, min_child_samples 56, subsample 0.796,
colsample_bytree 0.523, reg_lambda 0.269, reg_alpha 0.0048.
Compared with the untuned model: many more but much smaller trees (650 × 26 leaves vs 400 × 63) and stronger
feature subsampling.

## 6. Did the tuned model outperform the untuned one?
Both refitted and evaluated with the identical harness (same folds, same metrics, same partitions).

| | CV macro F1 (± std) | CV Top-1 | CV log loss | ST Top-1 | ST macro F1 | ST log loss | ST route continuity | Inference, 10 ST trips (median of 7) |
|---|---|---|---|---|---|---|---|---|
| Untuned | 0.9297 ± 0.0175 | 0.8865 | 0.0926 | 0.8895 | 0.9313 | 0.0839 | 0.9813 | 0.438 s |
| Tuned | 0.9301 ± 0.0177 | 0.8874 | 0.0922 | 0.8873 | 0.9304 | 0.0851 | 0.9825 | 0.556 s |

- On cross-validation (the selection criterion) the tuned model is marginally better on every metric
  (macro F1 +0.0004, Top-1 +0.0009, log loss −0.0004), but all differences are far smaller than the
  fold-to-fold standard deviation (≈0.018).
- On the 10 student-test trips it is marginally worse on Top-1 (−0.0022) and log loss (+0.0012) and
  marginally better on route continuity (+0.0011) — again within noise.
- Inference is ≈27 % slower (more trees), still only ≈0.06 s per trip.
- **Conclusion:** the untuned configuration was already close to optimal for this feature set; tuning did not
  give a meaningful improvement. Remaining errors are limited by the candidate set and features (e.g. two-way
  twin arcs, junctions), not by LightGBM's hyperparameters. Following the pre-declared CV-based selection rule,
  the tuned model was exported (`artifacts/model.joblib`, `artifacts/model_config.json`); the difference to the
  untuned model has no practical effect on the system.

Inference times: harness single timings depend on machine load (the untuned model was re-run while another
model was training), so the fair comparison uses `python -m src.ml.bench_inference lightgbm lightgbm_tuned`
(same process, same data, median of 7 runs).
