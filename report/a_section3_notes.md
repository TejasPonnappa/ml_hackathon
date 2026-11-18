# Section 3 notes — model comparison (3.1, 3.3) and LightGBM (3.2) (Person A: Tejas)

Sources: `results/a_model_comparison.csv`, `results/a_cv_<model>.json`, `results/a_fit_diagnostics.csv`,
plots `results/a_cmp_*.png`, `results/a_fit_diagnostics.png`, `results/a_cv_fold_stability.png`,
`results/a_loss_<model>.png`, `results/a_cm_<model>.png`. Code: `src/ml/harness.py`, `src/ml/train_*.py`,
`src/ml/compare_models.py`, `src/ml/fit_diagnostics.py`.

## 3.0 Common experimental protocol (applies to every model)
- Unit of classification: one (GPS observation, candidate arc) pair; label 1 if the candidate is the
  observation's reference arc (`src/common/labels.py`). Candidates: K = 10 nearest arcs within R = 50 m.
- 19 features (`src/ml/features.py`), identical for all models.
- Training data: 80 train trips minus 2 anomalous trips (00000054, 00000033) and minus observations > 50 m
  from their reference arc → 1,162,595 rows, 180,715 observations, 78 trips.
- 5-fold GroupKFold by trip_id (no trip in both training and validation). StandardScaler (LogReg, MLP only)
  fitted inside each fold on the training trips only. Metrics from out-of-fold predictions.
- Refit on all training trips, evaluation on all observations of the 10 student-test trips (unfiltered).
- Seed 42. Classification threshold 0.5 for candidate-level metrics; Top-1 and log loss use no threshold.

## 3.1 Overall comparison
| Model | Family | Key configuration | CV macro F1 (mean ± std over folds) | CV Top-1 | ST Top-1 | ST route continuity | ST macro F1 | ST log loss | Inference (ms / 1k candidate rows) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear / probabilistic | L2, C=1, lbfgs, standardised | 0.9045 ± 0.0164 | 0.8714 | 0.8606 | 0.9775 | 0.9043 | 0.1319 | 0.24 |
| MLP | Neural network | (32,16) ReLU, Adam lr 3e-3, batch 16,384, 15 epochs, 400k-row training sample, standardised | 0.9173 ± 0.0138 | 0.8714 | 0.8673 | 0.9793 | 0.9165 | 0.1046 | 0.35 |
| Random Forest | Bagging ensemble | 200 trees, depth ≤ 20, min leaf 5, sqrt features, 30 % bootstrap | 0.9289 ± 0.0172 | 0.8846 | 0.8803 | 0.9789 | 0.9285 | 0.0866 | 12.67 |
| XGBoost | Boosting (level-wise) | 400 rounds, depth 6, lr 0.05, 80 % rows/cols | 0.9289 ± 0.0179 | 0.8863 | 0.8901 | 0.9813 | 0.9310 | 0.0843 | 1.71 |
| LightGBM | Boosting (leaf-wise) | 400 rounds, 63 leaves, lr 0.05, min leaf 50, 80 % rows/cols | 0.9297 ± 0.0175 | 0.8865 | 0.8895 | 0.9813 | 0.9313 | 0.0839 | 4.18 |
| LightGBM tuned (§4) | Boosting (leaf-wise) | 650 rounds, 26 leaves, lr 0.057, min leaf 56 (Bayesian-optimised) | 0.9301 ± 0.0177 | 0.8874 | 0.8873 | 0.9825 | 0.9304 | 0.0851 | 5.93 |

Inference times above are single harness measurements and depend on machine load; the same-conditions
benchmark of the two LightGBM models is in `results/a_bench_inference.csv` (0.44 s vs 0.56 s for all 10 trips).
Route continuity here is for classifier-only (top-1 per observation) predictions, before the HMM.

### Primary metric and selection
- **Primary metric: CV observation-level Top-1 accuracy.** The task is to choose the right road segment for
  each GPS observation, which is exactly Top-1. CV (78 trips, 5 folds) rather than student-test is used for
  selection so the 10 student-test trips remain an independent check.
- **Tie-break: CV log loss**, because the HMM decoder consumes the probabilities (probability quality and
  decoder compatibility), when models are within one fold standard deviation.
- Result: Random Forest, XGBoost and LightGBM are statistically tied on CV accuracy (CV macro F1 within 0.001,
  fold std ≈ 0.017). LightGBM has the highest CV Top-1 and the lowest CV log loss, and it is the cheapest tree
  model to train (≈10 s full fit vs 26 s XGBoost and 162 s Random Forest) → **LightGBM selected** for Bayesian
  optimisation (§4); the tuned LightGBM is the deployed model (`artifacts/model.joblib`).
- The linear baseline is clearly worse (−2.5 points macro F1, −1.5 points CV Top-1, +55 % log loss), showing
  that non-linear feature interactions matter (e.g. heading only discriminates when the twin arc is equally
  close and the vehicle is moving).

### Overfitting / stability summary (`results/a_fit_diagnostics.csv`, `results/a_cv_fold_stability.png`)
| Model | train macro F1 (in-sample) | CV | student-test | train − CV | CV − test |
|---|---|---|---|---|---|
| LogReg | 0.9051 | 0.9042 | 0.9043 | 0.0009 | −0.0001 |
| MLP | 0.9212 | 0.9173 | 0.9165 | 0.0039 | 0.0008 |
| Random Forest | 0.9470 | 0.9285 | 0.9285 | 0.0184 | 0.0001 |
| XGBoost | 0.9461 | 0.9285 | 0.9310 | 0.0177 | −0.0025 |
| LightGBM | 0.9533 | 0.9293 | 0.9313 | 0.0240 | −0.0020 |
| LightGBM tuned | 0.9492 | 0.9297 | 0.9304 | 0.0195 | −0.0007 |
(CV column here = macro F1 of the pooled out-of-fold predictions.)
Fold ranking is identical across the full-data models (the same trip groups are hard for every model).
The MLP was trained on a 400k-row sample, so its GroupKFold assignment differs and its folds are not
comparable fold-by-fold.

## 3.2 LightGBM (selected model)

**A1. Why suitable.** Each GPS observation must be matched by combining heterogeneous evidence: distances,
direction agreement, neighbour consistency, rank among alternatives, junction structure. These interact
non-linearly (heading matters only when the vehicle moves and the twin arc is equally close). Gradient-boosted
trees model such interactions without feature scaling, cope with the 15 % positive rate, and optimise log loss
directly, so they output probabilities the HMM can use. LightGBM's histogram-based, leaf-wise growth makes it
fast on 1.16 M rows, which allowed 20 Bayesian-optimisation trials in 14.5 min.

**A2. Strengths and limitations observed.** Strengths: best CV Top-1 (0.8865) and lowest CV and student-test
log loss (0.0926 / 0.0839) of the five models; trains in ≈10 s; small model (2.7 MB untuned, 1.9 MB tuned).
Limitations: largest train–CV gap of all models (0.024 macro F1) — leaf-wise trees fit training trips closely;
accuracy is statistically tied with XGBoost and Random Forest, so its advantage is cost and probability
quality rather than accuracy; it scores each candidate independently, so it cannot enforce route connectivity
(continuity 0.9813) — that is left to the HMM; errors concentrate on trips with dense or parallel roads
(student-test trip 00000077: Top-1 0.747, trip 00000065: 0.852 vs 0.95 on trips 00000000/00000007/00000040).

**B5. Important hyperparameters and initial values.** n_estimators 400, learning_rate 0.05, num_leaves 63,
min_child_samples 50, subsample 0.8 (every round), colsample_bytree 0.8, reg_lambda 1.0, objective binary,
seed 42 (`src/ml/train_lightgbm.py`). Tuned values in §4.

**B6. Loss / objective.** Binary cross-entropy (log loss) minimised by gradient boosting; each new tree fits the
gradient of the log loss of the current ensemble, shrunk by the learning rate, with L2 regularisation on leaf
weights. Training and validation log loss were recorded every round (`results/a_loss_lightgbm.png`).

**C7–C8. Student-test performance.** Top-1 accuracy 0.8895 (0.8918 on the 99.74 % of observations whose
reference arc is among the candidates; candidate miss rate 0.26 %), macro F1 0.9313, log loss 0.0839,
accuracy 0.9659, precision 0.8852, recall 0.8797, route continuity 0.9813. Confusion matrix
[[TN 110,237, FP 2,181], [FN 2,299, TP 16,811]]. Tuned model: Top-1 0.8873, macro F1 0.9304, log loss 0.0851.

**C9. Over/underfitting evidence.** Mild overfitting, without harm to generalisation:
- in-sample macro F1 0.9533 vs CV 0.9293 (gap 0.024, the largest of the five models), but CV and student-test
  agree (0.9293 vs 0.9313), so performance on new trips is not degraded;
- the fold-1 learning curve: validation log loss falls from 0.419 to its minimum 0.1158 at round 229 and then
  rises slightly to 0.1170 at round 400 (+1.0 %) while training loss keeps falling to 0.0585 — the later rounds
  mostly fit the training trips. Bayesian optimisation chose smaller trees (26 leaves), which reduced the
  in-sample gap to 0.0195 at equal CV accuracy.
- Not underfitting: it has the best CV scores of all models and a training score clearly above LogReg.
