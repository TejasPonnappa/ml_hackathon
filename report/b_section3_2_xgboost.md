# Section 3.2 — XGBoost (DRAFT, Person B: Tejasvi)

Sources: `results/a_cv_xgboost.json`, `a_pertrip_xgboost.csv`, `a_loss_xgboost.png`, `a_cm_xgboost.png`,
`a_fit_diagnostics.csv`, `a_model_comparison.csv`, `results/logs/a_xgboost.log`. Code:
`src/ml/train_xgboost.py` (run through the shared harness `src/ml/harness.py`, same data, folds and
metrics as every other model).

## A. Model understanding

**1. Why did you consider this algorithm suitable for PathSnap?**
Each row of our table is one (GPS observation, candidate road) pair with 19 numeric features (distance to
the arc, position along it, heading difference, speed, distances of neighbouring observations, and so on),
and the label says whether that arc is the reference arc. This is tabular classification. Gradient-boosted
trees fit this kind of data well and can learn combinations such as "close to the arc *and* heading along
it". That matters here because about 92 % of the road network is two-way (`a_eda_summary.json`), and a
two-way road is stored as two arcs at the same position, so distance alone cannot separate them and the
heading features have to. The model needs no feature scaling (the features have different units), and it
outputs a probability per candidate, which the HMM uses as its emission score (section 5). We also
included it to check that the boosting result does not depend on one library: LightGBM grows trees
leaf-wise, XGBoost level-wise.

**2. Strengths and limitations for this task**
- Strengths: it was among the best models on student_test (Top-1 0.8901, the highest of the five
  untuned models; log loss 0.0843); it gave real per-round train/validation curves; fast inference
  (1.71 ms per 1,000 candidate rows vs 4.18 for LightGBM and 12.67 for Random Forest in the single harness
  measurements, which depend on machine load); it handles the class imbalance (15.5 % of training rows
  are positives) without re-weighting. Precision 0.8863 and recall 0.8776 on student_test are balanced.
- Limitations: each candidate is scored independently, so the model alone does not enforce a connected route
  (student_test route continuity 0.9813 before any decoding); it is tied statistically with Random Forest
  and LightGBM (CV macro F1 0.9289 ± 0.0179 vs 0.9289 ± 0.0172 and 0.9297 ± 0.0175, the differences are far
  smaller than the fold-to-fold spread), so we cannot claim it is better; it has more hyperparameters than
  the linear model and is harder to interpret.

## B. Training and configuration

**5. Important hyperparameters and initial values** (untuned, fixed seed 42)

| Hyperparameter | Value | Role |
|---|---|---|
| `n_estimators` | 400 | number of boosting rounds (trees) |
| `max_depth` | 6 | depth of each tree; the main complexity knob |
| `learning_rate` | 0.05 | shrinkage per tree; low rate with many rounds is more stable |
| `subsample` | 0.8 | each tree sees a random 80 % of the rows (reduces variance) |
| `colsample_bytree` | 0.8 | each tree sees a random 80 % of the features (decorrelates trees) |
| `tree_method` | hist | histogram-based split search, fast on 1.16 M rows |
| `objective` / `eval_metric` | binary:logistic / logloss | see question 6 |
| other | library defaults | e.g. L2 regularisation `reg_lambda` = 1 |

Training data: 1,162,595 rows from 78 training trips (two anomalous trips and observations more than 50 m
from their reference arc removed), 5-fold GroupKFold by trip.

**6. Loss function / objective**
Binary logistic loss (log loss). Trees are added one at a time; each new tree is fitted to the gradient of the
log loss of the current ensemble, and the model outputs a probability that the candidate is the true arc.
The same log loss is recorded after every round for the loss curve.

## C. Experimental results

**7–8. Performance on the student-testing dataset (10 trips, all observations, unfiltered)**

| Metric | Student-test | Cross-validation (78 trips) |
|---|---|---|
| Observation-level Top-1 accuracy | **0.8901** | 0.8863 |
| Macro F1 | **0.9310** | 0.9285 (fold mean 0.9289 ± 0.0179) |
| Log loss | **0.0843** | 0.0926 |
| Accuracy (candidate level, threshold 0.5) | 0.9659 | 0.9631 |
| Precision / recall (positive class) | 0.8863 / 0.8776 | 0.8956 / 0.8625 |
| Route continuity (classifier only) | 0.9813 | 0.9745 |
| Candidate miss rate | 0.0026 | 0.0028 |

Student-test confusion matrix (rows true, columns predicted; [0, 1]): `[[110,267, 2,151], [2,340, 16,770]]`.
Training time 25.9 s for the full fit (about 21 s per fold); inference on all student-test candidates
0.22 s. Top-1 per student-test trip ranges from 0.7553 (trip 00000077) to 0.9770 (trip 00000052)
(`a_pertrip_xgboost.csv`).

**9. Overfitting or underfitting? Evidence**
Mild overfitting, no underfitting.
- *Loss curves* (CV fold 1, validation = held-out trips; `a_loss_xgboost.png`): training log loss falls
  from 0.4085 to 0.0658 over 400 rounds; validation log loss falls from 0.4194 to a minimum of 0.1154 at
  round 295 and ends at 0.1158. Validation loss does not climb after its minimum, it flattens, so the
  model is not overfitting badly, but the gap between 0.0658 and 0.1158 shows it fits the training trips
  better than unseen ones. (Fold 1 is the hardest fold, its log loss 0.1158 is above the pooled 0.0926.)
- *Train vs CV (refit model scored on its own training data):* macro F1 0.9461 vs 0.9285 (gap 0.0177), Top-1
  0.9142 vs 0.8863, log loss 0.0715 vs 0.0926 (`a_fit_diagnostics.csv`). The gaps are modest.
- *CV vs student-test:* macro F1 0.9285 vs 0.9310, Top-1 0.8863 vs 0.8901, log loss 0.0926 vs 0.0843. The
  held-out trips score as well as cross-validation, so it generalises to new trips.
- *Not underfitting:* the training loss keeps falling to a low value, and it beats the linear model clearly
  (student-test log loss 0.0843 vs 0.1319 for Logistic Regression, Top-1 0.8901 vs 0.8606).
- *Possible improvement:* fewer rounds (about 300, where validation loss is lowest) or a lower depth could reduce the small gap (not tested); we kept the
  initial configuration for the untuned comparison, and the tuning step (section 4) was done on LightGBM.
- *Stability:* Top-1 varies from 0.8628 to 0.9299 across the five folds, i.e. performance depends on which
  trips are held out, as for every model in the comparison.
