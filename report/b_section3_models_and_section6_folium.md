# Report drafts: §3.2 (Logistic Regression, MLP) and §6 (Folium)

> DRAFT written with AI assistance. Before submitting: check every number against the files named below, rewrite in
> your own words, and disclose AI assistance as the event policy requires. Every team member must be able to explain it.

All numbers come from `results/a_cv_logreg.json`, `results/a_cv_mlp.json`, `results/a_pertrip_*.csv`,
`results/a_cm_*.png`, `results/a_loss_mlp.png`. All models run through the same harness (`src/ml/harness.py`):
80 train trips (2 anomalous trips dropped for fitting), 5-fold GroupKFold by trip with the scaler fitted inside each
fold, then the 10 student-testing trips (all observations, unfiltered). Candidates: K = 10 nearest arcs within 50 m
(reference arc missing from the candidate set for 0.26 % of student-test observations). Seed 42.

## 3.1 Rows for the comparison table
| Model | Family | Key configuration | CV Macro F1 (fold mean ± std) | CV Top-1 | Route continuity (CV / student) | Student-test Macro F1 | Student-test Top-1 | Log loss (student) | Inference (student, 131,528 rows) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear / probabilistic | L2, C=1.0, lbfgs, scaled | 0.9045 ± 0.0164 | 0.8714 | 0.9652 / 0.9775 | 0.9043 | 0.8606 | 0.1319 | 0.011 s |
| MLP | Neural network | 19-32-16-1, ReLU, Adam lr 3e-3, 15 epochs, batch 16384, 400k-row subsample, scaled | 0.9173 ± 0.0138 | 0.8714 | 0.9611 / 0.9793 | 0.9165 | 0.8673 | 0.1046 | 0.047 s |

## 3.2 Model 1: Logistic Regression
**A. Model understanding**
1. *Why suitable?* Each (GPS point, candidate arc) pair is a binary decision with a handful of numeric features, and
   the HMM later needs a probability per candidate (an emission score). Logistic regression gives calibrated
   probabilities directly, trains in seconds and its weights can be read, so it is the natural baseline against which
   the other models must justify their extra cost.
2. *Strengths / limitations.* Strengths seen in the experiments: fit in 7.9 s on 1,162,595 rows, inference 0.011 s for
   the whole student-test set, stable folds (Macro F1 std 0.016), and interpretable weights: after standardisation the
   largest are `heading_diff` (-4.00), `dist_gap` (-2.83) and `rank` (-2.33), i.e. arcs that point against the
   direction of travel or are farther than the nearest alternative are penalised. (`heading_cos` also gets a negative
   weight, -1.89; it is strongly correlated with `heading_diff`, so individual signs are not meaningful on their own.)
   Limitation: the decision boundary is linear in the features, so it cannot express interactions such as "distance
   matters less when the heading is unreliable (low speed)". Each observation is also scored independently, so
   connectivity is ignored until the HMM.

**B. Training and configuration**
5. Hyperparameters (initial = final, no tuning): `penalty=l2`, `C=1.0`, `solver=lbfgs`, `max_iter=2000`,
   `class_weight=None`, `random_state=42`; features standardised (scaler fitted on the training fold only). lbfgs
   converged in 45 iterations.
6. Objective: regularised cross-entropy (log loss + L2 penalty); decision threshold 0.5 for the class metrics. Top-1
   uses the arg-max candidate per observation.

**C. Experimental results**
7. Student-testing (19,160 observations, 131,528 candidate rows): accuracy 0.9526, precision 0.8393, recall 0.8333,
   F1 (correct-arc class) 0.8363, **Macro F1 0.9043**, confusion matrix [[109,370, 3,048], [3,186, 15,924]]
   (`results/a_cm_logreg.png`). Weakest trip 00000077 (Top-1 0.707), best 00000052 (0.947).
8. Observation-level Top-1 = **0.8606** (0.8629 when the reference arc is among the candidates), Macro F1 = **0.9043**,
   log loss = **0.1319**, route continuity 0.9775.
9. Over/underfitting: no sign of overfitting. CV Macro F1 (0.9045 ± 0.0164) and student-test Macro F1 (0.9043) are the
   same, and CV log loss (0.1434) is close to the student-test value (0.1319). The model is somewhat underfit (high
   bias): the MLP, using the same data and folds, reaches Macro F1 0.9165 and log loss 0.1046, so a nonlinear model
   extracts more from the same 19 features. Note: the harness does not compute training-set metrics, so the evidence
   is the CV-versus-test comparison and the comparison with the MLP.

## 3.2 Model 2: MLP
**A. Model understanding**
1. *Why suitable?* The same inputs, but hidden layers can learn interactions (distance × heading × speed) that the
   linear model cannot; the output is still a probability usable as an HMM emission score.
2. *Strengths / limitations.* Strength: better probability quality than the linear model (student-test log loss
   0.1046 vs 0.1319; precision 0.8734 vs 0.8393). Limitations: needs scaled inputs, is slower to train (8.8 s on a
   400k-row subsample vs 7.9 s on 1.16 M rows for logistic regression) and 4× slower at inference (0.047 s vs 0.011 s
   for the student-test set), and its weights are not interpretable. Its Top-1 gain over logistic regression is small
   (0.8673 vs 0.8606).

**B. Training and configuration**
5. Architecture 19 → 32 → 16 → 1 (≈1,000 parameters; sklearn `MLPClassifier` with a manual `partial_fit` loop), ReLU,
   Adam, `learning_rate_init=3e-3`, `alpha=1e-4` (L2), `batch_size=16384`, 15 epochs (fixed, no early stopping),
   trained on a 400,661-row subsample (whole observations, `max_train_rows=400000`) to keep training fast, scaled
   inputs, seed 42. Earlier, larger settings (64-32 network, full data, 20-30 epochs) took 11-16 minutes per fold
   and were abandoned before producing results; these are not reported as results.
6. Objective: binary cross-entropy with L2 penalty, minimised by Adam. The per-epoch train and validation log loss
   plotted in `results/a_loss_mlp.png` are real values from CV fold 1 (validation = the held-out trips of that fold,
   train loss measured on a 50k-row sample).

**C. Experimental results**
7. Student-testing: accuracy 0.9592, precision 0.8734, recall 0.8409, F1 (correct-arc class) 0.8568, **Macro F1 0.9165**,
   confusion matrix [[110,089, 2,329], [3,041, 16,069]] (`results/a_cm_mlp.png`). Weakest trip 00000077 (Top-1 0.723),
   best 00000000 (0.944).
8. Observation-level Top-1 = **0.8673** (0.8695 when the reference arc is among the candidates), Macro F1 = **0.9165**,
   log loss = **0.1046**, route continuity 0.9793. CV: Macro F1 0.9173 ± 0.0138, Top-1 0.8714.
9. Over/underfitting: no overfitting. Over the 15 epochs train loss fell 0.300 → 0.106 and validation loss
   0.314 → 0.134; the validation curve decreases monotonically and the final gap is 0.028, and student-test log loss
   (0.1046) is no worse than the fold validation loss. The curve was still falling slowly at epoch 15, so the model is
   slightly undertrained rather than overfit; more epochs or the full training set may help (not tested).

## 6. Geographical visualisation (Folium)
- Script: `src/viz/folium_map.py` (offline; reads the data, unlike the UI). Command:
  `python -m src.viz.folium_map 00000007 --partition student_test --api http://localhost:8000`
  → `results/b_folium_00000007.html`.
- Layers (layer control, top right; legend bottom left): raw GPS (blue circles, start/end markers), road network
  within ~200 m of the track (grey), reference route (green; rebuilt with `load_trip` + `align_route`), classifier-only
  top-1 route (orange), HMM/Viterbi route (red). Bounds are fitted to the track.
- Predictions come from the API (`/predict`, both classifier-only and HMM routes). Without `--api` the script falls
  back to a local classifier-only route from a saved model and omits the HMM layer.
- **Status:** the map file currently in `results/` was generated with the local fallback (classifier-only, logistic
  regression model, trip 00000007, student-test partition: 121 reference arcs, 2,483 network arcs near the track). The
  HMM layer appears only after the API and HMM are available; regenerate with `--api` then and insert a screenshot of
  the final map in the PDF.
