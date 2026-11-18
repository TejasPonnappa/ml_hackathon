# PathSnap – Model 1: Logistic Regression (results)

All numbers below come from `python src/run_model1_logreg.py` (outputs in `results/`, seed 42, sklearn 1.x, lbfgs is deterministic).
Data used: **80 training trips** (fit + 5-fold group CV) and **10 student-testing trips** (development comparison). The evaluator-testing partition is not present in or used by this work.

## 1. How the problem is framed
1. Each trip has its own road network (`.nodes`, directed `.arcs`) in a different city, so every trip is projected to local metres (equirectangular, centred on the network; error is negligible at city scale).
2. **Candidate generation:** for each GPS point take up to **10 nearest arcs within 100 m** (point-to-segment distance, KD-tree over arc samples every 20 m).
3. **Observation-level reference arc:** `.route` is only an ordered arc list, so each GPS point is assigned to a route arc by a monotone DP alignment (minimise point-to-arc distance, capped at 150 m; 5 m penalty per skipped arc). Median distance of a point to its assigned arc is about 4–6 m. *This is a derived reference label, not independently verified ground truth.*
4. **Classifier:** binary Logistic Regression on (observation, candidate-arc) pairs; label 1 = candidate is the reference arc. Predicted arc = highest-probability candidate (Top-1).
5. **Features (16 + 4 squares):** distance to arc, distance rank, distance minus nearest-candidate distance, ratio, position along arc, arc length, heading cosine & |cosine| (GPS heading vs arc direction) + validity flag, speed, time gap, candidate count, distance of the previous/next GPS point to the arc, out-degree of the arc's end node, in-degree of its start node. Squares of four distance features were added because a linear model otherwise cannot represent a Gaussian (quadratic-in-distance) emission. No label or route information is used as a feature.

## 2. Configuration (Section 3.2-B)
| Item | Value |
|---|---|
| Preprocessing | square-feature expansion → StandardScaler (fit on training folds only) |
| Model | `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight=None, random_state=42)` |
| Objective | regularised cross-entropy (log loss + L2); decision threshold 0.5 |
| Training rows | 1,615,143 candidate rows from 80 trips (208,417 GPS points); converged in 78 lbfgs iterations |
| Training time | 38.2 s (final fit); ~14–20 s per CV fold |
| Validation | `GroupKFold(5)` grouped by trip (no trip split across train/val) |

## 3. Results

### 3.1 Row for the consolidated comparison table
| Model | Family | Key config | CV Macro F1 | Student-test Top-1 | Route continuity | Student-test Macro F1 | Log loss | Inference time |
|---|---|---|---|---|---|---|---|---|
| Model 1 Logistic Regression | Linear / probabilistic | L2, C=1, lbfgs, std. + squared-distance features | **0.8682 ± 0.0444** | **0.8338** | **0.9223** | **0.8966** | **0.1246** | 0.0093 ms/obs (0.178 s for 19,160 obs) |

CV Top-1 = 0.7626 ± 0.1202, CV route continuity = 0.9212 ± 0.0128, CV log loss = 0.1501 ± 0.0369.

### 3.2 Metrics on the 10 student-testing trips (19,160 GPS observations, 157,969 candidate rows)
| Metric | Value |
|---|---|
| Accuracy (candidate level) | 0.9565 |
| Precision / Recall / F1 (correct-arc class) | 0.8276 / 0.8086 / 0.8180 |
| **Macro F1** | **0.8966** |
| Log loss | 0.1246 |
| Confusion matrix [[TN, FP],[FN, TP]] | [[135,633, 3,220],[3,658, 15,458]] |
| **Observation-level Top-1 accuracy** | **0.8338** (0.8357 when the reference arc is in the candidate set) |
| Candidate miss rate | 0.0023 (0.23 %) |
| Route continuity (consecutive predicted arcs identical or connected) | 0.9223 |
| Sequence level (de-duplicated predicted route vs `.route` arc set, mean over trips) | precision 0.799, recall 0.875, F1 0.831 |
| Inference time | 1.13 µs per candidate, 0.0093 ms per observation |

Per trip Top-1: 0.934, 0.904, 0.799, 0.904, 0.939, 0.863, 0.810, 0.817, **0.679 (trip 77)**, 0.820 (`results/model1_student_test_per_trip.csv`).

### 3.3 Cross-validation folds (80 training trips)
| Fold | Macro F1 | Top-1 | Route continuity | Log loss | Candidate miss rate |
|---|---|---|---|---|---|
| 1 | 0.8122 | 0.5926 | 0.9100 | 0.1958 | 0.3152 |
| 2 | 0.9254 | 0.8855 | 0.9303 | 0.1057 | 0.0110 |
| 3 | 0.8698 | 0.7989 | 0.9282 | 0.1573 | 0.0036 |
| 4 | 0.8397 | 0.6891 | 0.9326 | 0.1715 | 0.0931 |
| 5 | 0.8936 | 0.8467 | 0.9049 | 0.1204 | 0.0043 |

Fold variance is driven almost entirely by **candidate misses** (folds 1 and 4), i.e. trips with large GPS gaps / GPS far from the road (for example training trip 2 has gaps up to 160 s), not by the classifier itself.

## 4. Answers for report Section 3.2 (Model 1)
**A. Understanding**
- *Why suitable?* It is a fast, interpretable baseline that outputs calibrated probabilities (needed as HMM emission scores) and its coefficients can be checked against map-matching intuition.
- *Strengths:* probabilistic outputs, 0.009 ms/obs inference, stable convergence (78 iterations), readable coefficients. The largest effects match intuition: `dist_minus_min` (−5.13), distance rank (−1.93), distance of neighbouring GPS points (−1.4 to −1.6) lower the score; heading agreement `cos_heading` (+1.76) raises it. *Limitations:* linear decision boundary – interactions (e.g. distance × heading at junctions, parallel roads) are not captured; each observation is scored independently, so connectivity is ignored (route continuity 0.92 means ~8 % of consecutive predictions are not connected).

**B. Training/configuration:** see Section 2. Initial and final hyperparameters are identical (C = 1.0, L2); no tuning has been done yet.

**C. Results**
- Student-test: Top-1 **0.8338**, Macro F1 **0.8966**, log loss **0.1246**.
- *Over/underfitting:* **no overfitting** – train (Macro F1 0.886, Top-1 0.739) is not better than CV (0.868, 0.763) or student-test (0.897, 0.834). The train/CV Top-1 is depressed by trips with candidate misses. The model is **mildly underfit / high-bias**: Top-1 is only 0.836 even when the correct arc is among the candidates, and the weakest trip (77, Top-1 0.679, sequence precision 0.641) shows scattered wrong-arc picks on parallel roads that a sequential decoder should fix.

## 5. Caveats (be upfront with evaluators)
- The reference arc per GPS point is derived by alignment; label noise exists near junctions.
- Candidate coverage on the **training** set is 89.1 % at 100 m / K = 10 (92.4 % at 150 m), pulled down by a few gap-heavy trips; on student-test it is 99.8 %. The radius/K setting was fixed from `results/candidate_coverage_train.csv` (training data only) and could be widened.
- Macro F1 here is the mean of F1 for the "correct arc" and "wrong arc" classes at threshold 0.5; the class balance is ~12 % positive. Top-1 accuracy is the primary selection metric because it is what the route reconstruction uses.
- Not done yet (separate steps in the brief): the other four models, Bayesian optimisation, HMM/Viterbi, API/UI.
- Per the assessment policy, disclose AI assistance in this work and make sure every team member can explain the code and these numbers.

## 6. Files
- Code: `src/pathsnap_data.py` (loading, projection, candidates, alignment, features), `src/run_model1_logreg.py` (EDA tables, CV, training, evaluation, plots, Folium).
- Results: `results/model1_results.json`, `model1_cv_folds.csv`, `model1_student_test_per_trip.csv`, `model1_comparison_row.csv`, `model1_coefficients.csv`, `dataset_inventory.csv`, `candidate_coverage_train.csv`, trained model `model1_logreg.joblib`.
- Figures: `fig_cv_macro_f1.png`, `fig_confusion_matrix_student_test.png`, `fig_student_test_per_trip.png`, `fig_train_cv_test_gap.png`, `fig_coefficients.png`, `fig_candidate_coverage.png`.
- Folium maps: `results/map_model1_trip_00000007.html`, `results/map_model1_trip_00000052.html` (layers: GPS, road network, reference route, predicted route, mis-matched points).
