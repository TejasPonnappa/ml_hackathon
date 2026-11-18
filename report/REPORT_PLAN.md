# PathSnap technical report — assembly plan

One consolidated PDF, sections in the order of the Deliverables PDF (§2–§7). Every number must come from a
file in `results/` (evaluators verify against the repo). Every figure must be an actual generated file listed below.
Owner of the final assembly: Tejas. Number check against `results/`: Yashas.

## Status per section
| PDF section | Source text | Owner | Status |
|---|---|---|---|
| Title page: team, repo URL https://github.com/TejasPonnappa/ml_hackathon | — | Tejas | to write |
| 2 Dataset understanding & EDA | `report/a_section2_notes.md` (+ feature table in `a_section3_notes.md` §3.0 / `src/ml/features.py`) | Tejas | draft ready |
| 3.1 Overall comparison + selection | `report/a_section3_notes.md` §3.1 | Tejas | draft ready |
| 3.2 Logistic Regression | `report/b_section3_models_and_section6_folium.md` | Yuvraj | draft ready |
| 3.2 MLP | `report/b_section3_models_and_section6_folium.md` | Yuvraj | draft ready |
| 3.2 Random Forest | (to be sent) | Yashas | **missing** |
| 3.2 XGBoost | (to be sent) | Tejasvi | **missing** |
| 3.2 LightGBM | `report/a_section3_notes.md` §3.2 | Tejas | draft ready |
| 3.3 Model visualisations | figures below | Tejas | figures ready |
| 4 Bayesian optimisation | `report/a_section4_notes.md` | Tejas | draft ready |
| 5 HMM / Viterbi | (to be sent: states, emission, transition, gaps, disconnections, classifier-only vs HMM, cases where it did not help) | Tejasvi | **missing** |
| 6 Folium | `report/b_section3_models_and_section6_folium.md` §6 + regenerated map with HMM layer | Yuvraj | text ready, **map needs HMM layer** |
| 7 API, UI, deployment | screenshots: UI with a route + accuracy; API logs in CMD on Laptop 1 | Tejasvi + Tejas | **missing** |
| AI-use disclosure | text below | Tejas | ready |

Per-model §3.2 must answer, with our own numbers (no textbook answers): A1 suitability, A2 strengths/limits,
B5 hyperparameters + initial values, B6 loss/objective, C7 student-test performance, C8 student-test Top-1 /
macro F1 / log loss, C9 over/underfitting evidence (`results/a_fit_diagnostics.csv`, loss curve if any).

## Figures (all in `results/`)
| Section | Figure | File |
|---|---|---|
| 2 | Trip locations (worldwide) | `a_eda_trip_locations.png` |
| 2 | Trajectory size distributions | `a_eda_trip_sizes.png` |
| 2 | Sampling interval / time gaps | `a_eda_time_gaps.png` |
| 2 | Missing / invalid value analysis | `a_eda_missing_invalid.png` |
| 2 | Road-network size and connectivity | `a_eda_network.png` |
| 2 | GPS trajectories + network + reference route | `a_eda_trajectories.png` |
| 2 | GPS-to-reference distance (GPS noise) | `a_eda_gps_noise.png` |
| 2 | Candidate miss rate vs K, R | `a_eda_candidate_miss.png` |
| 2 | Vehicle speed | `a_eda_speed.png` |
| 2 | Folium EDA map (screenshot) | `a_eda_folium_train_00000001.html` |
| 3.3 | CV macro F1 comparison | `a_cmp_cv_macro_f1.png` |
| 3.3 | Student-test Top-1 and macro F1 | `a_cmp_student_test.png` |
| 3.3 | Log loss comparison | `a_cmp_log_loss.png` |
| 3.3 | Inference time | `a_cmp_inference_time.png` |
| 3.3 | Train vs CV vs test (over/underfitting) | `a_fit_diagnostics.png` |
| 3.3 | Fold stability | `a_cv_fold_stability.png` |
| 3.3 | Loss curves (epoch/round-based models only) | `a_loss_lightgbm.png`, `a_loss_xgboost.png`, `a_loss_mlp.png` |
| 3.3 | Confusion matrices (optional) | `a_cm_<model>.png` |
| 4 | Optimisation history | `a_bayesopt_history.png` |
| 4 | Untuned vs tuned table | `a_bayesopt_comparison.csv` |
| 5 | Classifier-only vs HMM | `b_hmm_comparison.csv` (+ plots from Tejasvi) |
| 6 | Report map with HMM layer (screenshot) | `b_folium_<trip>.html` |
| 7 | UI screenshot, API CMD log screenshot | to capture at the two-laptop test |

Random Forest and Logistic Regression have no epoch-based history, so no loss curves are shown for them
(the Deliverables PDF forbids artificial curves).

## Tables (copy numbers from)
- 3.1: `results/a_model_comparison.csv` (CV macro F1 = fold mean ± std; ST = student_test)
- 3.2 C9: `results/a_fit_diagnostics.csv`
- 4: `results/a_bayesopt_comparison.csv`, `a_bayesopt_best.json`, `a_bayesopt_trials.csv`, `a_bench_inference.csv`
- 5: `results/b_hmm_comparison.csv`, `b_hmm_tuning*.csv`, `b_hmm_best.json`

## AI-use disclosure (to include in the PDF)
AI coding assistants (Claude Code) were used during this project to help write and review code, generate
plotting/analysis scripts and draft parts of the report text. All experiments were executed by the team on
our own machines; every reported number and figure is produced by the code in this repository and stored in
`results/`. Design decisions (data preprocessing, labelling, candidate generation, features, model selection,
optimisation and decoding) were reviewed and can be explained by the team members.
