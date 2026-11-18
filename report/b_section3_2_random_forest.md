# Section 3.2 — Random Forest (Yashas; numbers checked and corrected by Tejas)

Sources: `results/a_cv_random_forest.json`, `results/a_model_comparison.csv`, `results/a_fit_diagnostics.csv`,
`results/a_cm_random_forest.png`, `results/logs/a_random_forest.log`. Code: `src/ml/train_random_forest.py`
(shared harness, same data, folds and metrics as every other model).

**A1 — Why it is appropriate.** Random Forest can model non-linear dependencies between the features of
PathSnap's candidate roads. Furthermore, it can work with the diverse feature set without feature scaling being
required.

**A2 — Strengths and limitations.** The ensemble is strong and stable and can capture the interaction effects of
GPS and road information through the splitting of trees. The major limitation of this model is its speed, since it
is the slowest of all models in both training and inference: it takes 162 seconds to train and 1.67 seconds for
student-test prediction. The model is also bulky: the saved model is 174 MB, against 1.9 MB for the deployed
LightGBM.

**B5 — Hyperparameters and initial values.** Initial parameters were set to number of trees: 200, max depth: 20,
min size of leaf: 5. The square root of the number of features was considered at each split and a bootstrap
sample of 0.3 was taken per tree. Bootstrap was enabled, the split criterion was Gini impurity. Random seed: 42.

**B6 — Training objective.** The splitting criterion for trees is Gini impurity. The Random Forest model does not
provide any global training loss record or epoch history; hence, no loss curve is shown.

**C7/C8 — Student-test results.** On the student-test dataset the model obtained observation-level Top-1 accuracy
of 0.8803 (candidate-level accuracy 0.9649), macro F1 of 0.9285 and log loss of 0.0866.

**C9 — Overfitting or underfitting.** The in-sample macro F1 on the training trips is 0.9470 against a
cross-validation macro F1 of 0.9285 (gap 0.018, `a_fit_diagnostics.csv`), so the forest fits the training trips
somewhat more closely than unseen trips: mild overfitting. The cross-validation macro F1 (fold mean 0.9289) is close
to the student-test macro F1 of 0.9285, which shows that this does not harm generalisation to new trips.
