# Model 1 - Logistic Regression (PathSnap)

Self-contained Model 1 experiment: candidate generation, observation-level reference alignment, 20 features,
L2 logistic regression, 5-fold trip-grouped CV, student-test evaluation, plots and Folium maps.
Full write-up with numbers: `MODEL1_LOGREG_REPORT.md`. Saved weights: `results/model1_logreg.joblib`.

Note: this uses its own candidate/feature pipeline (`src/pathsnap_data.py`), separate from the shared `src/ml/` harness,
so its numbers are not directly comparable with models trained through `src/ml/harness.py` until it is re-run on that data.

## Reproduce
Place the course data at `<repo root>/student_folder/Student folder/{train,student_test}/...` (git-ignored), then:

    cd model1_logreg
    python src/run_model1_logreg.py

Needs numpy, pandas, scipy, scikit-learn, joblib, matplotlib, folium. Seed 42. Writes to `model1_logreg/results/`
(the 250 MB candidate cache `cache_candidates.joblib` is regenerated on first run and not committed).
Only the 80 train and 10 student_test trips are used; the evaluator partition is never touched.
