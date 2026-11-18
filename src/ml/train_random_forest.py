"""Model 3: Random Forest (bagging ensemble of decision trees). Owner: Yashas.

Same data, folds, preprocessing and metrics as every other model (shared harness).
Initial (untuned) configuration:
  * n_estimators=200   - enough trees for stable probability averages
  * max_depth=20, min_samples_leaf=5 - limit tree size (1.16M rows would otherwise
    grow very deep, memory-heavy trees that memorise noisy GPS labels)
  * max_features="sqrt" - standard RF decorrelation of trees (~4 of 19 features per split)
  * max_samples=0.3    - each tree sees a 30 % bootstrap sample: faster, more diverse trees
  * criterion="gini"   - split criterion; RF has no global loss/epochs, so no loss curve
Run:  python -m src.ml.train_random_forest
"""
from sklearn.ensemble import RandomForestClassifier

from src.common.config import SEED
from src.ml.harness import run_experiment

CONFIG = dict(n_estimators=200, max_depth=20, min_samples_leaf=5, max_features="sqrt",
              max_samples=0.3, bootstrap=True, criterion="gini", n_jobs=-1, random_state=SEED)

if __name__ == "__main__":
    run_experiment(name="random_forest", family="Ensemble (bagging of decision trees)",
                   make_model=lambda: RandomForestClassifier(**CONFIG), config=CONFIG,
                   needs_scaling=False)
