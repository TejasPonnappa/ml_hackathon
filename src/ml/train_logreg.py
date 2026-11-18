"""Model 1: Logistic Regression through the shared harness (same data, folds, preprocessing, metrics as every model)."""
from sklearn.linear_model import LogisticRegression

from src.common.config import SEED
from src.ml.harness import run_experiment

CONFIG = dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight=None, random_state=SEED)

if __name__ == "__main__":
    run_experiment(name="logreg", family="Linear (probabilistic)",
                   make_model=lambda: LogisticRegression(**CONFIG),
                   config=CONFIG, needs_scaling=True)
