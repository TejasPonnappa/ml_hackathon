"""Model 5: LightGBM (gradient-boosted trees, leaf-wise growth). Owner: Tejas.

Initial (untuned) configuration - values chosen as sensible defaults for ~1.2M
rows x 19 features; Bayesian optimisation later searches around them.
Run:  python -m src.ml.train_lightgbm
"""
from lightgbm import LGBMClassifier

from src.common.config import SEED
from src.ml.harness import run_experiment

CONFIG = dict(
    n_estimators=400,        # boosting rounds
    learning_rate=0.05,      # shrinkage per tree
    num_leaves=63,           # max leaves per tree (leaf-wise growth -> main complexity knob)
    min_child_samples=50,    # min rows per leaf (regularises noisy GPS labels)
    subsample=0.8, subsample_freq=1,   # row bagging each round
    colsample_bytree=0.8,    # feature bagging
    reg_lambda=1.0,          # L2 on leaf weights
    objective="binary",      # log loss -> calibrated probabilities for the HMM
    random_state=SEED, n_jobs=-1, verbose=-1,
)


def fit_kwargs(X_tr, y_tr, X_va, y_va):
    # evaluate train + held-out-trip validation loss every round (for the real loss curve)
    return dict(eval_X=(X_tr, X_va), eval_y=(y_tr, y_va), eval_names=["train", "val"],
                eval_metric="binary_logloss")


def history(model):
    r = model.evals_result_
    return {"train": r["train"]["binary_logloss"], "val": r["val"]["binary_logloss"],
            "metric": "binary log loss", "x_label": "boosting round"}


if __name__ == "__main__":
    run_experiment(name="lightgbm", family="Boosting (gradient-boosted trees, leaf-wise)",
                   make_model=lambda: LGBMClassifier(**CONFIG), config=CONFIG,
                   needs_scaling=False, fit_kwargs_fn=fit_kwargs, history_fn=history)
