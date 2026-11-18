"""Model: XGBoost (gradient-boosted trees, level-wise growth). Owner: Tejasvi.

Initial (untuned) configuration. Everything else (data, folds, metrics, saving)
comes from the shared harness so the comparison with other models is fair.
Run:  python -m src.ml.train_xgboost
"""
from xgboost import XGBClassifier

from src.common.config import SEED
from src.ml.harness import run_experiment

CONFIG = dict(
    n_estimators=400,        # boosting rounds
    max_depth=6,             # depth of each tree (level-wise growth -> main complexity knob)
    learning_rate=0.05,      # shrinkage per tree
    subsample=0.8,           # row bagging each round
    colsample_bytree=0.8,    # feature bagging per tree
    objective="binary:logistic",  # log loss -> probabilities for the HMM
    eval_metric="logloss",   # metric recorded every round for the loss curve
    tree_method="hist",      # histogram split finding (fast on ~1.2M rows)
    random_state=SEED, n_jobs=-1,
)


def fit_kwargs(X_tr, y_tr, X_va, y_va):
    # evaluate train + held-out-trip validation loss every round (for the real loss curve)
    return dict(eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)


def history(model):
    r = model.evals_result()
    return {"train": r["validation_0"]["logloss"], "val": r["validation_1"]["logloss"],
            "metric": "log loss", "x_label": "boosting round"}


if __name__ == "__main__":
    run_experiment(name="xgboost", family="Boosting (gradient-boosted trees, level-wise)",
                   make_model=lambda: XGBClassifier(**CONFIG), config=CONFIG,
                   needs_scaling=False, fit_kwargs_fn=fit_kwargs, history_fn=history)
