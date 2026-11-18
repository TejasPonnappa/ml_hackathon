"""Model 2: MLP through the shared harness.
sklearn MLPClassifier driven by a manual partial_fit loop that records REAL per-epoch train and validation log loss
(validation = the held-out group of trips of the CV fold; it is only logged, never used for early stopping or selection,
so training length is fixed)."""
from src.common.config import SEED
from src.ml.harness import run_experiment
from src.ml.mlp_model import EpochMLP

CONFIG = dict(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", learning_rate_init=3e-3, alpha=1e-4,
              batch_size=16384, epochs=15, loss_eval_rows=50_000, max_train_rows=400_000, random_state=SEED)


if __name__ == "__main__":
    run_experiment(name="mlp", family="Neural network (MLP)",
                   make_model=lambda: EpochMLP(CONFIG), config=CONFIG, needs_scaling=True,
                   fit_kwargs_fn=lambda Xt, yt, Xv, yv: dict(X_val=Xv, y_val=yv),
                   history_fn=lambda m: m.history_, max_train_rows=CONFIG["max_train_rows"])
