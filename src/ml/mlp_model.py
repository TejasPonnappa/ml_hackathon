"""Picklable MLP wrapper for Model 2 (MLP through the shared harness).

The wrapper lives in its own module so joblib saves a reference to
``src.ml.mlp_model.EpochMLP`` instead of ``__main__.EpochMLP``;
the artefact can then be reloaded from a fresh process for inference.
"""
import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier


class EpochMLP:
    def __init__(self, cfg):
        self.cfg = cfg
        self.history_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        c = self.cfg
        rng = np.random.RandomState(c["random_state"])
        self.net = MLPClassifier(hidden_layer_sizes=c["hidden_layer_sizes"], activation=c["activation"], solver="adam",
                                 learning_rate_init=c["learning_rate_init"], alpha=c["alpha"],
                                 batch_size=c["batch_size"], random_state=c["random_state"])
        ti = rng.choice(len(X), min(len(X), c["loss_eval_rows"]), replace=False)
        vi = rng.choice(len(X_val), min(len(X_val), c["loss_eval_rows"]), replace=False) if X_val is not None else None
        hist = {"train": [], "val": [], "metric": "log loss", "x_label": "epoch"}
        for _ in range(c["epochs"]):
            perm = rng.permutation(len(X))
            for s in range(0, len(perm), c["batch_size"]):
                i = perm[s:s + c["batch_size"]]
                self.net.partial_fit(X[i], y[i], classes=[0, 1])
            if X_val is not None:
                hist["train"].append(log_loss(y[ti], self.net.predict_proba(X[ti])[:, 1], labels=[0, 1]))
                hist["val"].append(log_loss(y_val[vi], self.net.predict_proba(X_val[vi])[:, 1], labels=[0, 1]))
        self.history_ = hist
        return self

    def predict_proba(self, X):
        return self.net.predict_proba(X)
