"""Over/under-fitting and fold-stability diagnostics for every model. Owner: Tejas.

Report section 3.2 question 9 asks for evidence of over- or under-fitting. The standard evidence is
the gap between
  TRAIN (in-sample: the model scored on the same rows it was fitted on),
  CV    (out-of-fold, unseen trips; from results/a_cv_<name>.json), and
  TEST  (the 10 student_test trips; from results/a_cv_<name>.json).
Large train >> CV  -> overfitting.  Low train AND low CV -> underfitting (high bias).

Where the in-sample model comes from (same final model as the harness):
  * the saved harness bundle (data_cache/models/<name>.joblib or artifacts/<name>.joblib), or, if
    the bundle is not on this laptop,
  * a deterministic refit of the harness final model with the exact CONFIG of src/ml/train_<name>.py
    on the same training rows (only for LogReg and XGBoost).
The training rows are rebuilt exactly as in the harness (same filter, and the same seeded subsample
for models trained with max_train_rows, i.e. the MLP).

Outputs: results/a_fit_diagnostics.csv, results/a_fit_diagnostics.png, results/a_cv_fold_stability.png
Run:  python -m src.ml.fit_diagnostics
"""
from __future__ import annotations

import importlib
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR as R
from src.common.metrics import evaluate
from src.ml.features import FEATURE_COLS
from src.ml.harness import _subsample, load_partition

MODELS = ["logreg", "mlp", "random_forest", "xgboost", "lightgbm", "lightgbm_tuned"]


def _refit(name: str, res: dict, X, y):
    """Deterministic refit of the harness final model from the training script's CONFIG."""
    from sklearn.preprocessing import StandardScaler
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**importlib.import_module("src.ml.train_logreg").CONFIG)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**importlib.import_module("src.ml.train_xgboost").CONFIG)
    else:
        raise FileNotFoundError(f"no saved bundle for {name}; run python -m src.ml.train_{name}")
    scaler = StandardScaler().fit(X) if res["needs_scaling"] else None
    model.fit(scaler.transform(X) if scaler is not None else X, y)
    return {"model": model, "scaler": scaler}


def _bundle(name: str):
    # committed bundle first (artifacts/, matches the committed result json), then the local harness cache
    for p in (ARTIFACTS_DIR / f"{name}.joblib", CACHE_DIR / "models" / f"{name}.joblib"):
        if p.exists():
            return joblib.load(p), str(p.relative_to(p.parents[1]))
    return None, None


def main():
    cand_all, obs_all = load_partition("train", train_filter=True)
    rows, folds = [], []
    for name in MODELS:
        f = R / f"a_cv_{name}.json"
        if not f.exists():
            print(f"skip {name}: {f.name} missing"); continue
        res = json.loads(f.read_text())
        cand, obs = _subsample(cand_all, obs_all, res.get("max_train_rows"), res["seed"])
        X = cand[FEATURE_COLS].to_numpy(np.float32)
        y = cand["label"].to_numpy()
        b, src = _bundle(name)
        if b is None:
            b, src = _refit(name, res, X, y), "refit from train script CONFIG"
        Xs = b["scaler"].transform(X) if b.get("scaler") is not None else X
        tr = evaluate(cand.assign(prob=b["model"].predict_proba(Xs)[:, 1]), obs, "train", with_continuity=False)
        cv, st = res["cv"], res["student_test"]
        rows.append({
            "model": name, "train_rows": len(cand), "train_scores_from": src,
            "train_macro_f1": tr["macro_f1"], "cv_macro_f1": cv["macro_f1"], "st_macro_f1": st["macro_f1"],
            "train_top1": tr["top1_accuracy"], "cv_top1": cv["top1_accuracy"], "st_top1": st["top1_accuracy"],
            "train_log_loss": tr["log_loss"], "cv_log_loss": cv["log_loss"], "st_log_loss": st["log_loss"],
            "gap_train_minus_cv_macro_f1": tr["macro_f1"] - cv["macro_f1"],
            "gap_cv_minus_test_macro_f1": cv["macro_f1"] - st["macro_f1"],
        })
        for fr in res["cv_folds"]:
            folds.append({"model": name, **fr})
        print(f"{name:>15}: train f1={tr['macro_f1']:.4f}  cv={cv['macro_f1']:.4f}  test={st['macro_f1']:.4f} | "
              f"train top1={tr['top1_accuracy']:.4f} cv={cv['top1_accuracy']:.4f} test={st['top1_accuracy']:.4f} "
              f"| train logloss={tr['log_loss']:.4f}  ({src})")

    df = pd.DataFrame(rows)
    df.to_csv(R / "a_fit_diagnostics.csv", index=False)

    # train / CV / test bars for three metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    x = np.arange(len(df)); w = 0.27
    for ax, (metric, label) in zip(axes, [("macro_f1", "macro F1"), ("top1", "Top-1 accuracy"),
                                          ("log_loss", "log loss (lower = better)")]):
        for i, (split, col) in enumerate([("train", "tab:gray"), ("cv", "tab:blue"), ("st", "tab:orange")]):
            ax.bar(x + (i - 1) * w, df[f"{split}_{metric}"], w, color=col,
                   label={"train": "train (in-sample)", "cv": "CV (unseen trips)", "st": "student_test"}[split])
        ax.set_xticks(x, df.model, rotation=20); ax.set_title(label)
        if metric != "log_loss":
            lo = min(df[f"{s}_{metric}"].min() for s in ("train", "cv", "st"))
            ax.set_ylim(max(0, lo - 0.05), 1)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Over/under-fitting check: train vs cross-validation vs student_test")
    plt.tight_layout(); plt.savefig(R / "a_fit_diagnostics.png", dpi=120); plt.close()

    # per-fold stability
    fd = pd.DataFrame(folds)
    # Models trained on ALL rows share identical GroupKFold folds (same held-out trips). A model trained
    # on a row subsample (max_train_rows, i.e. the MLP) gets a different trip-to-fold assignment, so its
    # fold k is not comparable fold-by-fold and is not connected by the grey lines.
    full = [n for n, r in zip(df.model, rows) if r["train_rows"] == len(cand_all)]
    labels = [n if n in full else f"{n}\n(own folds: subsample)" for n in df.model]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, metric, label in ((axes[0], "macro_f1", "macro F1"), (axes[1], "top1_accuracy", "Top-1 accuracy")):
        for i, name in enumerate(df.model):
            v = fd.loc[fd.model == name].sort_values("fold")
            ax.plot([i] * len(v), v[metric], "o", alpha=.7)
            ax.hlines(v[metric].mean(), i - 0.25, i + 0.25, color="k")
        pos = [i for i, n in enumerate(df.model) if n in full]
        for k in range(1, fd.fold.max() + 1):     # same fold = same held-out trips
            ax.plot(pos, [fd[(fd.model == df.model[i]) & (fd.fold == k)][metric].iloc[0] for i in pos],
                    color="gray", lw=.6, alpha=.6)
        ax.set_xticks(range(len(df)), labels, rotation=20, fontsize=8)
        ax.set_title(f"{label} per CV fold (grey line = same held-out trips)")
    fig.suptitle("Stability across the 5 GroupKFold folds (black bar = mean)")
    plt.tight_layout(); plt.savefig(R / "a_cv_fold_stability.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
