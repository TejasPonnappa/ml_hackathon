"""Model comparison table + plots from every results/a_cv_<name>.json (harness output).

Primary selection metric: CV observation-level Top-1 accuracy (mean over the 5
GroupKFold folds' out-of-fold predictions on the 80 train trips).
  Why: the task is to pick the right arc for each GPS observation, which is
  exactly Top-1; CV (not student_test) is used for selection so the 10
  student_test trips stay an independent check. Tie-break (within one fold
  std): lower CV log loss, because the HMM consumes the probabilities.

Outputs
  results/a_model_comparison.csv          all metrics, one row per model
  results/a_cmp_cv_macro_f1.png           CV macro F1 (+/- fold std)
  results/a_cmp_student_test.png          student_test Top-1 and macro F1
  results/a_cmp_log_loss.png              CV and student_test log loss
  results/a_cmp_inference_time.png        inference time per 1k candidate rows
Run:  python -m src.ml.compare_models
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import RESULTS_DIR as R

ORDER = ["logreg", "mlp", "random_forest", "xgboost", "lightgbm"]   # display order if present


def _short_cfg(cfg: dict) -> str:
    skip = {"random_state", "n_jobs", "verbose", "verbosity", "eval_metric", "objective", "tree_method"}
    return ", ".join(f"{k}={v}" for k, v in cfg.items() if k not in skip)


def load_results() -> pd.DataFrame:
    rows = []
    for f in sorted(R.glob("a_cv_*.json")):
        if f.stem.startswith("a_cv__"):          # temporary / check runs
            continue
        r = json.loads(f.read_text())
        cv, st = r["cv"], r["student_test"]
        rows.append({
            "model": r["name"], "family": r["family"], "key_config": _short_cfg(r["config"]),
            "cv_macro_f1": cv["macro_f1"], "cv_macro_f1_fold_mean": cv["macro_f1_fold_mean"],
            "cv_macro_f1_fold_std": cv["macro_f1_fold_std"], "cv_top1": cv["top1_accuracy"],
            "cv_log_loss": cv["log_loss"], "cv_route_continuity": cv["route_continuity"],
            "cv_accuracy": cv["accuracy"], "cv_precision": cv["precision"], "cv_recall": cv["recall"],
            "st_macro_f1": st["macro_f1"], "st_top1": st["top1_accuracy"],
            "st_top1_covered": st["top1_accuracy_covered"], "st_log_loss": st["log_loss"],
            "st_route_continuity": st["route_continuity"], "st_accuracy": st["accuracy"],
            "st_precision": st["precision"], "st_recall": st["recall"],
            "st_candidate_miss_rate": st["candidate_miss_rate"],
            "st_confusion_matrix": json.dumps(st["confusion_matrix"]),
            "train_seconds_full_fit": r["train_seconds_full_fit"],
            "inference_seconds_student_test": r["inference_seconds_student_test"],
            "inference_ms_per_1k_rows": r["inference_ms_per_1k_rows"],
            "train_rows": r["train_rows"], "max_train_rows": r.get("max_train_rows"),
            "has_loss_history": r.get("history") is not None, "seed": r["seed"],
            "timestamp": r["timestamp"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_o"] = df["model"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return df.sort_values(["_o", "model"]).drop(columns="_o").reset_index(drop=True)


def _bar(ax, names, vals, err=None, color="tab:blue", fmt="{:.3f}"):
    b = ax.bar(names, vals, yerr=err, capsize=4, color=color)
    for rect, v in zip(b, vals):
        ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def plots(df: pd.DataFrame):
    n = df["model"].tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    _bar(ax, n, df.cv_macro_f1_fold_mean, df.cv_macro_f1_fold_std)
    ax.set_ylim(max(0, df.cv_macro_f1_fold_mean.min() - 0.1), 1); ax.set_ylabel("macro F1")
    ax.set_title("CV macro F1 (GroupKFold by trip, 5 folds; mean ± std) - train")
    plt.tight_layout(); plt.savefig(R / "a_cmp_cv_macro_f1.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(n)); w = 0.38
    b1 = ax.bar(x - w / 2, df.st_top1, w, label="Top-1 accuracy (observation level)")
    b2 = ax.bar(x + w / 2, df.st_macro_f1, w, label="macro F1 (candidate level)")
    for bars, vals in ((b1, df.st_top1), (b2, df.st_macro_f1)):
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, n); ax.set_ylim(max(0, min(df.st_top1.min(), df.st_macro_f1.min()) - 0.1), 1)
    ax.set_title("Student-test (10 trips): Top-1 accuracy and macro F1"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(R / "a_cmp_student_test.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w / 2, df.cv_log_loss, w, label="CV (train, out-of-fold)")
    b2 = ax.bar(x + w / 2, df.st_log_loss, w, label="student_test")
    for bars, vals in ((b1, df.cv_log_loss), (b2, df.st_log_loss)):
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, n); ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Log loss: probability quality"); ax.legend()
    plt.tight_layout(); plt.savefig(R / "a_cmp_log_loss.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    _bar(ax, n, df.inference_ms_per_1k_rows, color="tab:gray", fmt="{:.2f}")
    ax.set_ylabel("ms per 1,000 candidate rows"); ax.set_yscale("log")
    ax.set_title("Inference time (student_test, single process)")
    plt.tight_layout(); plt.savefig(R / "a_cmp_inference_time.png", dpi=120); plt.close()


def main():
    df = load_results()
    if df.empty:
        print("no results/a_cv_*.json found"); return
    df.to_csv(R / "a_model_comparison.csv", index=False)
    plots(df)

    show = df[["model", "family", "cv_macro_f1_fold_mean", "cv_macro_f1_fold_std", "cv_top1", "cv_log_loss",
               "st_top1", "st_macro_f1", "st_log_loss", "st_route_continuity", "inference_ms_per_1k_rows"]]
    print(show.round(4).to_string(index=False))

    # selection by the stated primary metric (CV Top-1), tie-break CV log loss
    best = df.sort_values("cv_top1", ascending=False).iloc[0]
    tol = df.loc[df.model == best.model, "cv_macro_f1_fold_std"].iloc[0]
    close = df[df.cv_top1 >= best.cv_top1 - tol / 2]
    pick = close.sort_values("cv_log_loss").iloc[0]
    print(f"\nHighest CV Top-1: {best.model} ({best.cv_top1:.4f}); "
          f"within tolerance: {close.model.tolist()} -> selected by lower CV log loss: {pick.model}")


if __name__ == "__main__":
    main()
