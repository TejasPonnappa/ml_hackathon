"""Bayesian hyperparameter optimisation of the selected model (LightGBM). Owner: Tejas.

Why LightGBM: highest CV Top-1 and lowest CV log loss of the 5 compared models
(results/a_model_comparison.csv), and the fastest of the tree models to train, so
many optimisation trials fit in the time budget.

Method
  * Optuna TPE sampler (Tree-structured Parzen Estimator = Bayesian optimisation:
    it models which regions of the search space gave good scores and samples there),
    seed 42 for reproducibility.
  * Objective (minimised): mean validation LOG LOSS over the same 5-fold
    GroupKFold-by-trip split as the harness, on the 80 train trips only
    (78 after removing the 2 anomalous trips). Log loss is chosen because the
    HMM decoder consumes the probabilities, so probability quality matters,
    and it is smoother than macro F1 / Top-1, which helps the optimiser.
    Macro F1 and Top-1 are recorded for every trial too.
  * MedianPruner: after each fold the running mean is compared with earlier
    trials; clearly worse trials are stopped early to save compute.
  * Student_test and evaluator data are NOT used for the search.
After the search the best configuration is refit with the shared harness
(name "lightgbm_tuned") so tuned vs untuned use identical folds and metrics.

Outputs
  results/a_bayesopt_trials.csv      every trial: params, CV log loss, macro F1, Top-1, state, seconds
  results/a_bayesopt_best.json       best params, number of trials, total compute time
  results/a_bayesopt_history.png     objective per trial + best-so-far
  results/a_bayesopt_comparison.csv  untuned vs tuned (CV macro F1, ST Top-1, ST macro F1, log loss, inference)
  results/a_cv_lightgbm_tuned.json   full harness result of the tuned model
Run:  python -m src.ml.bayesopt_lightgbm --trials 20
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from src.common.config import RESULTS_DIR as R, SEED
from src.common.metrics import evaluate
from src.ml.features import FEATURE_COLS
from src.ml.harness import N_FOLDS, load_partition, run_experiment
from src.ml.train_lightgbm import CONFIG as UNTUNED, fit_kwargs, history

FIXED = dict(objective="binary", subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)


def search_space(trial: optuna.Trial) -> dict:
    """Ranges centred on the untuned config, wide enough to under- and over-fit."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),   # step size per tree
        n_estimators=trial.suggest_int("n_estimators", 150, 800, step=50),          # number of trees
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),              # tree complexity
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),  # leaf size / regularisation
        subsample=trial.suggest_float("subsample", 0.5, 1.0),                       # row bagging
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),         # feature bagging
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),         # L2 on leaf weights
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),           # L1 on leaf weights
    )


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}] [bayesopt]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--no-refit", action="store_true", help="skip the final harness refit")
    args = ap.parse_args()

    cand, obs = load_partition("train", train_filter=True)
    X = cand[FEATURE_COLS].to_numpy(np.float32)
    y = cand["label"].to_numpy()
    groups = cand["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    log(f"data: {len(cand):,} rows, {cand.trip_id.nunique()} train trips, {N_FOLDS} folds by trip; "
        f"{args.trials} trials, objective = mean CV log loss (minimise)")

    rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {**search_space(trial), **FIXED}
        t0 = time.time()
        losses, f1s, top1s = [], [], []
        for k, (tr, va) in enumerate(folds):
            m = LGBMClassifier(**params).fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            met = evaluate(cand.iloc[va].assign(prob=p), obs[obs.trip_id.isin(set(groups[va]))],
                           "train", with_continuity=False)
            losses.append(met["log_loss"]); f1s.append(met["macro_f1"]); top1s.append(met["top1_accuracy"])
            trial.report(float(np.mean(losses)), step=k)
            if trial.should_prune():
                rows.append({"trial": trial.number, "state": "PRUNED", "folds_run": k + 1,
                             "cv_log_loss": np.mean(losses), "cv_macro_f1": np.mean(f1s),
                             "cv_top1": np.mean(top1s), "seconds": time.time() - t0, **trial.params})
                log(f"trial {trial.number:>2}: PRUNED after fold {k+1} (running logloss={np.mean(losses):.4f})")
                pd.DataFrame(rows).to_csv(R / "a_bayesopt_trials.csv", index=False)
                raise optuna.TrialPruned()
        val = float(np.mean(losses))
        rows.append({"trial": trial.number, "state": "COMPLETE", "folds_run": N_FOLDS, "cv_log_loss": val,
                     "cv_macro_f1": np.mean(f1s), "cv_top1": np.mean(top1s), "seconds": time.time() - t0,
                     **trial.params})
        pd.DataFrame(rows).to_csv(R / "a_bayesopt_trials.csv", index=False)
        log(f"trial {trial.number:>2}: logloss={val:.4f} macro_f1={np.mean(f1s):.4f} top1={np.mean(top1s):.4f} "
            f"({time.time() - t0:.0f}s) {trial.params}")
        return val

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    # trial 0 = the untuned configuration, so the search starts from (and must beat) it
    study.enqueue_trial({k: UNTUNED[k] for k in ("learning_rate", "n_estimators", "num_leaves",
                                                  "min_child_samples", "subsample", "colsample_bytree",
                                                  "reg_lambda")} | {"reg_alpha": 1e-3})
    t_start = time.time()
    study.optimize(objective, n_trials=args.trials)
    total = time.time() - t_start

    trials = pd.DataFrame(rows)
    best = {**study.best_params, **FIXED}
    n_done = int((trials.state == "COMPLETE").sum())
    summary = {"best_params": best, "best_cv_log_loss": study.best_value, "best_trial": study.best_trial.number,
               "n_trials": len(trials), "n_complete": n_done, "n_pruned": int((trials.state == "PRUNED").sum()),
               "total_search_seconds": total, "objective": "mean 5-fold GroupKFold CV log loss (train only)",
               "sampler": "Optuna TPESampler(seed=42)", "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
               "timestamp": datetime.now().isoformat(timespec="seconds")}
    (R / "a_bayesopt_best.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done: {len(trials)} trials ({n_done} complete, {summary['n_pruned']} pruned) in {total/60:.1f} min; "
        f"best trial {study.best_trial.number} logloss={study.best_value:.4f}")
    log(f"best params: {study.best_params}")

    c = trials[trials.state == "COMPLETE"].sort_values("trial")
    plt.figure(figsize=(7, 4))
    plt.scatter(c.trial, c.cv_log_loss, label="trial (complete)")
    plt.plot(c.trial, c.cv_log_loss.cummin(), color="tab:red", label="best so far")
    plt.axhline(c.loc[c.trial == 0, "cv_log_loss"].iloc[0], color="gray", ls="--", label="untuned (trial 0)")
    plt.xlabel("trial"); plt.ylabel("mean CV log loss"); plt.legend()
    plt.title("Bayesian optimisation (Optuna TPE) of LightGBM - train, GroupKFold by trip")
    plt.tight_layout(); plt.savefig(R / "a_bayesopt_history.png", dpi=120); plt.close()

    if args.no_refit:
        return
    log("refitting best configuration with the shared harness (CV + student_test) ...")
    tuned = run_experiment(name="lightgbm_tuned", family="Boosting (LightGBM, Bayesian-optimised)",
                           make_model=lambda: LGBMClassifier(**best), config=best, needs_scaling=False,
                           fit_kwargs_fn=fit_kwargs, history_fn=history)
    untuned = json.loads((R / "a_cv_lightgbm.json").read_text())
    comp = pd.DataFrame([{
        "model": name, "cv_macro_f1": r["cv"]["macro_f1_fold_mean"], "cv_macro_f1_std": r["cv"]["macro_f1_fold_std"],
        "cv_top1": r["cv"]["top1_accuracy"], "cv_log_loss": r["cv"]["log_loss"],
        "st_top1": r["student_test"]["top1_accuracy"], "st_macro_f1": r["student_test"]["macro_f1"],
        "st_log_loss": r["student_test"]["log_loss"], "st_route_continuity": r["student_test"]["route_continuity"],
        "train_seconds_full_fit": r["train_seconds_full_fit"],
        "inference_seconds_student_test": r["inference_seconds_student_test"],
    } for name, r in (("lightgbm_untuned", untuned), ("lightgbm_tuned", tuned))])
    comp.to_csv(R / "a_bayesopt_comparison.csv", index=False)
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
