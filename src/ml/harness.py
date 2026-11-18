"""Shared training / evaluation harness. EVERY model goes through run_experiment()
so all models see identical data, folds, preprocessing and metrics.

Protocol
  1. Load data_cache/candidates_train.parquet (build with: python -m src.ml.build_dataset).
  2. Training data filter (train partition only):
       - drop ANOMALOUS_TRIPS (route does not match the track, see EDA)
       - drop observations whose reference arc is > R m from the GPS point
         (label unreliable / outside the candidate radius)
  3. 5-fold GroupKFold by trip_id: no trip is ever split across train/validation.
     If needs_scaling, a StandardScaler is fitted on the training fold only.
     Out-of-fold (OOF) predictions -> CV metrics; per-fold macro F1 -> mean/std.
  4. Refit on all (filtered) train trips, then evaluate on ALL observations of the
     10 student_test trips (no filtering, so the numbers are honest).
  5. Save results/a_cv_<name>.json, results/a_loss_<name>.png (only if the
     model has a real per-epoch/round history), results/a_cm_<name>.png,
     per-trip student-test Top-1 to results/a_pertrip_<name>.csv, the fitted model to
     data_cache/models/<name>.joblib and predictions to data_cache/preds_<name>_*.parquet.

Usage (in src/ml/train_<name>.py):
    run_experiment(name="logreg", family="Linear", make_model=lambda: LogisticRegression(**CFG),
                   config=CFG, needs_scaling=True)
Optional hooks for models with a loss history (XGBoost, LightGBM, MLP):
    fit_kwargs_fn(X_tr, y_tr, X_va, y_va) -> dict of extra kwargs for model.fit
    history_fn(model) -> {"train": [...], "val": [...], "metric": "logloss", "x_label": "round"}
    (history is taken from fold 1 of the CV, where "val" is a held-out group of trips)
Optional: max_train_rows to subsample training rows (by whole observations) for slow models.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.common.candidates import DEFAULT_K, DEFAULT_R
from src.common.config import ANOMALOUS_TRIPS, CACHE_DIR, RESULTS_DIR, SEED
from src.common.metrics import evaluate, top1_table
from src.ml.features import FEATURE_COLS

N_FOLDS = 5


def load_partition(partition: str, train_filter: bool):
    cand = pd.read_parquet(CACHE_DIR / f"candidates_{partition}.parquet")
    obs = pd.read_parquet(CACHE_DIR / f"obs_{partition}.parquet")
    if train_filter:
        obs = obs[~obs["trip_id"].isin(ANOMALOUS_TRIPS) & (obs["ref_dist_m"] <= DEFAULT_R)]
        cand = cand.merge(obs[["trip_id", "obs_idx"]], on=["trip_id", "obs_idx"])
    return cand.reset_index(drop=True), obs.reset_index(drop=True)


def _subsample(cand, obs, max_rows, seed):
    """Keep whole observations (all their candidates) until ~max_rows rows."""
    if max_rows is None or len(cand) <= max_rows:
        return cand, obs
    keys = obs[["trip_id", "obs_idx"]].sample(frac=min(1.0, max_rows / len(cand)), random_state=seed)
    return (cand.merge(keys, on=["trip_id", "obs_idx"]).reset_index(drop=True),
            obs.merge(keys, on=["trip_id", "obs_idx"]).reset_index(drop=True))


def _plot_history(name, hist):
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="validation (held-out trips)")
    plt.xlabel(hist.get("x_label", "epoch")); plt.ylabel(hist.get("metric", "log loss"))
    plt.title(f"{name}: training history (CV fold 1)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"a_loss_{name}.png", dpi=120); plt.close()


def _plot_cm(name, cm):
    cm = np.asarray(cm)
    plt.figure(figsize=(4, 3.5))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center")
    plt.xticks([0, 1], ["pred 0", "pred 1"]); plt.yticks([0, 1], ["true 0", "true 1"])
    plt.title(f"{name}: student_test confusion matrix"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"a_cm_{name}.png", dpi=120); plt.close()


def run_experiment(name: str, family: str, make_model, config: dict,
                   needs_scaling: bool = False, fit_kwargs_fn=None, history_fn=None,
                   max_train_rows: int | None = None, seed: int = SEED) -> dict:
    t_start = time.time()
    log = lambda *a: print(f"[{datetime.now():%H:%M:%S}] [{name}]", *a, flush=True)
    (CACHE_DIR / "models").mkdir(parents=True, exist_ok=True)

    cand, obs = load_partition("train", train_filter=True)
    cand, obs = _subsample(cand, obs, max_train_rows, seed)
    X = cand[FEATURE_COLS].to_numpy(np.float32)
    y = cand["label"].to_numpy()
    groups = cand["trip_id"].to_numpy()
    log(f"train rows={len(cand):,} obs={len(obs):,} trips={cand.trip_id.nunique()} pos_rate={y.mean():.3f}")

    # ---------------- cross-validation
    oof = np.zeros(len(cand))
    fold_rows, history = [], None
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups)):
        X_tr, X_va = X[tr], X[va]
        if needs_scaling:
            sc = StandardScaler().fit(X_tr)          # fitted on the training fold only
            X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)
        model = make_model()
        kw = fit_kwargs_fn(X_tr, y[tr], X_va, y[va]) if fit_kwargs_fn else {}
        t0 = time.time()
        model.fit(X_tr, y[tr], **kw)
        fit_s = time.time() - t0
        oof[va] = model.predict_proba(X_va)[:, 1]
        if k == 0 and history_fn is not None:
            history = history_fn(model)
        va_cand = cand.iloc[va].assign(prob=oof[va])
        va_obs = obs[obs["trip_id"].isin(set(groups[va]))]
        fm = evaluate(va_cand, va_obs, "train", with_continuity=False)
        fold_rows.append({"fold": k + 1, "fit_seconds": fit_s, **{m: fm[m] for m in
                          ("macro_f1", "log_loss", "top1_accuracy", "accuracy")}})
        log(f"fold {k+1}: macro_f1={fm['macro_f1']:.4f} top1={fm['top1_accuracy']:.4f} "
            f"logloss={fm['log_loss']:.4f} fit={fit_s:.1f}s")

    cv = evaluate(cand.assign(prob=oof), obs, "train", with_continuity=True)
    folds = pd.DataFrame(fold_rows)
    cv["macro_f1_fold_mean"] = float(folds["macro_f1"].mean())
    cv["macro_f1_fold_std"] = float(folds["macro_f1"].std())

    # ---------------- final fit on all train trips
    scaler = StandardScaler().fit(X) if needs_scaling else None
    Xf = scaler.transform(X) if scaler is not None else X
    model = make_model()
    t0 = time.time()
    model.fit(Xf, y)
    train_seconds = time.time() - t0

    # ---------------- student_test (all observations, unfiltered)
    st_cand, st_obs = load_partition("student_test", train_filter=False)
    Xs = st_cand[FEATURE_COLS].to_numpy(np.float32)
    t0 = time.time()
    Xs_ = scaler.transform(Xs) if scaler is not None else Xs
    st_prob = model.predict_proba(Xs_)[:, 1]
    infer_seconds = time.time() - t0
    st_cand = st_cand.assign(prob=st_prob)
    st = evaluate(st_cand, st_obs, "student_test", with_continuity=True)
    per_trip = top1_table(st_cand, st_obs).groupby("trip_id")["correct"].mean().rename("top1_accuracy")
    per_trip.to_csv(RESULTS_DIR / f"a_pertrip_{name}.csv")

    # ---------------- save everything
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS,
                 "K": DEFAULT_K, "R": DEFAULT_R, "name": name}, CACHE_DIR / "models" / f"{name}.joblib")
    # native weight files as well (git-ignored, local only)
    if hasattr(model, "booster_"):                       # LightGBM
        model.booster_.save_model(str(CACHE_DIR / "models" / f"{name}.txt"))
    elif hasattr(model, "get_booster"):                  # XGBoost
        model.get_booster().save_model(str(CACHE_DIR / "models" / f"{name}.json"))
    cand[["trip_id", "obs_idx", "arc_id", "label"]].assign(prob=oof).to_parquet(
        CACHE_DIR / f"preds_{name}_oof.parquet", index=False)
    st_cand[["trip_id", "obs_idx", "arc_id", "dist_m", "label", "prob"]].to_parquet(
        CACHE_DIR / f"preds_{name}_student_test.parquet", index=False)
    if history is not None:
        _plot_history(name, history)
    _plot_cm(name, st["confusion_matrix"])

    result = {
        "name": name, "family": family, "config": config, "seed": seed,
        "needs_scaling": needs_scaling, "features": FEATURE_COLS,
        "candidate_K": DEFAULT_K, "candidate_R_m": DEFAULT_R, "n_folds": N_FOLDS,
        "max_train_rows": max_train_rows, "train_rows": int(len(cand)),
        "cv": cv, "cv_folds": fold_rows, "student_test": st,
        "train_seconds_full_fit": train_seconds,
        "inference_seconds_student_test": infer_seconds,
        "inference_ms_per_1k_rows": 1000 * infer_seconds / len(st_cand) * 1000,
        "history": history, "total_seconds": time.time() - t_start,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }
    (RESULTS_DIR / f"a_cv_{name}.json").write_text(json.dumps(result, indent=2, default=float))
    log(f"CV macro_f1={cv['macro_f1']:.4f}±{cv['macro_f1_fold_std']:.4f} top1={cv['top1_accuracy']:.4f} "
        f"| student_test macro_f1={st['macro_f1']:.4f} top1={st['top1_accuracy']:.4f} "
        f"logloss={st['log_loss']:.4f} continuity={st['route_continuity']:.4f} "
        f"| fit={train_seconds:.1f}s infer={infer_seconds:.2f}s")
    return result
