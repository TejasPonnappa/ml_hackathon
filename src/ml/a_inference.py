"""Inference entry point used by the API (owner: Person A).

The API calls ONE function per trip:

    from src.ml.a_inference import score_trip, to_hmm_input
    trip   = load_trip(trip_id, DATA_DIR, with_route=False)   # never load the route here
    scored = score_trip(trip)            # DataFrame: obs_idx, arc_id, dist_m, t_along, rank, prob
    hmm_in = to_hmm_input(scored)        # {obs_idx: [(arc_id, prob), ...]}

score_trip runs exactly the training-time code path (candidates.py -> features.py ->
saved preprocessing -> saved model), so training and serving can never drift apart.

Which model file is loaded (first that exists):
  1. env PATHSNAP_MODEL (explicit path to a .joblib bundle)
  2. artifacts/model.joblib            (final exported model, after Bayesian optimisation)
  3. data_cache/models/lightgbm.joblib (local, produced by src.ml.train_lightgbm)
  4. data_cache/models/xgboost.joblib  (local, produced by src.ml.train_xgboost)
A bundle is the dict saved by the harness: model, scaler, feature_cols, K, R, name.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.candidates import generate_candidates
from src.common.config import ARTIFACTS_DIR, CACHE_DIR
from src.ml.features import add_features

CANDIDATE_PATHS = [
    ARTIFACTS_DIR / "model.joblib",
    CACHE_DIR / "models" / "lightgbm.joblib",
    CACHE_DIR / "models" / "xgboost.joblib",
]


def model_path() -> Path:
    env = os.environ.get("PATHSNAP_MODEL")
    if env:
        return Path(env)
    for p in CANDIDATE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("No model found. Train one (python -m src.ml.train_lightgbm) "
                            "or set PATHSNAP_MODEL. Looked in: " + ", ".join(map(str, CANDIDATE_PATHS)))


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Load the model bundle once per process (the API keeps it in memory)."""
    path = model_path()
    b = joblib.load(path)
    b["path"] = str(path)
    return b


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """P(candidate is the true arc) for rows that already have the feature columns."""
    b = load_bundle()
    X = df[b["feature_cols"]].to_numpy(np.float32)
    if b.get("scaler") is not None:
        X = b["scaler"].transform(X)
    return b["model"].predict_proba(X)[:, 1]


def score_trip(trip) -> pd.DataFrame:
    """Candidates + features + probabilities for one trip (uses track + network only)."""
    b = load_bundle()
    cand = generate_candidates(trip, k=b["K"], r=b["R"])
    if cand.empty:
        return cand.assign(prob=pd.Series(dtype=float))
    feats = add_features(trip, cand)
    feats["prob"] = predict_proba(feats)
    return feats[["trip_id", "obs_idx", "arc_id", "dist_m", "t_along", "rank", "prob"]]


def to_hmm_input(scored: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    """{obs_idx: [(arc_id, prob), ...]} sorted by prob, the generic HMM input format.
    Observations with no candidate within R are simply absent (the HMM must handle gaps)."""
    s = scored.sort_values(["obs_idx", "prob"], ascending=[True, False])
    return {int(o): list(zip(g["arc_id"].astype(int), g["prob"].astype(float)))
            for o, g in s.groupby("obs_idx", sort=True)}


def model_info() -> dict:
    """Small description for the API /health endpoint and logs."""
    b = load_bundle()
    return {"model": b.get("name"), "path": b["path"], "K": b["K"], "R_m": b["R"],
            "n_features": len(b["feature_cols"])}
