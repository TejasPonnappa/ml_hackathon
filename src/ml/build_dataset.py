"""Build the candidate + label + feature tables for train and student_test.

Outputs (git-ignored, regenerable in ~1-2 min):
  data_cache/candidates_<partition>.parquet  one row per (obs, candidate arc):
      trip_id, obs_idx, arc_id, dist_m, label, ref_arc, ref_dist_m, <FEATURE_COLS>
  data_cache/obs_<partition>.parquet          one row per GPS observation:
      trip_id, obs_idx, ref_arc, ref_dist_m, n_cands, ref_in_cands
      (needed so Top-1 accuracy counts observations that have NO candidate)

Run:  python -m src.ml.build_dataset
"""
from __future__ import annotations

import json
import time

import pandas as pd
from joblib import Parallel, delayed

from src.common.candidates import DEFAULT_K, DEFAULT_R, generate_candidates
from src.common.config import ANOMALOUS_TRIPS, CACHE_DIR, RESULTS_DIR
from src.common.io import list_trips, load_trip
from src.common.labels import align_route
from src.ml.features import add_features


def build_trip(trip_id: str, partition: str):
    trip = load_trip(trip_id, partition)
    cand = add_features(trip, generate_candidates(trip, DEFAULT_K, DEFAULT_R))
    lab = align_route(trip)
    cand = cand.merge(lab[["obs_idx", "ref_arc", "ref_dist_m"]], on="obs_idx", how="left")
    cand["label"] = (cand["arc_id"] == cand["ref_arc"]).astype("int8")

    obs = lab[["obs_idx", "ref_arc", "ref_dist_m"]].copy()
    obs.insert(0, "trip_id", trip_id)
    nc = cand.groupby("obs_idx").size()
    hit = cand.groupby("obs_idx")["label"].max()
    obs["n_cands"] = obs["obs_idx"].map(nc).fillna(0).astype(int)
    obs["ref_in_cands"] = obs["obs_idx"].map(hit).fillna(0).astype(int)
    return cand, obs


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    summary = {"K": DEFAULT_K, "R_m": DEFAULT_R, "anomalous_trips": sorted(ANOMALOUS_TRIPS)}
    for part in ("train", "student_test"):
        t0 = time.time()
        res = Parallel(n_jobs=-1)(delayed(build_trip)(t, part) for t in list_trips(part))
        cand = pd.concat([r[0] for r in res], ignore_index=True)
        obs = pd.concat([r[1] for r in res], ignore_index=True)
        cand.to_parquet(CACHE_DIR / f"candidates_{part}.parquet", index=False)
        obs.to_parquet(CACHE_DIR / f"obs_{part}.parquet", index=False)
        clean = ~obs["trip_id"].isin(ANOMALOUS_TRIPS)
        summary[part] = {
            "trips": int(obs["trip_id"].nunique()), "obs": int(len(obs)), "candidate_rows": int(len(cand)),
            "positive_rate": float(cand["label"].mean()),
            "candidate_miss_rate_all": float(1 - obs["ref_in_cands"].mean()),
            "candidate_miss_rate_excl_anomalous": float(1 - obs.loc[clean, "ref_in_cands"].mean()),
            "obs_without_candidates": int((obs["n_cands"] == 0).sum()),
            "build_seconds": round(time.time() - t0, 1),
        }
        print(part, summary[part])
    (RESULTS_DIR / "a_dataset_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
