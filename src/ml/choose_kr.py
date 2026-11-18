"""Choose candidate parameters K (max candidates) and R (radius, m) from TRAIN data only.

For every train observation we record
  * ref_dist_m : distance from the GPS point to its reference arc (from labels.py)
  * ref_rank   : rank of the reference arc among ALL arcs sorted by distance
                 (searched with a generous K=64, R=300 m; -1 = not found)
Then the candidate miss rate for a grid of (K, R) is simply
  miss(K, R) = fraction of obs where NOT (ref_rank < K and ref_dist_m <= R).

Outputs: results/a_eda_kr_obs.csv (per-observation), results/a_eda_kr_grid.csv
Run:  python -m src.ml.choose_kr
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.common.candidates import arc_index
from src.common.config import RESULTS_DIR
from src.common.geo import track_xy
from src.common.io import list_trips, load_trip
from src.common.labels import align_route

BIG_K, BIG_R = 64, 300.0


def _one(trip_id: str, partition: str) -> pd.DataFrame:
    trip = load_trip(trip_id, partition)
    lab = align_route(trip)
    px, py = track_xy(trip)
    cand = arc_index(trip).query(px, py, BIG_K, BIG_R)
    m = cand.merge(pd.DataFrame({"obs_pos": np.arange(len(lab)), "ref_arc": lab["ref_arc"]}),
                   on="obs_pos")
    hit = m[m["arc_id"] == m["ref_arc"]].set_index("obs_pos")["rank"]
    out = lab.copy()
    out["ref_rank"] = hit.reindex(np.arange(len(lab))).fillna(-1).astype(int).to_numpy()
    out["trip_id"] = trip_id
    out["partition"] = partition
    return out


def main():
    rows = []
    for part in ("train", "student_test"):
        rows += Parallel(n_jobs=-1)(delayed(_one)(t, part) for t in list_trips(part))
    obs = pd.concat(rows, ignore_index=True)
    obs.to_csv(RESULTS_DIR / "a_eda_kr_obs.csv", index=False)

    grid = []
    for part, d in obs.groupby("partition"):
        found = d["ref_rank"] >= 0
        for K in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20):
            for R in (20, 30, 40, 50, 75, 100, 150):
                ok = found & (d["ref_rank"] < K) & (d["ref_dist_m"] <= R)
                grid.append(dict(partition=part, K=K, R=R, miss_rate=1 - ok.mean(), n_obs=len(d)))
    grid = pd.DataFrame(grid)
    grid.to_csv(RESULTS_DIR / "a_eda_kr_grid.csv", index=False)

    tr = obs[obs.partition == "train"]
    print("train obs:", len(tr), "| ref_dist_m percentiles:")
    print(tr["ref_dist_m"].quantile([.5, .9, .95, .99, .995, .999]).round(2).to_string())
    print("ref_rank distribution (train):")
    print(tr["ref_rank"].value_counts(normalize=True).sort_index().head(15).round(4).to_string())
    print(grid[grid.partition == "train"].pivot(index="K", columns="R", values="miss_rate").round(4))


if __name__ == "__main__":
    main()
