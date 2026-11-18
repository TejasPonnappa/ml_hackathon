"""Candidate arc generation (owner: Person A).

For each GPS observation we keep the K nearest arcs that lie within R metres.
These candidates are the "hidden states" the classifier scores and the HMM
decodes over.

SPATIAL INDEX
  Each trip's network has ~165k arcs, so brute-force distance (1300 points x
  165k arcs) is too slow. We sample points along every arc every SPACING m and
  put them in a KD-tree. Any arc within R of a GPS point has a sample point
  within R + SPACING/2, so a ball query with that radius finds every arc that
  could qualify; exact point-to-segment distances are then computed only for
  those arcs.

K and R are chosen from EDA on the train partition (see report/a_section2_notes.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import arc_geometry, point_segment_distance, track_xy

# Chosen from results/a_eda_kr_grid.csv (train only, anomalous trips excluded):
# K=10, R=50 m -> miss rate 1.66 % train / 0.26 % student_test; going to K=16,
# R=100 m gains <1 % but grows the table ~60 %.
DEFAULT_K = 10
DEFAULT_R = 50.0      # metres
SPACING = 10.0        # metres between sample points on an arc


class ArcIndex:
    """KD-tree over points sampled along every arc of one trip's network."""

    def __init__(self, geom: pd.DataFrame, spacing: float = SPACING):
        self.geom = geom
        self.spacing = spacing
        n_pts = np.maximum(1, np.ceil(geom["length_m"].to_numpy() / spacing)).astype(int) + 1
        arc_of_pt = np.repeat(geom["arc_id"].to_numpy(), n_pts)
        # fraction along the arc for each sample point: 0, 1/(n-1), ..., 1
        starts = np.repeat(np.cumsum(n_pts) - n_pts, n_pts)
        k = np.arange(n_pts.sum()) - starts
        frac = k / np.repeat(n_pts - 1, n_pts)
        ax = np.repeat(geom["ax"].to_numpy(), n_pts); ay = np.repeat(geom["ay"].to_numpy(), n_pts)
        bx = np.repeat(geom["bx"].to_numpy(), n_pts); by = np.repeat(geom["by"].to_numpy(), n_pts)
        self.arc_of_pt = arc_of_pt
        self.tree = cKDTree(np.column_stack([ax + frac * (bx - ax), ay + frac * (by - ay)]))

    def query(self, px: np.ndarray, py: np.ndarray, k: int, r: float) -> pd.DataFrame:
        """K nearest arcs within r metres of each point.

        Returns long table: obs_pos (0..len(px)-1), arc_id, dist_m, t_along, rank
        (rank 0 = nearest arc).
        """
        hits = self.tree.query_ball_point(np.column_stack([px, py]), r + self.spacing / 2)
        lens = np.fromiter((len(h) for h in hits), dtype=int, count=len(hits))
        if lens.sum() == 0:
            return pd.DataFrame(columns=["obs_pos", "arc_id", "dist_m", "t_along", "rank"])
        obs_pos = np.repeat(np.arange(len(hits)), lens)
        arc = self.arc_of_pt[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if h])]
        # unique (obs, arc) pairs
        pairs = np.unique(np.column_stack([obs_pos, arc]), axis=0)
        obs_pos, arc = pairs[:, 0], pairs[:, 1]
        g = self.geom
        d, t = point_segment_distance(px[obs_pos], py[obs_pos],
                                      g["ax"].to_numpy()[arc], g["ay"].to_numpy()[arc],
                                      g["bx"].to_numpy()[arc], g["by"].to_numpy()[arc])
        df = pd.DataFrame({"obs_pos": obs_pos, "arc_id": arc, "dist_m": d, "t_along": t})
        df = df[df["dist_m"] <= r]
        df = df.sort_values(["obs_pos", "dist_m", "arc_id"], kind="mergesort")
        df["rank"] = df.groupby("obs_pos").cumcount()
        return df[df["rank"] < k].reset_index(drop=True)


def arc_index(trip) -> ArcIndex:
    """KD-tree for a trip's network (cached on the Trip object)."""
    idx = getattr(trip, "_arc_index", None)
    if idx is None:
        idx = ArcIndex(arc_geometry(trip))
        trip._arc_index = idx
    return idx


def generate_candidates(trip, k: int = DEFAULT_K, r: float = DEFAULT_R) -> pd.DataFrame:
    """Candidate table for one trip: trip_id, obs_idx, arc_id, dist_m, t_along, rank.

    Uses only the track and network (never the route), so it is safe to call
    in the prediction path.
    """
    index = arc_index(trip)
    px, py = track_xy(trip)
    cand = index.query(px, py, k, r)
    cand.insert(0, "obs_idx", trip.track["obs_idx"].to_numpy()[cand.pop("obs_pos").to_numpy()])
    cand.insert(0, "trip_id", trip.trip_id)
    return cand
