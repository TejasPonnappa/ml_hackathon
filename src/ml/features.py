"""Feature engineering for (GPS observation, candidate arc) pairs (owner: Person A).

Every row of the candidate table gets the features below. None of them use the
reference route, so the same function runs unchanged in the API.

Feature                why it helps
---------------------  ----------------------------------------------------------
dist_m                 GPS noise is mostly small: nearer arcs are more likely.
t_along                where the GPS point projects on the arc (0 = start node,
                       1 = end node); points clamped to an end often belong to
                       the NEXT/PREVIOUS arc instead.
length_m               short arcs (junction stubs) are easy to land near but
                       rarely the true arc for long.
heading_diff,          angle between the vehicle's direction of motion and the
heading_cos            arc's direction. THE key feature: two-way roads are stored
                       as two arcs (u->v and v->u) at identical distance; EDA showed
                       ~82% of reference arcs are at rank 0/1, i.e. a twin tie that
                       only direction can break.
speed_mps, low_speed   heading is unreliable when the vehicle is (almost) stopped;
                       the model can learn to trust heading less then.
rank, dist_gap,        relative position among this observation's candidates:
dist_ratio, n_cands    what matters is being nearer than the alternatives, and
                       dense areas (many candidates) are harder.
dist_prev1/next1,      distance from the neighbouring GPS points (i-2..i+2) to the
dist_prev2/next2,      SAME arc. The true arc stays close over consecutive points;
dist_win_mean          a crossing street is close only at one instant.
dt_prev                time gap to the previous point (gaps make neighbours less
                       informative).
from_deg_in,           connectivity of the arc's endpoints (number of arcs at the
to_deg_out             node) as a proxy for road type / junction density.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.geo import angle_diff_deg, arc_geometry, bearing_deg, point_segment_distance, track_xy

FEATURE_COLS = [
    "dist_m", "t_along", "length_m",
    "heading_diff", "heading_cos", "speed_mps", "low_speed",
    "rank", "dist_gap", "dist_ratio", "n_cands",
    "dist_prev1", "dist_next1", "dist_prev2", "dist_next2", "dist_win_mean",
    "dt_prev", "from_deg_in", "to_deg_out",
]

LOW_SPEED_MPS = 1.0   # below ~3.6 km/h the motion direction is mostly GPS jitter
HEADING_WINDOW = 2    # heading from point i-2 to i+2 (smooths jitter)


def _motion(trip):
    """Per-observation motion heading (deg), speed (m/s) and time gap (s)."""
    x, y = track_xy(trip)
    t = trip.track["t"].to_numpy()
    n = len(x)
    i0 = np.clip(np.arange(n) - HEADING_WINDOW, 0, n - 1)
    i1 = np.clip(np.arange(n) + HEADING_WINDOW, 0, n - 1)
    dx, dy = x[i1] - x[i0], y[i1] - y[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.hypot(dx, dy) / dt
    heading = bearing_deg(dx, dy)
    dt_prev = np.r_[0.0, np.diff(t)]
    return x, y, heading, speed, dt_prev


def add_features(trip, cand: pd.DataFrame) -> pd.DataFrame:
    """Add FEATURE_COLS to a candidate table of ONE trip (output of generate_candidates)."""
    g = arc_geometry(trip)
    x, y, heading, speed, dt_prev = _motion(trip)
    n = len(x)
    out = cand.copy()
    o = out["obs_idx"].to_numpy()
    a = out["arc_id"].to_numpy()
    ax, ay = g["ax"].to_numpy()[a], g["ay"].to_numpy()[a]
    bx, by = g["bx"].to_numpy()[a], g["by"].to_numpy()[a]

    out["length_m"] = g["length_m"].to_numpy()[a]
    hd = angle_diff_deg(heading[o], g["bearing"].to_numpy()[a])
    out["heading_diff"] = hd
    out["heading_cos"] = np.cos(np.radians(hd))
    out["speed_mps"] = speed[o]
    out["low_speed"] = (speed[o] < LOW_SPEED_MPS).astype(np.int8)

    grp = out.groupby("obs_idx")["dist_m"]
    dmin = grp.transform("min").to_numpy()
    out["dist_gap"] = out["dist_m"].to_numpy() - dmin
    out["dist_ratio"] = out["dist_m"].to_numpy() / (dmin + 1.0)
    out["n_cands"] = grp.transform("size").to_numpy()

    # distance of neighbouring GPS points to the same arc
    for name, off in (("dist_prev1", -1), ("dist_next1", 1), ("dist_prev2", -2), ("dist_next2", 2)):
        j = np.clip(o + off, 0, n - 1)
        out[name], _ = point_segment_distance(x[j], y[j], ax, ay, bx, by)
    out["dist_win_mean"] = out[["dist_m", "dist_prev1", "dist_next1",
                                "dist_prev2", "dist_next2"]].mean(axis=1)

    out["dt_prev"] = dt_prev[o]
    deg_in = np.bincount(g["to_node"].to_numpy(), minlength=len(trip.nodes))
    deg_out = np.bincount(g["from_node"].to_numpy(), minlength=len(trip.nodes))
    out["from_deg_in"] = deg_in[g["from_node"].to_numpy()[a]]
    out["to_deg_out"] = deg_out[g["to_node"].to_numpy()[a]]

    out[FEATURE_COLS] = out[FEATURE_COLS].astype(np.float32)
    return out
