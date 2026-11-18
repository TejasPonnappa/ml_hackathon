"""Per-observation reference arcs from the .route file (owner: Person A).

PROBLEM
  .route lists the arcs the vehicle drove, in order, but NOT which GPS point
  belongs to which arc (route has far fewer rows than the track: e.g. 525 arcs
  for 1315 GPS points in train trip 00000001).

METHOD: monotone alignment by dynamic programming
  Let c[i, j] = metric distance from GPS point i to route arc j.
  We choose a route position j(i) for every observation such that
      j(0) <= j(1) <= ... <= j(N-1)          (the car never drives backwards
                                             along its own route)
  and the total distance  sum_i c[i, j(i)]  is minimal.

  Recurrence (D = best total cost of points 0..i with point i on arc j):
      D[0, j] = c[0, j]
      D[i, j] = c[i, j] + min_{k <= j} D[i-1, k]
  The inner min is a running (prefix) minimum, so the whole DP is O(N * M).

  Why not simply "nearest route arc"? Routes can pass the same street twice
  or run next to themselves (U-turns, loops); nearest-arc would let a point
  jump to the wrong pass. The monotone constraint removes that ambiguity.

USED BY
  * training labels (label = 1 for the candidate whose arc_id == ref_arc)
  * the API's server-side accuracy (Person B). This is the ONLY module that
    reads the reference route; prediction code never imports it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import arc_geometry, point_segment_distance, track_xy
from .io import Trip


def align_route(trip: Trip) -> pd.DataFrame:
    """Return one reference arc per GPS observation.

    Columns: obs_idx, ref_arc, ref_seq (position in .route), ref_dist_m
    ref_dist_m = distance from the GPS point to its reference arc. Large values
    flag observations where the label is unreliable (GPS off the route, or route
    not covering the start/end of the track).
    """
    if trip.route is None:
        raise ValueError(f"trip {trip.trip_id} has no .route file")
    route_arcs = trip.route["arc_id"].to_numpy()
    g = arc_geometry(trip).loc[route_arcs]
    px, py = track_xy(trip)

    # c[i, j]: distance of GPS point i to route arc j  (N x M)
    c, _ = point_segment_distance(px[:, None], py[:, None],
                                  g["ax"].to_numpy()[None, :], g["ay"].to_numpy()[None, :],
                                  g["bx"].to_numpy()[None, :], g["by"].to_numpy()[None, :])
    n, m = c.shape
    idx = np.arange(m)

    D = c[0].copy()
    back = np.empty((n, m), dtype=np.int32)   # back[i, j] = best k for point i-1
    back[0] = idx
    for i in range(1, n):
        prefix_min = np.minimum.accumulate(D)
        # position of the running minimum = last k <= j where D[k] equals it
        prefix_arg = np.maximum.accumulate(np.where(D == prefix_min, idx, -1))
        back[i] = prefix_arg
        D = c[i] + prefix_min

    # backtrack from the best final position
    seq = np.empty(n, dtype=np.int64)
    seq[-1] = int(np.argmin(D))
    for i in range(n - 1, 0, -1):
        seq[i - 1] = back[i, seq[i]]

    return pd.DataFrame({
        "obs_idx": trip.track["obs_idx"].to_numpy(),
        "ref_arc": route_arcs[seq],
        "ref_seq": seq,
        "ref_dist_m": c[np.arange(n), seq],
    })
