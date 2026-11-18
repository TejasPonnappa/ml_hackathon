"""Geometry helpers in metric coordinates (owner: Person A).

WHY UTM ZONE 32N (EPSG:32632)?
  * The data lies around lon ~8.6 E, lat ~49.8 N (Darmstadt / Rhine-Main area,
    Germany). UTM zone 32N covers 6-12 E, so every trip is inside one zone.
  * UTM is conformal and its scale error inside a zone is < 0.1 %, i.e. < 1 m
    over a 1 km trip, far below GPS noise (~5-20 m). Distances and angles
    computed with plain Euclidean geometry on (x, y) metres are therefore valid.
  * Raw degree-Euclidean distance is WRONG here: at 49.8 N one degree of
    longitude is ~72 km but one degree of latitude is ~111 km, so it would
    distort distances by ~35 % depending on direction.
  * haversine() below is kept only as an independent sanity check.

All functions are vectorised with numpy so they work on whole trips at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import METRIC_CRS

# always_xy=True -> input order is (lon, lat), output order is (x=easting, y=northing).
# This guards against the classic lat/lon swap bug.
_TO_METRIC = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
_TO_LONLAT = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)

EARTH_RADIUS_M = 6_371_008.8


def to_xy(lon, lat):
    """lon/lat in degrees -> (x, y) in metres (UTM 32N)."""
    x, y = _TO_METRIC.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def to_lonlat(x, y):
    """Inverse of to_xy (used to send geometry back to maps)."""
    lon, lat = _TO_LONLAT.transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon), np.asarray(lat)


def add_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a DataFrame with lon/lat columns plus projected x, y columns."""
    out = df.copy()
    out["x"], out["y"] = to_xy(out["lon"].to_numpy(), out["lat"].to_numpy())
    return out


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (sanity check for the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(dx, dy):
    """Compass bearing of a vector in degrees: 0 = north, 90 = east, range [0, 360)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (all arrays broadcast together).

    Returns
    -------
    dist : metres from P to the closest point on segment AB
    t    : position of that closest point along AB, clamped to [0, 1]
           (0 = at A / from-node, 1 = at B / to-node)

    Math: project P onto the infinite line through A,B:
          t = ((P-A)·(B-A)) / |B-A|^2, clamp t to [0,1], closest = A + t(B-A).
    Zero-length segments (A == B) fall back to the distance to A.
    """
    px, py, ax, ay, bx, by = map(lambda v: np.asarray(v, float), (px, py, ax, ay, bx, by))
    abx, aby = bx - ax, by - ay
    len2 = abx * abx + aby * aby
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * abx + (py - ay) * aby) / len2
    t = np.where(len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * abx, ay + t * aby
    dist = np.hypot(px - cx, py - cy)
    return dist, t


def arc_geometry(nodes: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    """Per-arc metric geometry: endpoints, length, bearing, midpoint.

    Arcs are treated as straight segments between their two nodes (the data
    only gives node coordinates, no intermediate shape points).
    Returned DataFrame is indexed by arc_id.
    """
    nx_, ny_ = to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    u = arcs["from_node"].to_numpy()
    v = arcs["to_node"].to_numpy()
    ax, ay, bx, by = nx_[u], ny_[u], nx_[v], ny_[v]
    g = pd.DataFrame({
        "arc_id": arcs["arc_id"].to_numpy(),
        "from_node": u, "to_node": v,
        "ax": ax, "ay": ay, "bx": bx, "by": by,
        "length_m": np.hypot(bx - ax, by - ay),
        "bearing": bearing_deg(bx - ax, by - ay),
        "mx": (ax + bx) / 2, "my": (ay + by) / 2,
    })
    return g.set_index("arc_id", drop=False)
