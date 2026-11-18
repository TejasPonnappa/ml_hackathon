"""Geometry helpers in metric coordinates (owner: Person A).

PROJECTION CHOICE: per-trip local Azimuthal Equidistant (AEQD)
  * EDA showed the trips are spread over the whole world (Europe, Russia,
    North/South America, Asia, Africa), with longitudes from about -100 to +137
    and latitudes from about -35 to +61. A single fixed CRS (e.g. one UTM zone)
    is only valid within a 6-degree band and breaks for most trips.
  * Each trip is local (a few km to tens of km), so we build one projection per
    trip, centred on the median of its GPS track:
        +proj=aeqd +lat_0=<lat0> +lon_0=<lon0> +datum=WGS84 +units=m
    Distances measured from the centre are exact; the scale error at distance d
    from the centre is about (d/R)^2/6, i.e. ~0.001 % at 50 km. That is negligible
    compared with GPS noise (metres), so plain Euclidean geometry on (x, y)
    metres is valid inside a trip.
  * The centre comes from the TRACK only (never the route), so this is safe to
    use in the prediction pipeline.
  * Degree-Euclidean distance is wrong: at 55 N one degree of longitude is ~64 km
    while one degree of latitude is ~111 km.
  * haversine() is kept as an independent sanity check.

All functions are vectorised with numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

EARTH_RADIUS_M = 6_371_008.8


class LocalProjector:
    """lon/lat <-> local metric (x east, y north) centred on (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        crs = (f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} "
               f"+datum=WGS84 +units=m +no_defs")
        # always_xy=True -> input/output order is (lon, lat) / (x, y); avoids lat/lon swaps
        self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def trip_projector(trip) -> LocalProjector:
    """Projector centred on the median GPS position of a trip (cached on the Trip)."""
    proj = getattr(trip, "_proj", None)
    if proj is None:
        proj = LocalProjector(trip.track["lon"].median(), trip.track["lat"].median())
        trip._proj = proj
    return proj


def track_xy(trip):
    """(x, y) metres of every GPS observation of a trip."""
    return trip_projector(trip).to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())


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

    Math: project P onto the line through A,B:
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


def arc_geometry(trip) -> pd.DataFrame:
    """Per-arc metric geometry for a trip: endpoints, length, bearing, midpoint.

    Arcs are straight segments between their two nodes (the data gives only
    node coordinates, no intermediate shape points). Indexed by arc_id.
    Cached on the Trip object because candidates, labels and features all need it.
    """
    g = getattr(trip, "_geom", None)
    if g is not None:
        return g
    nodes, arcs = trip.nodes, trip.arcs
    nx_, ny_ = trip_projector(trip).to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
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
    }).set_index("arc_id", drop=False)
    trip._geom = g
    return g
