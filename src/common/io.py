"""Raw-data loaders for PathSnap (owner: Person A).

File formats (verified against the data, see report/a_section2_notes.md):
  <id>.track : lon  lat  timestamp_s     (whitespace separated, no header, ~1 Hz)
  <id>.nodes : lon  lat                  (row index = node_id, 0-based)
  <id>.arcs  : from_node  to_node        (row index = arc_id, 0-based; directed)
  <id>.route : arc_id                    (chronological reference route)

Important: a .track row does NOT correspond to a .nodes/.arcs row. The link
between GPS points and arcs is exactly what we have to learn.

Everything is returned as pandas DataFrames with explicit column names so other
modules never depend on column positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_ROOT

EXTENSIONS = ("track", "nodes", "arcs", "route")


@dataclass
class Trip:
    """All raw information for one trip.

    track : columns [obs_idx, lon, lat, t]           one row per GPS observation
    nodes : columns [node_id, lon, lat]              index == node_id
    arcs  : columns [arc_id, from_node, to_node]     index == arc_id
    route : columns [seq, arc_id]                    reference route, in order
            (None if the trip has no .route file, e.g. a hidden partition)
    """
    trip_id: str
    partition_dir: Path
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame | None


def partition_dir(partition: str | Path) -> Path:
    """Resolve a partition name ('train', 'student_test') or an explicit folder path."""
    p = Path(partition)
    return p if p.is_dir() else DATA_ROOT / str(partition)


def list_trips(partition: str | Path) -> list[str]:
    """Sorted trip ids (folder names) in a partition."""
    pdir = partition_dir(partition)
    if not pdir.is_dir():
        raise FileNotFoundError(f"Partition folder not found: {pdir} "
                                f"(set PATHSNAP_DATA_ROOT?)")
    return sorted(d.name for d in pdir.iterdir() if d.is_dir())


def trip_files(trip_id: str, partition: str | Path) -> dict[str, Path]:
    """Paths of the four raw files of a trip (they may not all exist)."""
    tdir = partition_dir(partition) / trip_id
    return {ext: tdir / f"{trip_id}.{ext}" for ext in EXTENSIONS}


def _read_ws(path: Path, names: list[str], dtypes: dict) -> pd.DataFrame:
    """Read a whitespace-separated, header-less file."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=names,
                       dtype=dtypes, engine="c")


def load_track(path: Path) -> pd.DataFrame:
    df = _read_ws(path, ["lon", "lat", "t"],
                  {"lon": "float64", "lat": "float64", "t": "float64"})
    df.insert(0, "obs_idx", range(len(df)))   # position in the file = observation index
    return df


def load_nodes(path: Path) -> pd.DataFrame:
    df = _read_ws(path, ["lon", "lat"], {"lon": "float64", "lat": "float64"})
    df.insert(0, "node_id", range(len(df)))   # 0-based row index is the node id
    return df


def load_arcs(path: Path) -> pd.DataFrame:
    df = _read_ws(path, ["from_node", "to_node"],
                  {"from_node": "int64", "to_node": "int64"})
    df.insert(0, "arc_id", range(len(df)))    # 0-based row index is the arc id
    return df


def load_route(path: Path) -> pd.DataFrame:
    df = _read_ws(path, ["arc_id"], {"arc_id": "int64"})
    df.insert(0, "seq", range(len(df)))       # order along the route
    return df


def load_trip(trip_id: str, partition: str | Path = "train",
              with_route: bool = True) -> Trip:
    """Load one trip.

    with_route=False lets the prediction pipeline load a trip WITHOUT touching
    the reference route, so ground truth can never leak into the model input.
    """
    f = trip_files(trip_id, partition)
    route = None
    if with_route and f["route"].exists():
        route = load_route(f["route"])
    return Trip(
        trip_id=trip_id,
        partition_dir=partition_dir(partition),
        track=load_track(f["track"]),
        nodes=load_nodes(f["nodes"]),
        arcs=load_arcs(f["arcs"]),
        route=route,
    )


def iter_trips(partition: str | Path, with_route: bool = True):
    """Yield Trip objects for every trip in a partition."""
    for tid in list_trips(partition):
        yield load_trip(tid, partition, with_route=with_route)
