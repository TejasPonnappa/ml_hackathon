"""EDA Folium map: raw GPS + nearby road network + reference route for one trip.

Layers (toggle with the layer control, top right):
  * Road network (grey)   - arcs within 200 m of the track's bounding box
                            (full networks have up to ~300k arcs; clipping keeps the HTML light)
  * Reference route (green)
  * Raw GPS points (blue), start (green marker) / end (red marker)
Run:  python -m src.viz.eda_folium 00000001 train
Output: results/a_eda_folium_<partition>_<trip>.html
"""
from __future__ import annotations

import sys

import folium
import numpy as np

from src.common.config import RESULTS_DIR
from src.common.geo import arc_geometry, track_xy
from src.common.io import load_trip

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #999; border-radius: 4px; font-size: 13px;">
<b>{title}</b><br>
<span style="color:#1f4fd1;">&#9679;</span> raw GPS observations<br>
<span style="color:#888;">&#9473;</span> road network (arcs)<br>
<span style="color:#1a9e1a;">&#9473;</span> reference route (.route)<br>
</div>"""


def _lines(trip, arc_ids):
    """Arc ids -> list of [[lat, lon], [lat, lon]] segments for folium."""
    nodes, arcs = trip.nodes, trip.arcs
    u = arcs.from_node.to_numpy()[arc_ids]; v = arcs.to_node.to_numpy()[arc_ids]
    la, lo = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    return [[[la[a], lo[a]], [la[b], lo[b]]] for a, b in zip(u, v)]


def make_map(trip_id: str, partition: str = "train", pad_m: float = 200.0) -> folium.Map:
    trip = load_trip(trip_id, partition)
    g = arc_geometry(trip)
    x, y = track_xy(trip)
    near = ((g[["ax", "bx"]].max(axis=1) > x.min() - pad_m) & (g[["ax", "bx"]].min(axis=1) < x.max() + pad_m) &
            (g[["ay", "by"]].max(axis=1) > y.min() - pad_m) & (g[["ay", "by"]].min(axis=1) < y.max() + pad_m))
    tr = trip.track

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])

    net = folium.FeatureGroup(name=f"Road network ({int(near.sum()):,} arcs near track)")
    folium.PolyLine(_lines(trip, np.flatnonzero(near.to_numpy())), color="#888", weight=1.5, opacity=.7).add_to(net)
    net.add_to(m)

    if trip.route is not None:
        rt = folium.FeatureGroup(name=f"Reference route ({len(trip.route)} arcs)")
        folium.PolyLine(_lines(trip, trip.route.arc_id.to_numpy()), color="#1a9e1a", weight=5, opacity=.8).add_to(rt)
        rt.add_to(m)

    gps = folium.FeatureGroup(name=f"Raw GPS ({len(tr):,} points)")
    step = max(1, len(tr) // 3000)          # keep the HTML responsive for very long trips
    for r in tr.iloc[::step].itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=2, color="#1f4fd1", fill=True, fill_opacity=.8, weight=0,
                            tooltip=f"obs {r.obs_idx}, t={r.t:.0f}s").add_to(gps)
    folium.Marker([tr.lat.iloc[0], tr.lon.iloc[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(gps)
    folium.Marker([tr.lat.iloc[-1], tr.lon.iloc[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(gps)
    gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=f"PathSnap EDA - {partition} trip {trip_id}")))
    return m


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "00000001"
    part = sys.argv[2] if len(sys.argv) > 2 else "train"
    out = RESULTS_DIR / f"a_eda_folium_{part}_{tid}.html"
    make_map(tid, part).save(out)
    print("saved", out)
