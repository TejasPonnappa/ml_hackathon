"""Report map (section 6): GPS + road network + reference route + classifier-only route + HMM route for ONE trip.

OFFLINE script for the report - it reads the data files (the UI must not). Layers (layer control, top right):
  road network within ~200 m of the track (grey) | reference route (green) | classifier-only top-1 (orange)
  | HMM/Viterbi route (red) | raw GPS (blue)

Predictions come from
  * the API (recommended, gives BOTH classifier-only and HMM):   --api http://localhost:8000
  * or, with no --api, a local classifier-only fallback using the saved model (src.ml.a_inference; point
    PATHSNAP_MODEL at a .joblib bundle). The HMM layer is then omitted.
The reference route is rebuilt with load_trip(...) + align_route(...) (the same labels used for training).

Run:  python -m src.viz.folium_map 00000007 --partition student_test --api http://localhost:8000
Output: results/b_folium_<trip>.html
"""
from __future__ import annotations

import argparse

import folium
import numpy as np
import requests

from src.common.config import RESULTS_DIR
from src.common.geo import arc_geometry, track_xy
from src.common.io import load_trip
from src.common.labels import align_route

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 8px 12px;
            border: 1px solid #999; border-radius: 4px; font-size: 13px; line-height: 1.5;">
<b>{title}</b><br>
<span style="color:#1f4fd1;">&#9679;</span> raw GPS observations<br>
<span style="color:#888;">&#9473;</span> road network (arcs within ~200 m)<br>
<span style="color:#1a9e1a;">&#9473;</span> reference route (aligned .route)<br>
<span style="color:#ff7f0e;">&#9473;</span> classifier-only route<br>
<span style="color:#d62728;">&#9473;</span> HMM predicted route{note}
</div>"""


def _lines(trip, arc_ids):
    """Arc ids -> list of [[lat, lon], [lat, lon]] segments for folium."""
    arcs, nodes = trip.arcs, trip.nodes
    u = arcs.from_node.to_numpy()[arc_ids]; v = arcs.to_node.to_numpy()[arc_ids]
    la, lo = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    return [[[la[a], lo[a]], [la[b], lo[b]]] for a, b in zip(u, v)]


def _dedupe(seq):
    seq = np.asarray(seq)
    return seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq


def predictions_from_api(api_url: str, trip_id: str) -> dict:
    r = requests.post(api_url.rstrip("/") + "/predict", json={"trip_id": trip_id}, timeout=120)
    r.raise_for_status()
    return r.json()


def classifier_only_local(trip) -> list[list[list[float]]]:
    """Top-1 arc per observation from the saved model -> polylines (one per distinct consecutive arc)."""
    from src.ml.a_inference import score_trip
    scored = score_trip(trip)
    top = scored.sort_values(["obs_idx", "prob"], ascending=[True, False]).drop_duplicates("obs_idx")
    arcs = _dedupe(top.sort_values("obs_idx")["arc_id"].to_numpy())
    return _lines(trip, arcs)


def make_map(trip_id: str, partition: str = "student_test", api_url: str | None = None,
             pad_m: float = 200.0) -> folium.Map:
    trip = load_trip(trip_id, partition)                       # reads the reference route: offline use only
    tr = trip.track
    pred = predictions_from_api(api_url, trip_id) if api_url else None

    g = arc_geometry(trip)
    x, y = track_xy(trip)
    near = ((g[["ax", "bx"]].max(axis=1) > x.min() - pad_m) & (g[["ax", "bx"]].min(axis=1) < x.max() + pad_m) &
            (g[["ay", "by"]].max(axis=1) > y.min() - pad_m) & (g[["ay", "by"]].min(axis=1) < y.max() + pad_m))

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])

    net = folium.FeatureGroup(name=f"Road network ({int(near.sum()):,} arcs near track)")
    folium.PolyLine(_lines(trip, np.flatnonzero(near.to_numpy())), color="#888", weight=1.5, opacity=.7).add_to(net)
    net.add_to(m)

    ref_arcs = _dedupe(align_route(trip)["ref_arc"].to_numpy())
    ref = folium.FeatureGroup(name=f"Reference route ({len(ref_arcs)} arcs)")
    folium.PolyLine(_lines(trip, ref_arcs), color="#1a9e1a", weight=7, opacity=.75).add_to(ref)
    ref.add_to(m)

    clf_lines = pred["classifier_route"] if pred else classifier_only_local(trip)
    clf = folium.FeatureGroup(name="Classifier-only route")
    for line in clf_lines:
        folium.PolyLine(line, color="#ff7f0e", weight=4, opacity=.9).add_to(clf)
    clf.add_to(m)

    note = ""
    if pred:
        hmm = folium.FeatureGroup(name="HMM predicted route")
        for line in pred["hmm_route"]:
            folium.PolyLine(line, color="#d62728", weight=3, opacity=.95).add_to(hmm)
        hmm.add_to(m)
    else:
        note = " <i>(not available: run with --api)</i>"

    gps = folium.FeatureGroup(name=f"Raw GPS ({len(tr):,} points)")
    step = max(1, len(tr) // 3000)
    for r in tr.iloc[::step].itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=2, color="#1f4fd1", fill=True, fill_opacity=.8, weight=0,
                            tooltip=f"obs {r.obs_idx}, t={r.t:.0f}s").add_to(gps)
    folium.Marker([tr.lat.iloc[0], tr.lon.iloc[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(gps)
    folium.Marker([tr.lat.iloc[-1], tr.lon.iloc[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(gps)
    gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=f"PathSnap - {partition} trip {trip_id}", note=note)))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_id")
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--api", default=None, help="API base URL; omit for the local classifier-only fallback")
    a = ap.parse_args()
    out = RESULTS_DIR / f"b_folium_{a.trip_id}.html"
    make_map(a.trip_id, a.partition, a.api).save(out)
    print("saved", out)
