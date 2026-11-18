"""PathSnap evaluator UI (Streamlit + Folium).

Runs on Laptop 2 and talks to the API on Laptop 1. Everything shown comes from the API: this file reads NO data
files and never sees ground truth (the API only returns aggregate accuracy numbers).

Run:  streamlit run src/ui/app.py --server.port 8501
Env:  API_URL   API base address (default http://localhost:8000; editable in the sidebar)
      USE_FAKE  "1" = use the local test fixture src/ui/dev/fake_api.json instead of HTTP (development only;
                remove this switch before the demo)
Coordinates from the API are [lat, lon] (Leaflet order) and are drawn as-is.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
USE_FAKE = os.environ.get("USE_FAKE") == "1"
FAKE_PATH = Path(__file__).parent / "dev" / "fake_api.json"
PREDICT_TIMEOUT_S = 120
GPS_MAX_POINTS = 3000          # thin the blue dots on very long trips so the map stays responsive

st.set_page_config(page_title="PathSnap", page_icon=":world_map:", layout="wide")


# ------------------------------------------------------------------ API access
class ApiError(Exception):
    """Any failure talking to the API, with a message that is safe to show to the evaluator."""


def _fake():
    return json.loads(FAKE_PATH.read_text())


def api_get(base: str, path: str, timeout: float = 10):
    if USE_FAKE:
        return _fake()[path.strip("/")]
    try:
        r = requests.get(base.rstrip("/") + path, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the API at {base}. Is it running, and is the address/firewall correct?")
    except requests.exceptions.Timeout:
        raise ApiError(f"The API at {base} did not answer within {timeout:.0f} s.")
    return _json_or_error(r)


def api_predict(base: str, trip_id: str):
    if USE_FAKE:
        fx = _fake()["predict"]
        if trip_id not in fx:
            raise ApiError(f"API error 404: trip {trip_id} not found")
        return fx[trip_id]
    try:
        r = requests.post(base.rstrip("/") + "/predict", json={"trip_id": trip_id}, timeout=PREDICT_TIMEOUT_S)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the API at {base}. Is it running, and is the address/firewall correct?")
    except requests.exceptions.Timeout:
        raise ApiError(f"The API did not answer within {PREDICT_TIMEOUT_S} s.")
    return _json_or_error(r)


def _json_or_error(r: requests.Response):
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text[:300]
        raise ApiError(f"API error {r.status_code}: {detail}")
    try:
        return r.json()
    except ValueError:
        raise ApiError("The API returned a response that is not valid JSON.")


# ------------------------------------------------------------------ map
LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 8px 12px;
            border: 1px solid #999; border-radius: 4px; font-size: 13px; line-height: 1.5;">
<b>Trip {trip}</b><br>
<span style="color:#1f4fd1;">&#9679;</span> raw GPS observations<br>
<span style="color:#d62728;">&#9473;</span> HMM predicted route<br>
<span style="color:#ff7f0e;">&#9473;</span> classifier-only route
</div>"""


def build_map(res: dict, show_classifier: bool) -> folium.Map:
    gps = res["gps"]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    if show_classifier:
        clf = folium.FeatureGroup(name="Classifier-only route (orange)", show=True)
        for line in res.get("classifier_route", []):
            if len(line) > 1:
                folium.PolyLine(line, color="#ff7f0e", weight=4, opacity=0.85).add_to(clf)
        clf.add_to(m)

    hmm = folium.FeatureGroup(name="HMM predicted route (red)", show=True)
    for line in res.get("hmm_route", []):
        if len(line) > 1:
            folium.PolyLine(line, color="#d62728", weight=4, opacity=0.9).add_to(hmm)
    hmm.add_to(m)

    pts = folium.FeatureGroup(name=f"Raw GPS ({len(gps):,} points)", show=True)
    step = max(1, len(gps) // GPS_MAX_POINTS)
    for i in range(0, len(gps), step):
        folium.CircleMarker(gps[i], radius=2, color="#1f4fd1", fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"obs {i}").add_to(pts)
    pts.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(trip=res["trip_id"])))
    return m


# ------------------------------------------------------------------ UI
st.title("PathSnap: map-matching evaluation")

with st.sidebar:
    st.header("API")
    api_url = st.text_input("API address", value=DEFAULT_API_URL, help="Set the default with the API_URL env var")
    if USE_FAKE:
        st.warning("USE_FAKE=1: showing the local test fixture, not a real API.")
    if st.button("Reconnect / reload trips"):
        st.session_state.pop("result", None)
        st.rerun()

# --- health
health = None
try:
    health = api_get(api_url, "/health")
    st.success(f"API connected - model: **{health.get('model', '?')}** "
               f"(data: {health.get('data_dir', '?')}, {health.get('n_trips', '?')} trips)")
except ApiError as e:
    st.error(str(e))

# --- trip list
trips = []
if health is not None:
    try:
        trips = api_get(api_url, "/trips")["trips"]
    except (ApiError, KeyError) as e:
        st.error(f"Could not load the trip list: {e}")

if trips:
    labels = {t["trip_id"]: f"{t['trip_id']}  ({t['n_obs']:,} GPS points)" for t in trips}
    c1, c2 = st.columns([3, 1])
    trip_id = c1.selectbox("Evaluation trip", list(labels), format_func=labels.get)
    c2.write("")
    run = c2.button("Run map-matching", type="primary", width="stretch")

    if run:
        with st.spinner(f"Running the pipeline on trip {trip_id} (up to {PREDICT_TIMEOUT_S} s)..."):
            try:
                st.session_state["result"] = api_predict(api_url, trip_id)
                st.session_state.pop("error", None)
            except ApiError as e:
                st.session_state["error"] = str(e)
                st.session_state.pop("result", None)

if "error" in st.session_state:
    st.error(st.session_state["error"])

res = st.session_state.get("result")
if res:
    try:
        met, tim = res["metrics"], res["timings_ms"]
        st.subheader(f"Trip {res['trip_id']}  -  model: {res.get('model', '?')}")
        fmt = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"
        cols = st.columns(5)
        cols[0].metric("HMM Top-1 accuracy", fmt(met.get("top1_accuracy_hmm")))
        cols[1].metric("Classifier-only Top-1", fmt(met.get("top1_accuracy_classifier")))
        cols[2].metric("Route continuity (HMM)", fmt(met.get("route_continuity_hmm")),
                       help=f"classifier-only: {fmt(met.get('route_continuity_classifier'))}")
        cols[3].metric("GPS points", f"{res['n_obs']:,}")
        cols[4].metric("Total time", f"{tim['total'] / 1000:.2f} s")

        left, right = st.columns([1, 2])
        left.markdown("**Timings (ms)**")
        left.dataframe(pd.DataFrame({"stage": list(tim), "ms": list(tim.values())}), hide_index=True,
                       width="stretch")
        left.caption(f"Observations evaluated: {met.get('n_obs_evaluated', '?')}")
        show_clf = right.checkbox("Show classifier-only route (orange)", value=True)
        components.html(build_map(res, show_clf)._repr_html_(), height=600)
    except (KeyError, TypeError, ValueError) as e:
        st.error(f"The API response is missing or has an unexpected field: {e!r}")
