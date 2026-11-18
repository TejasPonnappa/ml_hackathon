"""PathSnap FastAPI backend (Laptop 1). Implements API_CONTRACT.md.

Endpoints: GET /health, GET /trips, POST /predict {"trip_id": "..."}
Run:  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
Env:  DATA_DIR           partition name (e.g. student_test) or absolute folder of trips to serve
      PATHSNAP_DATA_ROOT folder containing the partition folders (see src/common/config.py)
      PATHSNAP_MODEL     optional model bundle override (default artifacts/model.joblib)

Pipeline for /predict (order matters - ground truth is only touched in step 5):
  1. load the trip WITHOUT its route
  2. candidates -> 19 features -> model probabilities   (src/ml/a_inference.score_trip)
  3. classifier-only = most probable arc per observation; HMM = Viterbi decoder (src/hmm, via decode_hmm)
  4. build [lat, lon] geometry for GPS points and both routes
  5. ONLY NOW load the route (server side), align it to the GPS points, compute aggregate accuracy
The response never contains the reference route, labels or per-point correctness.
Every request is logged as one line to the terminal and to results/logs/b_api_requests.log.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.common.config import RESULTS_DIR
from src.common.io import list_trips, load_trip, partition_dir, trip_files
from src.common.labels import align_route
from src.common.metrics import MAX_HOPS, _adjacency, route_continuity
from src.ml.a_inference import load_bundle, model_info, score_trip, to_hmm_input

DATA_DIR = os.environ.get("DATA_DIR", "student_test")

# ------------------------------------------------------------------ logging
LOG_FILE = RESULTS_DIR / "logs" / "b_api_requests.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False

@asynccontextmanager
async def lifespan(_app):
    load_bundle()                                # load the model once, before the first request
    info = model_info()
    log.info(f"STARTUP | model={info['model']} ({info['path']}) | data_dir={partition_dir(DATA_DIR)} "
             f"| n_trips={len(list_trips(DATA_DIR))} | HMM available={_hmm_available()} | log={LOG_FILE}")
    if _hmm_available():
        log.info("STARTUP | " + " ".join(_hmm_params_note().split()))
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _hmm_available() -> bool:
    try:
        from src.hmm.viterbi import decode_trip  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------ HMM adapter
def decode_hmm(trip, hmm_in: dict) -> dict[int, int] | None:
    """Call Person B's HMM/Viterbi decoder. Returns {obs_idx: arc_id}, or None if not available yet.
    This is the ONLY place the API depends on src/hmm."""
    try:
        from src.hmm.viterbi import decode_trip      # Person B's entry point: decode_trip(trip, hmm_in, params=None)
    except ImportError:
        return None
    # params=None -> the decoder reads its tuned parameters from results/b_hmm_best.json itself
    return {int(k): int(v) for k, v in decode_trip(trip, hmm_in).items()}


HMM_PARAMS_FILE = RESULTS_DIR / "b_hmm_best.json"


def _hmm_params_note() -> str:
    """Startup log: which decoder parameter file the HMM will use (read, never copied into this file)."""
    if not HMM_PARAMS_FILE.exists():
        return "HMM params: b_hmm_best.json MISSING -> decoder defaults"
    return f"HMM params: {HMM_PARAMS_FILE.name} = {HMM_PARAMS_FILE.read_text(encoding='utf-8')[:300].strip()}"


# ------------------------------------------------------------------ geometry helpers
def _bfs_path(adj, a: int, b: int, max_hops: int = MAX_HOPS) -> list[int] | None:
    """Shortest list of intermediate arcs leading from arc a to arc b (directed), or None."""
    frm, to, order, offsets = adj
    target = frm[b]
    start = int(to[a])
    if start == target:
        return []
    prev = {start: None}                 # node -> (previous node, arc used)
    frontier = [start]
    for _ in range(max_hops):
        nxt = []
        for node in frontier:
            for arc in order[offsets[node]:offsets[node + 1]]:
                w = int(to[arc])
                if w in prev:
                    continue
                prev[w] = (node, int(arc))
                if w == target:
                    path, cur = [], w
                    while prev[cur] is not None:
                        cur, arc_used = prev[cur]
                        path.append(arc_used)
                    return path[::-1]
                nxt.append(w)
        frontier = nxt
        if not frontier:
            break
    return None


def route_polylines(arc_seq: list[int], trip, data_dir) -> list[list[list[float]]]:
    """Per-observation arc sequence -> list of [lat, lon] polylines.
    Repeated arcs are collapsed; consecutive arcs connected within MAX_HOPS are joined (the intermediate
    arcs are filled in); otherwise a new polyline starts (gap / jump)."""
    seq = [a for i, a in enumerate(arc_seq) if a >= 0 and (i == 0 or a != arc_seq[i - 1])]
    if not seq:
        return []
    adj = _adjacency(trip.trip_id, str(data_dir))
    frm, to = adj[0], adj[1]
    lat, lon = trip.nodes["lat"].to_numpy(), trip.nodes["lon"].to_numpy()
    pt = lambda n: [round(float(lat[n]), 6), round(float(lon[n]), 6)]
    lines, cur = [], [pt(frm[seq[0]]), pt(to[seq[0]])]
    for a, b in zip(seq[:-1], seq[1:]):
        path = _bfs_path(adj, a, b)
        if path is None:                         # not connected: start a new polyline
            lines.append(cur)
            cur = [pt(frm[b]), pt(to[b])]
        else:
            for arc in path + [b]:
                cur.append(pt(to[arc]))
    lines.append(cur)
    return lines


# ------------------------------------------------------------------ endpoints
class PredictRequest(BaseModel):
    trip_id: str


def _trips() -> list[str]:
    return list_trips(DATA_DIR)


def _n_obs(trip_id: str) -> int:
    with open(trip_files(trip_id, DATA_DIR)["track"], "rb") as f:
        return sum(1 for line in f if line.strip())


@app.get("/health")
def health(request: Request):
    out = {"status": "ok", "model": model_info()["model"], "data_dir": str(DATA_DIR),
           "n_trips": len(_trips()), "server_time": datetime.now().isoformat(timespec="seconds")}
    log.info(f"GET /health | client={request.client.host}")
    return out


@app.get("/trips")
def trips(request: Request):
    ids = _trips()
    log.info(f"GET /trips | client={request.client.host} | n_trips={len(ids)} | data_dir={DATA_DIR}")
    return {"data_dir": str(DATA_DIR), "trips": [{"trip_id": t, "n_obs": _n_obs(t)} for t in ids]}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    t0 = time.perf_counter()
    tid = req.trip_id
    if tid not in _trips():
        log.info(f"POST /predict | trip={tid} | client={request.client.host} | 404 not found")
        raise HTTPException(status_code=404, detail=f"trip {tid} not found")
    try:
        # 1. load WITHOUT the route
        trip = load_trip(tid, DATA_DIR, with_route=False)
        n = len(trip.track)
        t_load = time.perf_counter()

        # 2. candidates -> features -> probabilities
        scored = score_trip(trip)
        t_model = time.perf_counter()

        # 3. classifier-only and HMM decisions per observation (-1 = no candidate)
        clf = np.full(n, -1, dtype=np.int64)
        if len(scored):
            best = scored.sort_values(["obs_idx", "prob"], ascending=[True, False]).drop_duplicates("obs_idx")
            clf[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
        hmm_map = decode_hmm(trip, to_hmm_input(scored)) if len(scored) else {}
        hmm = None
        if hmm_map is not None:
            hmm = np.full(n, -1, dtype=np.int64)
            for o, a in hmm_map.items():
                hmm[o] = a
        t_hmm = time.perf_counter()

        # 4. geometry
        gps = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(trip.track["lat"], trip.track["lon"])]
        clf_route = route_polylines(clf.tolist(), trip, DATA_DIR)
        hmm_route = route_polylines(hmm.tolist(), trip, DATA_DIR) if hmm is not None else []

        # 5. server-side evaluation (route loaded only now)
        metrics = {"top1_accuracy_hmm": None, "top1_accuracy_classifier": None, "route_continuity_hmm": None,
                   "route_continuity_classifier": None, "n_obs_evaluated": 0}
        if trip_files(tid, DATA_DIR)["route"].exists():
            ref = align_route(load_trip(tid, DATA_DIR, with_route=True))["ref_arc"].to_numpy()
            metrics["top1_accuracy_classifier"] = float((clf == ref).mean())
            metrics["route_continuity_classifier"] = route_continuity(clf[clf >= 0], tid, str(DATA_DIR))
            if hmm is not None:
                metrics["top1_accuracy_hmm"] = float((hmm == ref).mean())
                metrics["route_continuity_hmm"] = route_continuity(hmm[hmm >= 0], tid, str(DATA_DIR))
            metrics["n_obs_evaluated"] = int(n)
        t_end = time.perf_counter()
    except HTTPException:
        raise
    except Exception as e:                      # report a short message, keep the server alive
        log.info(f"POST /predict | trip={tid} | client={request.client.host} | 500 {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    ms = lambda a, b: int(round(1000 * (b - a)))
    timings = {"load": ms(t0, t_load), "model": ms(t_load, t_model), "hmm": ms(t_model, t_hmm),
               "evaluation": ms(t_hmm, t_end), "total": ms(t0, t_end)}
    f4 = lambda v: "n/a" if v is None else f"{v:.4f}"
    log.info(f"POST /predict | trip={tid} | client={request.client.host} | n_obs={n} | "
             f"load={timings['load']}ms model={timings['model']}ms hmm={timings['hmm']}ms "
             f"eval={timings['evaluation']}ms total={timings['total']}ms | "
             f"top1_hmm={f4(metrics['top1_accuracy_hmm'])} top1_clf={f4(metrics['top1_accuracy_classifier'])} | "
             f"cont_hmm={f4(metrics['route_continuity_hmm'])}"
             + ("" if hmm is not None else " | HMM not available (src/hmm/viterbi.py missing)"))
    return {"trip_id": tid, "model": model_info()["model"], "n_obs": n, "gps": gps,
            "hmm_route": hmm_route, "classifier_route": clf_route, "metrics": metrics, "timings_ms": timings}
