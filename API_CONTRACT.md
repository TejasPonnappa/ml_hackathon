# PathSnap API contract (API on Laptop 1 ↔ UI on Laptop 2)

Single source of truth for the interface between `src/api/` (Tejasvi) and `src/ui/` (Yuvraj).
Change it only by agreement, and update this file in the same PR.

## 0. General rules
- Transport: HTTP + JSON (`Content-Type: application/json`), UTF-8. No authentication, no secrets.
- Base URL: `http://<laptop1-ip>:8000`. API started with
  `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`. CORS enabled for all origins.
- **Coordinates are always `[lat, lon]`** (Leaflet/Folium order). Raw files are lon-lat; the API swaps them.
  Rounded to 6 decimals (~0.1 m).
- `trip_id` is always a **string** with leading zeros, e.g. `"00000007"`.
- **Ground truth never leaves the API.** No reference route, reference arcs, labels or per-point
  correct/incorrect flags in any response. Only aggregate accuracy numbers computed server-side.
- The UI never reads data files; everything it shows comes from these endpoints.
- No hardcoded trip lists or outputs anywhere: trips come from the folder `DATA_DIR` at request time.

## 1. Configuration (environment variables)
| Variable | Side | Default | Meaning |
|---|---|---|---|
| `PATHSNAP_DATA_ROOT` | API | `<repo>/data_raw/Student folder` | folder containing partition folders |
| `DATA_DIR` | API | `student_test` | partition name **or** absolute path of the folder whose trips are served (evaluator folder at demo) |
| `PATHSNAP_MODEL` | API | `artifacts/model.joblib` | optional override of the model bundle |
| `API_URL` | UI | `http://localhost:8000` | base URL of the API |

## 2. `GET /health`
```json
{
  "status": "ok",
  "model": "lightgbm_tuned",
  "data_dir": "student_test",
  "n_trips": 10,
  "server_time": "2026-11-18T13:45:02"
}
```

## 3. `GET /trips`
```json
{
  "data_dir": "student_test",
  "trips": [
    {"trip_id": "00000000", "n_obs": 2503},
    {"trip_id": "00000007", "n_obs": 1354}
  ]
}
```
Sorted by `trip_id`. `n_obs` = rows in the `.track` file (no route information).

## 4. `POST /predict`
Request:
```json
{"trip_id": "00000007"}
```
Response (200):
```json
{
  "trip_id": "00000007",
  "model": "lightgbm_tuned",
  "n_obs": 1354,
  "gps": [[43.388762, 77.122052], [43.388809, 77.122188]],
  "hmm_route": [
    [[43.388700, 77.122000], [43.388900, 77.122300], [43.389100, 77.122600]]
  ],
  "classifier_route": [
    [[43.388700, 77.122000], [43.388900, 77.122300]],
    [[43.390100, 77.123000], [43.390300, 77.123400]]
  ],
  "metrics": {
    "top1_accuracy_hmm": 0.9123,
    "top1_accuracy_classifier": 0.8890,
    "route_continuity_hmm": 0.9950,
    "route_continuity_classifier": 0.9810,
    "n_obs_evaluated": 1354
  },
  "timings_ms": {"load": 20, "model": 230, "hmm": 410, "evaluation": 35, "total": 700}
}
```
Field meanings:
- `gps`: one `[lat, lon]` per GPS observation, in `obs_idx` order.
- `hmm_route`: predicted route after HMM/Viterbi as a **list of polylines**. Consecutive predicted arcs
  that are connected are drawn as one polyline; a new polyline starts after a break (GPS gap / chain restart).
  Each polyline = node coordinates of the arcs in driving order, duplicates at joins removed.
- `classifier_route`: same format, from the classifier-only top-1 arc per observation.
- `metrics`: computed **server-side** with `src/common/labels.align_route` + `src/common/metrics`
  (`top1_table`, `route_continuity`, MAX_HOPS = 5). Values in [0, 1].
- `timings_ms`: integer milliseconds per stage; `total` = whole request.

Errors:
| Case | HTTP | Body |
|---|---|---|
| unknown trip | 404 | `{"detail": "trip 00000999 not found"}` |
| malformed request | 422 | FastAPI default validation body |
| pipeline failure | 500 | `{"detail": "<short error message>"}` |

The UI should wait up to **120 s** for `/predict`.

## 5. Server log line (shown to evaluators in the Laptop 1 terminal)
One line per request, plus FastAPI/uvicorn's own access log:
```
2026-11-18 13:45:10 | POST /predict | trip=00000007 | client=192.168.43.21 | n_obs=1354 | load=20ms model=230ms hmm=410ms eval=35ms total=700ms | top1_hmm=0.9123 top1_clf=0.8890 | cont_hmm=0.9950
2026-11-18 13:44:58 | GET /trips | client=192.168.43.21 | n_trips=10 | data_dir=student_test
```
Also written to `results/logs/b_api_requests.log` (append).

## 6. Server-side pipeline for `/predict` (order matters)
1. `trip = load_trip(trip_id, DATA_DIR, with_route=False)`: prediction never sees the route.
2. `scored = score_trip(trip)`; `hmm_in = to_hmm_input(scored)` (`src/ml/a_inference.py`).
3. Classifier-only = highest-probability arc per observation; HMM = Viterbi on `hmm_in`.
4. Build `gps`, `hmm_route`, `classifier_route`.
5. **Only now**: `ref = align_route(load_trip(trip_id, DATA_DIR))` and compute `metrics`.
   If the folder has no `.route` file, return `metrics` with `null` values.
