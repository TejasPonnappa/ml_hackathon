# PathSnap: map-matching GPS trips to a road network

Pipeline: GPS point -> candidate road arcs -> classifier scores each (point, arc) pair -> HMM/Viterbi picks a connected
route. A FastAPI backend (Laptop 1) serves the pipeline; a Streamlit UI (Laptop 2) shows the evaluator's chosen trip.

## 1. Install
Python 3.12 or 3.13. From the repo root:

    python -m venv .venv
    .venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
    pip install -r requirements.txt

`requirements.txt` is pinned so every laptop (and the saved model) uses the same versions.

## 2. Data path
The raw course data is not in git. Point `PATHSNAP_DATA_ROOT` at the folder that contains `train/` and `student_test/`
(default: `<repo>/data_raw/Student folder`):

    $env:PATHSNAP_DATA_ROOT = "D:\data\Student folder"          # PowerShell
    export PATHSNAP_DATA_ROOT="/data/Student folder"            # bash

Build the candidate/feature tables once (written to the git-ignored `data_cache/`), then train a model:

    python -m src.ml.build_dataset
    python -m src.ml.train_logreg        # also train_mlp, train_random_forest, train_xgboost, train_lightgbm

## 3. API on Laptop 1 (model, HMM, road network, server-side evaluation)
    $env:DATA_DIR = "<folder with the trips the API serves>"   # see section 5
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

`--host 0.0.0.0` makes it reachable from Laptop 2. Allow the port through the Windows firewall (admin PowerShell, once):

    netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000

Find Laptop 1's address with `ipconfig` (IPv4 of the Wi-Fi/LAN adapter); both laptops must be on the same network.
Test from Laptop 2: `curl http://<laptop1-ip>:8000/health`. Keep this terminal visible: the request log is shown to the
evaluator.

Contract (agreed): `GET /health`, `GET /trips`, `POST /predict {"trip_id": "00000007"}`. Coordinates are `[lat, lon]`.
Errors: HTTP 404 `{"detail": "trip ... not found"}`, HTTP 500 `{"detail": "..."}`. Ground truth never leaves the API.

## 4. UI on Laptop 2
    $env:API_URL = "http://<laptop1-ip>:8000"
    streamlit run src/ui/app.py --server.port 8501

Open http://localhost:8501. The API address can also be edited in the sidebar. The UI reads no data files: trip list,
routes and accuracy all come from the API. It shows "API connected" (green) or a red error, a trip dropdown, a
"Run map-matching" button (120 s timeout), metric cards (HMM Top-1, classifier-only Top-1, route continuity, GPS
points, total time), a timings table and a Folium map (blue GPS, red HMM route, orange classifier-only route, layer
control, legend).

Development without the API: `USE_FAKE=1` loads the synthetic fixture `src/ui/dev/fake_api.json`
(`$env:USE_FAKE = "1"`), or run the mock API `uvicorn src.ui.dev.mock_api:app --port 8000` to exercise the real HTTP
path. **Remove the `USE_FAKE` switch (and `src/ui/dev/`) before the demo.**

## 5. Switching to the evaluator trips at demo time
The API reads its trips from `DATA_DIR` (named in `/health` as `data_dir`). During development it points at
`student_test`; at the demo set it to the folder holding the 10 reserved evaluator trips, restart the API, and the UI
lists them automatically (nothing in the UI changes). The evaluator-test partition is never used for training, model
selection or tuning.

## 6. Report map and visualisations
    python -m src.viz.eda_folium 00000001 train                         # EDA map (results/a_eda_folium_*.html)
    python -m src.viz.folium_map 00000007 --partition student_test --api http://localhost:8000
                                                                        # results/b_folium_00000007.html

`folium_map.py` is offline (reads the data); without `--api` it builds a classifier-only route locally from a saved
model (`PATHSNAP_MODEL=<bundle>.joblib`) and omits the HMM layer.

## 7. Repository layout
    src/common/   config, data loading, geometry, candidate generation, reference-route alignment, metrics
    src/ml/       dataset build, features, shared harness, one train_<model>.py per model, inference entry point
    src/hmm/      HMM / Viterbi route reconstruction
    src/api/      FastAPI backend
    src/ui/       Streamlit evaluator UI (app.py); dev/ = fixture + mock API for development only
    src/viz/      Folium maps (EDA map, report map)
    results/      experiment outputs (a_cv_<model>.json, plots, per-trip CSVs, Folium HTML)
    report/       report section drafts
    artifacts/    exported model files
    data_cache/   regenerable candidate/feature tables and local models (git-ignored)
