# PathSnap demo runbook

**Laptop 1 = TEJASVI → API** (model + HMM + data + request logs shown to evaluators)
**Laptop 2 = TEJAS → UI** (evaluator picks a trip, sees the route and accuracy)

Run the steps in order. Each step is labelled with who runs it.

---

## A. Once, before the demo day (both laptops)

**[TEJASVI] A1 — update code and environment**
```powershell
cd C:\Tejasvi\PESU\3rd_year\ML\Hackathon
git checkout main
git pull origin main
.venv\Scripts\activate
pip install -r requirements.txt
```

**[TEJASVI] A2 — firewall rule for port 8000** (PowerShell **as Administrator**, only once)
```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
```

**[TEJAS] A3 — update code and environment**
```powershell
cd "C:\Users\TEJAS PONNAPPA\Desktop\Ml hackathon"
git checkout main
git pull origin main
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## B. Part A terminal demo (TEJAS, any time the evaluators ask "show us the training")

Run in the activated venv on Tejas's laptop, in this order:

| # | Command | Time | What to say |
|---|---|---|---|
| B1 | `python -m src.ml.build_dataset` | ~15 s | candidates (10 nearest roads within 50 m) + labels + 19 features; miss rate 1.66 % train / 0.26 % test |
| B2 | `python -m src.ml.train_logreg` | ~30 s | baseline model; 5-fold CV grouped by trip, then student_test |
| B3 | `python -m src.ml.train_lightgbm` | ~2 min | selected model, same protocol; numbers equal the committed results (seed 42) |
| B4 | `python -m src.ml.compare_models` | ~2 s | 5-model table + which model is selected and why (CV Top-1, tie-break log loss) |
| B5 | `python -m src.ml.bayesopt_lightgbm --trials 3 --no-refit` | ~3 min | Bayesian optimisation demo (full run was 20 trials: `results\a_bayesopt_trials.csv`) |

Show logs of the long runs instead of re-running: `results\logs\a_random_forest.log`, `a_xgboost.log`.
Afterwards restore the committed result files: `git checkout -- results/`

---

## C. Two-laptop live demo

**[TEJASVI] C1 — turn on the hotspot**
Settings → Network & Internet → **Mobile hotspot** → On. Tell Tejas the network name and password.

**[TEJAS] C2 — connect to Tejasvi's hotspot** (Wi-Fi icon → his network), then check:
```powershell
ipconfig        # Wi-Fi IPv4 must be 192.168.137.x
```

**[TEJASVI] C3 — start the API** (keep this window large and visible: evaluators watch these logs)
```powershell
.venv\Scripts\activate
$env:DATA_DIR = "student_test"          # at the real evaluation: see C7
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Wait for: `STARTUP | model=lightgbm_tuned ... | n_trips=10 | HMM available=True` and the `HMM params` line.

**[TEJAS] C4 — test the connection**
```powershell
Test-NetConnection 192.168.137.1 -Port 8000      # TcpTestSucceeded : True
```
Optional: open `http://192.168.137.1:8000/docs` in the browser (a log line appears on Tejasvi's screen).

**[TEJAS] C5 — start the UI**
```powershell
.venv\Scripts\activate
$env:API_URL = "http://192.168.137.1:8000"
streamlit run src/ui/app.py --server.port 8501 --client.toolbarMode viewer
```
Browser opens `http://localhost:8501` → green "API connected – model: lightgbm_tuned", 10 trips listed.

**[TEJAS] C6 — run a trip**
Pick a trip → **Run map-matching** → map (blue GPS, red HMM route, orange classifier-only) + HMM Top-1,
classifier-only Top-1, route continuity, timings.
**[TEJASVI]** at the same moment points to his log line:
`POST /predict | trip=... | client=192.168.137.145 | ... | top1_hmm=... top1_clf=...`
Good trip to show the HMM effect: **00000077** (0.747 → 0.899).

**[TEJASVI] C7 — switch to the evaluator trips (real evaluation only)**
Unzip the evaluator trips into their **own** folder (not into `data_raw\`). Then in the API window: `Ctrl + C`, and
```powershell
$env:DATA_DIR = "C:\path\to\evaluator_trips"     # the folder that contains the 10 trip folders
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
**[TEJAS]** in the UI sidebar click **Reconnect / reload trips** → the 10 evaluator trips appear. The evaluator
picks any trip; repeat C6. Nothing else changes. Never set `USE_FAKE`.

**[BOTH] C8 — stop**
`Ctrl + C` in the API window (Tejasvi) and in the Streamlit window (Tejas).

---

## D. If something fails

| Symptom | Who | Fix |
|---|---|---|
| UI shows red "Cannot reach the API" | TEJAS | check `Test-NetConnection 192.168.137.1 -Port 8000`; check the address in the UI sidebar |
| `TcpTestSucceeded : False` | TEJASVI | API window must say `Uvicorn running on http://0.0.0.0:8000`; `http://localhost:8000/docs` must work on his laptop; `Get-NetFirewallRule -DisplayName "PathSnap*"` must exist |
| Tejas's IP is not 192.168.137.x | TEJAS | connected to the wrong Wi-Fi; join Tejasvi's hotspot |
| `HMM available=False` at startup | TEJASVI | `git pull origin main` (src/hmm missing) |
| `/trips` shows 0 trips | TEJASVI | `DATA_DIR` points to the wrong folder; it must contain the trip folders directly |
| Map background is blank | — | no internet on the hotspot; routes and accuracy still work |
| Port 8000 / 8501 busy | the owner | `Get-NetTCPConnection -LocalPort 8000 -State Listen` → `Stop-Process -Id <OwningProcess> -Force` |

## Rules to remember during the demo
- Ground truth stays on Tejasvi's laptop; the UI only receives aggregate accuracy.
- The evaluator trips are used only in step C7, never for training or tuning.
- Every number we quote is in `results/` on GitHub.
