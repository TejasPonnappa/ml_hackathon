"""Dev-only mock of the PathSnap API (same endpoints/format as the agreed contract), serving src/ui/dev/fake_api.json.
Lets the UI's real HTTP code path (incl. 404 / 500 handling) be tested before Tejasvi's API is ready.
Run:  uvicorn src.ui.dev.mock_api:app --port 8000"""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

FX = json.loads((Path(__file__).parent / "fake_api.json").read_text())
app = FastAPI()


class Req(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return FX["health"]


@app.get("/trips")
def trips():
    # two extra IDs let the UI's error handling be tested: ...404 is not served, ...999 raises a 500
    extra = [{"trip_id": "00000404", "n_obs": 1}, {"trip_id": "00000999", "n_obs": 1}]
    return {**FX["trips"], "trips": FX["trips"]["trips"] + extra}


@app.post("/predict")
def predict(req: Req):
    if req.trip_id == "00000999":
        raise HTTPException(500, "simulated server error")
    if req.trip_id not in FX["predict"]:
        raise HTTPException(404, f"trip {req.trip_id} not found")
    return FX["predict"][req.trip_id]
