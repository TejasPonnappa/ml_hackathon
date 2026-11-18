"""Export the selected model to artifacts/ for the API. Owner: Tejas.

Copies the harness bundle data_cache/models/<name>.joblib (model + scaler + feature column
order + candidate K/R) to artifacts/model.joblib, saves the native LightGBM text model, and
writes artifacts/model_config.json describing what was exported and why.
src/ml/a_inference.py loads artifacts/model.joblib first, so the API needs no code change.
Run:  python -m src.ml.export_model lightgbm_tuned
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime

import joblib

from src.common.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR


def main(name: str):
    src = CACHE_DIR / "models" / f"{name}.joblib"
    dst = ARTIFACTS_DIR / "model.joblib"
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(src, dst)
    b = joblib.load(dst)
    if hasattr(b["model"], "booster_"):
        b["model"].booster_.save_model(str(ARTIFACTS_DIR / "model_lightgbm.txt"))
    res = json.loads((RESULTS_DIR / f"a_cv_{name}.json").read_text())
    cfg = {
        "exported_model": name, "family": res["family"], "hyperparameters": res["config"],
        "feature_cols": b["feature_cols"], "candidate_K": b["K"], "candidate_R_m": b["R"],
        "scaler": None if b.get("scaler") is None else type(b["scaler"]).__name__,
        "selection_rule": "highest CV observation-level Top-1; within one fold-std, lowest CV log loss "
                          "(see src/ml/compare_models.py and results/a_model_comparison.csv)",
        "cv": {k: res["cv"][k] for k in ("macro_f1_fold_mean", "macro_f1_fold_std", "top1_accuracy", "log_loss")},
        "student_test": {k: res["student_test"][k] for k in ("top1_accuracy", "macro_f1", "log_loss",
                                                             "route_continuity")},
        "sha256_model_joblib": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    (ARTIFACTS_DIR / "model_config.json").write_text(json.dumps(cfg, indent=2, default=float))
    print(f"exported {name} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lightgbm_tuned")
