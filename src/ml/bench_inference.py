"""Fair inference-time benchmark: several saved models scored on the SAME data, same process,
same machine state, repeated N times (median reported). Owner: Tejas.

Harness timings are single measurements taken whenever each model was trained, so they are
affected by whatever else the laptop was doing. This script removes that noise.
Run:  python -m src.ml.bench_inference lightgbm lightgbm_tuned --repeats 7
Output: results/a_bench_inference.csv
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from src.common.config import CACHE_DIR, RESULTS_DIR
from src.ml.harness import load_partition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    cand, _ = load_partition("student_test", train_filter=False)
    rows = []
    for name in args.models:
        b = joblib.load(CACHE_DIR / "models" / f"{name}.joblib")
        X = cand[b["feature_cols"]].to_numpy(np.float32)
        times = []
        for _ in range(args.repeats + 1):          # first call = warm-up, discarded
            t0 = time.perf_counter()
            Xs = b["scaler"].transform(X) if b.get("scaler") is not None else X
            b["model"].predict_proba(Xs)
            times.append(time.perf_counter() - t0)
        t = np.array(times[1:])
        rows.append({"model": name, "rows": len(X), "repeats": args.repeats,
                     "median_seconds": float(np.median(t)), "min_seconds": float(t.min()),
                     "ms_per_1k_rows": float(np.median(t) / len(X) * 1e6)})
        print(f"{name:>16}: median {np.median(t):.3f}s (min {t.min():.3f}s) over {args.repeats} runs, "
              f"{rows[-1]['ms_per_1k_rows']:.2f} ms / 1k rows")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "a_bench_inference.csv", index=False)


if __name__ == "__main__":
    main()
