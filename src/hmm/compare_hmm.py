"""Classifier-only vs classifier + HMM/Viterbi on the 10 student_test trips (report section 5).

The classifier is the exported final model (artifacts/model.joblib via score_trip). The decoder
parameters come from results/b_hmm_best.json (tuned on TRAIN out-of-fold probabilities) or, if
that file does not exist yet, the defaults. Same metric code as the model table
(src/common/metrics.py).

Run:  python -m src.hmm.compare_hmm
Output: results/b_hmm_comparison.csv, b_hmm_pertrip.csv (incl. fixed/worse counts), b_hmm_worse_obs.csv
"""
from __future__ import annotations

import time

import pandas as pd

from src.common.config import CACHE_DIR, RESULTS_DIR
from src.common.io import list_trips, load_trip
from src.hmm.evaluate import classifier_only, per_trip_top1, score
from src.hmm.runlog import tee_output
from src.hmm.viterbi import BEST_PARAMS_FILE, decode_scored, load_best_params, params_dict
from src.ml.a_inference import model_info, score_trip

PART = "student_test"


def main():
    p = load_best_params()
    print(f"[hmm] classifier: {model_info()}")
    print(f"[hmm] decoder params ({'tuned' if BEST_PARAMS_FILE.exists() else 'DEFAULT, not tuned'}): "
          f"{params_dict(p)}", flush=True)
    obs = pd.read_parquet(CACHE_DIR / f"obs_{PART}.parquet")[["trip_id", "obs_idx", "ref_arc"]]

    cands, hmm_preds, t_clf, t_hmm = [], [], 0.0, 0.0
    for tid in list_trips(PART):
        trip = load_trip(tid, PART)
        t0 = time.time()
        scored = score_trip(trip)                      # candidates -> features -> model (no route used)
        t_clf += time.time() - t0
        t0 = time.time()
        pred = decode_scored(scored, trip, p)
        t_hmm += time.time() - t0
        cands.append(scored)
        hmm_preds.append(pred)
        print(f"[hmm] {tid}: obs={len(trip.track):>5} classifier {t_clf:.1f}s cumulative, "
              f"decode {t_hmm:.1f}s cumulative", flush=True)

    cand = pd.concat(cands, ignore_index=True)
    clf_m, clf_t1 = score(classifier_only(cand), obs, PART)
    hmm_m, hmm_t1 = score(pd.concat(hmm_preds, ignore_index=True), obs, PART)

    table = pd.DataFrame({"classifier_only": clf_m, "classifier_plus_hmm": hmm_m}).T
    table["delta_vs_classifier"] = None
    table.loc["classifier_plus_hmm", "delta_vs_classifier"] = hmm_m["top1_accuracy"] - clf_m["top1_accuracy"]
    table.insert(0, "scoring_seconds", [t_clf, t_clf + t_hmm])
    table.to_csv(RESULTS_DIR / "b_hmm_comparison.csv")

    # where did the HMM change the answer?
    m = clf_t1[["trip_id", "obs_idx", "correct"]].merge(
        hmm_t1[["trip_id", "obs_idx", "correct"]], on=["trip_id", "obs_idx"], suffixes=("_clf", "_hmm"))
    m["fixed"] = (m.correct_clf == 0) & (m.correct_hmm == 1)
    m["worse"] = (m.correct_clf == 1) & (m.correct_hmm == 0)
    fixed, broke = int(m["fixed"].sum()), int(m["worse"].sum())

    pt = pd.DataFrame({"classifier_top1": per_trip_top1(clf_t1), "hmm_top1": per_trip_top1(hmm_t1)})
    pt["delta"] = pt["hmm_top1"] - pt["classifier_top1"]
    pt["n_obs"] = m.groupby("trip_id").size()
    pt["fixed_by_hmm"] = m.groupby("trip_id")["fixed"].sum().astype(int)
    pt["worse_by_hmm"] = m.groupby("trip_id")["worse"].sum().astype(int)
    pt.to_csv(RESULTS_DIR / "b_hmm_pertrip.csv")
    # observations the HMM got wrong but the classifier had right (for the "did not help" analysis / map)
    m.loc[m["worse"], ["trip_id", "obs_idx"]].to_csv(RESULTS_DIR / "b_hmm_worse_obs.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== student_test: classifier-only vs classifier + HMM/Viterbi ===")
    print(table.drop(columns=["scoring_seconds"]).round(4).to_string())
    print(f"\nobservations fixed by HMM: {fixed}   made worse by HMM: {broke}   (n_obs={len(m)})")
    print("\n=== per-trip Top-1 ===")
    print(pt.round(4).to_string())
    print("\n[hmm] saved results/b_hmm_comparison.csv, b_hmm_pertrip.csv, b_hmm_worse_obs.csv")


if __name__ == "__main__":
    tee_output(RESULTS_DIR / "logs" / "b_hmm_compare.log")   # terminal + UTF-8 log
    main()
