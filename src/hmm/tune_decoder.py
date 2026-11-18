"""Tune the HMM decoder parameters (alpha, beta, invalid, uturn).

Data used: the 78 TRAIN trips only (same filter as the classifiers: anomalous trips and
observations whose reference arc is > R m away are dropped). Probabilities are the
OUT-OF-FOLD predictions saved by the harness (data_cache/preds_<model>_oof.parquet), i.e. each
trip is scored by a model that never saw it, so the decoder is not tuned on over-confident
in-sample probabilities. student_test is NOT touched here.

Objective: maximise observation-level Top-1 accuracy of the decoded route (pooled over trips).

Run:  python -m src.hmm.tune_decoder                 (uses the lightgbm_tuned OOF predictions,
                                                       i.e. the final model)
      python -m src.hmm.tune_decoder --model xgboost  (any other harness model)
Output: results/b_hmm_tuning.csv, results/b_hmm_best.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from src.common.config import CACHE_DIR, RESULTS_DIR
from src.common.io import load_trip
from src.hmm.runlog import tee_output
from src.hmm.viterbi import ArcGraph, DecoderParams, build_sequence, params_dict, viterbi
from src.ml.harness import load_partition

# Grid history (selection always on train OOF Top-1 only):
#   grid 1: alpha 0.5-2, beta 0.5-2, invalid 3-12, uturn 0-2          best 0.9327, on the edge for all 4 params
#   grid 2: alpha 0.25-1, beta 2-8, invalid 12-48, uturn 2-8         best 0.9529, still on the edge for 3 params
#   grid 3 (final, log-spaced beyond the edges, beta kept around its best value 4):
GRID = dict(alpha=[0.05, 0.1, 0.25], beta=[3.0, 4.0, 6.0], invalid=[48.0, 100.0, 200.0], uturn=[8.0, 16.0, 32.0])
# Full range searched over all three grids (used for the "is the optimum on the edge?" check).
SEARCHED = dict(alpha=(0.05, 2.0), beta=(0.5, 8.0), invalid=(3.0, 200.0), uturn=(0.0, 32.0))


def main(model: str, max_trips: int | None):
    t_start = time.time()
    oof_path = CACHE_DIR / f"preds_{model}_oof.parquet"
    if not oof_path.exists():
        raise SystemExit(f"{oof_path} not found. It is produced by the harness when {model} is trained "
                         f"(local, git-ignored); copy it from the teammate who trained it.")
    oof = pd.read_parquet(oof_path)[["trip_id", "obs_idx", "arc_id", "prob"]]
    _, obs = load_partition("train", train_filter=True)          # filtered observations + ref_arc
    trips = sorted(obs["trip_id"].unique())[:max_trips]
    print(f"[tune] model={model} trips={len(trips)} obs={int(obs.trip_id.isin(trips).sum()):,}", flush=True)

    seqs, refs = [], []
    for k, tid in enumerate(trips, 1):
        trip = load_trip(tid, "train", with_route=False)
        c = oof[oof["trip_id"] == tid]
        seqs.append(build_sequence(c, trip, ArcGraph(trip.arcs)))
        o = obs[obs["trip_id"] == tid].set_index("obs_idx")["ref_arc"]
        refs.append(o)
        if k % 10 == 0 or k == len(trips):
            print(f"[tune] built sequences {k}/{len(trips)} ({time.time() - t_start:.0f}s)", flush=True)

    n_obs = sum(len(r) for r in refs)

    def pooled_top1(decoder):
        right = 0
        for s, ref in zip(seqs, refs):
            pred = pd.Series(decoder(s), index=s.obs_idx)
            right += int((pred.reindex(ref.index) == ref).sum())
        return right / n_obs

    base = pooled_top1(lambda s: s.arcs[np.arange(len(s.arcs)), s.prob.argmax(1)])
    print(f"[tune] classifier-only Top-1 on these trips (OOF) = {base:.4f}", flush=True)

    rows = []
    combos = list(itertools.product(*GRID.values()))
    for i, vals in enumerate(combos, 1):
        p = DecoderParams(**dict(zip(GRID, vals)))
        t0 = time.time()
        top1 = pooled_top1(lambda s: viterbi(s, p))
        rows.append({**params_dict(p), "top1": top1, "gain_vs_classifier": top1 - base})
        print(f"[tune] {i:>2}/{len(combos)} alpha={p.alpha} beta={p.beta} invalid={p.invalid} "
              f"uturn={p.uturn} -> top1={top1:.4f} ({top1 - base:+.4f}) [{time.time() - t0:.1f}s]", flush=True)

    # ties (identical Top-1) go to the LEAST extreme setting: smaller uturn / invalid / beta, larger alpha
    res = pd.DataFrame(rows).sort_values(["top1", "uturn", "invalid", "beta", "alpha"],
                                         ascending=[False, True, True, True, False])
    res.to_csv(RESULTS_DIR / "b_hmm_tuning.csv", index=False)
    best = res.iloc[0]
    best_p = DecoderParams(**{k: float(best[k]) for k in GRID})
    edge = {k: bool(best[k] in (min(v), max(v))) for k, v in GRID.items()}           # this grid
    edge_all = {k: bool(best[k] in v) for k, v in SEARCHED.items()}                   # all grids
    (RESULTS_DIR / "b_hmm_best.json").write_text(json.dumps({
        "params": params_dict(best_p), "tuning_top1": float(best["top1"]),
        "classifier_only_top1_same_trips": base, "tuned_on": f"train OOF probabilities of {model}",
        "n_trips": len(trips), "n_obs": n_obs, "grid": GRID, "searched_range_all_grids": SEARCHED,
        "best_on_edge_of_this_grid": edge, "best_on_edge_of_searched_range": edge_all, "seconds": time.time() - t_start}, indent=2))
    print(f"[tune] best on edge of this grid: {edge} | of the full searched range: {edge_all}")
    print(f"[tune] top-5 Top-1: "
          f"{', '.join(f'{x:.4f}' for x in res['top1'].head(5))}")
    print(f"[tune] BEST {params_dict(best_p)} top1={best['top1']:.4f} (classifier-only {base:.4f}) "
          f"-> results/b_hmm_best.json  total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    tee_output(RESULTS_DIR / "logs" / "b_hmm_tune.log")   # terminal + UTF-8 log
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm_tuned")
    ap.add_argument("--max-trips", type=int, default=None, help="subset of trips for a quick test")
    a = ap.parse_args()
    main(a.model, a.max_trips)
