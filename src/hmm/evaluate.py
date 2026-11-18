"""Scoring helpers for the classifier-only vs classifier+HMM comparison.

Top-1 and route continuity reuse src/common/metrics.py (the same definitions as every
classifier in the model table). Sequence-level route overlap is added here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.io import load_trip
from src.common.metrics import route_continuity, top1_table


def classifier_only(cand: pd.DataFrame) -> pd.DataFrame:
    """Independent per-observation prediction: the highest-probability candidate."""
    best = (cand.sort_values(["trip_id", "obs_idx", "prob"], ascending=[True, True, False])
                .drop_duplicates(["trip_id", "obs_idx"])[["trip_id", "obs_idx", "arc_id"]]
                .rename(columns={"arc_id": "pred_arc"}))
    return best


def score(pred: pd.DataFrame, obs: pd.DataFrame, partition: str, with_route_sets: bool = True):
    """Returns (metrics dict, per-observation table).
    pred: trip_id, obs_idx, pred_arc (missing observations count as wrong).
    obs : trip_id, obs_idx, ref_arc (reference arc per observation)."""
    cand_like = pred.rename(columns={"pred_arc": "arc_id"}).assign(prob=1.0)
    t1 = top1_table(cand_like, obs)
    out = {"top1_accuracy": float(t1["correct"].mean()), "n_obs": int(len(t1))}
    conts, p_, r_, f_ = [], [], [], []
    for tid, d in t1.sort_values(["trip_id", "obs_idx"]).groupby("trip_id"):
        seq = d.loc[d["pred_arc"] >= 0, "pred_arc"].to_numpy()
        conts.append(route_continuity(seq, tid, partition))
        if with_route_sets:
            ref = set(load_trip(tid, partition).route["arc_id"].tolist())
            pr = set(seq.tolist())
            inter = len(pr & ref)
            prec, rec = inter / max(len(pr), 1), inter / max(len(ref), 1)
            p_.append(prec); r_.append(rec)
            f_.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    out["route_continuity"] = float(np.mean(conts))
    if with_route_sets:
        out.update(route_precision=float(np.mean(p_)), route_recall=float(np.mean(r_)),
                   route_f1=float(np.mean(f_)))
    return out, t1


def per_trip_top1(t1: pd.DataFrame) -> pd.Series:
    return t1.groupby("trip_id")["correct"].mean()
