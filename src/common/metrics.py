"""Shared evaluation metrics (owner: Person A; used by src/ml AND src/hmm).

Keeping one implementation guarantees the classifier comparison table and the
HMM comparison table use identical definitions.

* Classification metrics are per candidate row at threshold 0.5.
* Observation-level Top-1 accuracy: for every GPS observation take the
  candidate with the highest probability and compare its arc_id with the
  reference arc. Observations with NO candidate (or whose reference arc is not
  among the candidates) count as wrong, so Top-1 <= 1 - candidate miss rate.
* Route continuity: fraction of consecutive observation pairs (i, i+1) whose
  predicted arcs are the same arc or are connected in the DIRECTED road network
  within MAX_HOPS arcs (u->v followed by v->w is 1 hop). At 1 Hz a car can cross
  several short arcs between two fixes (median arc length ~20 m), so MAX_HOPS > 1.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from .io import load_arcs, trip_files

MAX_HOPS = 5


# ---------------------------------------------------------------- connectivity
@lru_cache(maxsize=128)
def _adjacency(trip_id: str, partition: str):
    """Directed arc table + CSR index of outgoing arcs per node."""
    arcs = load_arcs(trip_files(trip_id, partition)["arcs"])
    frm = arcs["from_node"].to_numpy()
    to = arcs["to_node"].to_numpy()
    order = np.argsort(frm, kind="stable")
    n_nodes = int(max(frm.max(), to.max())) + 1
    offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(offsets, frm + 1, 1)
    offsets = np.cumsum(offsets)
    return frm, to, order, offsets


def arcs_connected(adj, a: int, b: int, max_hops: int = MAX_HOPS) -> bool:
    """True if arc b equals arc a or can be reached from a within max_hops arcs."""
    if a == b:
        return True
    frm, to, order, offsets = adj
    target_start = frm[b]
    frontier = {int(to[a])}
    seen = set(frontier)
    for _ in range(max_hops):
        if target_start in frontier:
            return True
        nxt = set()
        for node in frontier:
            for arc in order[offsets[node]:offsets[node + 1]]:
                w = int(to[arc])
                if w not in seen:
                    seen.add(w)
                    nxt.add(w)
        if not nxt:
            return False
        frontier = nxt
    return target_start in frontier


def route_continuity(arc_seq, trip_id: str, partition: str, max_hops: int = MAX_HOPS) -> float:
    """Fraction of consecutive predicted arcs that are connected (see module doc)."""
    seq = np.asarray(arc_seq)
    if len(seq) < 2:
        return 1.0
    adj = _adjacency(trip_id, partition)
    ok = [arcs_connected(adj, int(a), int(b), max_hops) for a, b in zip(seq[:-1], seq[1:])]
    return float(np.mean(ok))


# ---------------------------------------------------------------- predictions
def top1_table(cand: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Per-observation best candidate vs reference.

    cand : rows with trip_id, obs_idx, arc_id, prob
    obs  : one row per observation with trip_id, obs_idx, ref_arc
    Returns obs with columns pred_arc (-1 if no candidate) and correct.
    """
    best = (cand.sort_values(["trip_id", "obs_idx", "prob"], ascending=[True, True, False])
                .drop_duplicates(["trip_id", "obs_idx"])[["trip_id", "obs_idx", "arc_id"]]
                .rename(columns={"arc_id": "pred_arc"}))
    out = obs.merge(best, on=["trip_id", "obs_idx"], how="left")
    out["pred_arc"] = out["pred_arc"].fillna(-1).astype(np.int64)
    out["correct"] = (out["pred_arc"] == out["ref_arc"]).astype(int)
    return out


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.clip(np.asarray(prob, float), 1e-7, 1 - 1e-7)
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1_pos": f1_score(y_true, pred, zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, prob, labels=[0, 1]),
        "confusion_matrix": cm.tolist(),   # [[TN, FP], [FN, TP]]
    }


def evaluate(cand: pd.DataFrame, obs: pd.DataFrame, partition: str,
             with_continuity: bool = True) -> dict:
    """All metrics for a set of trips. cand needs label + prob, obs needs ref_arc, ref_in_cands."""
    m = classification_metrics(cand["label"], cand["prob"])
    t1 = top1_table(cand, obs)
    m["top1_accuracy"] = float(t1["correct"].mean())
    covered = obs["ref_in_cands"].to_numpy() == 1
    m["top1_accuracy_covered"] = float(t1.loc[covered, "correct"].mean()) if covered.any() else float("nan")
    m["candidate_miss_rate"] = float(1 - covered.mean())
    m["n_obs"] = int(len(obs))
    m["n_rows"] = int(len(cand))
    if with_continuity:
        conts, ref_conts = [], []
        for tid, d in t1.sort_values(["trip_id", "obs_idx"]).groupby("trip_id"):
            seq = d.loc[d["pred_arc"] >= 0, "pred_arc"].to_numpy()
            conts.append(route_continuity(seq, tid, partition))
        m["route_continuity"] = float(np.mean(conts))
    return m
