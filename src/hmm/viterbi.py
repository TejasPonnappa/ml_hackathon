"""HMM + Viterbi route reconstruction (Part B).

Hidden state
    For GPS observation i the hidden state is WHICH candidate arc the vehicle is on.
    State space at step i = the candidate arcs from src/common/candidates.py (K <= 10,
    within R = 50 m), so the candidate -> road-segment correspondence is preserved:
    every state IS an arc_id.

Emission (how well a state explains the observation)
    log P(obs | arc) = alpha * log(p)      p = classifier probability that the arc is the true one.
    The classifier already used distance, heading, speed ... (19 features), so the HMM
    reuses its output instead of recomputing a Gaussian distance model.

Transition (is moving from arc a at step i to arc b at step i+1 plausible?)
    a == b                                   cost 0            (still on the same road)
    b reachable from a in d further arcs     cost beta*(1+d)   (d = 0..max_hops-1, directed network)
    b NOT reachable within max_hops          cost `invalid`    (finite, so decoding never fails)
    b is the reverse of a (U-turn)           extra cost `uturn`
    log P(b | a) = -cost.   (Unnormalised: Viterbi only needs relative scores.)

Viterbi recurrence
    delta_1(s)   = emission_1(s)
    delta_i(b)   = emission_i(b) + max_a [ delta_{i-1}(a) + log P(b | a) ]
    psi_i(b)     = argmax_a [...]        (back-pointer)
    path         = backtrack from argmax_s delta_n(s)

Edge cases
    * Observation with no candidate within R: it is dropped from the chain (prediction -1) and
      the missing obs_idx is treated like a GPS gap (see next point).
    * GPS gap (time gap > gap_s, or missing observation indices, between consecutive decoded
      observations): the connectivity constraint is switched off for that step (all transitions
      cost 0), i.e. the chain restarts, because a vehicle can legitimately travel far during a gap.
    * Disconnected candidates at consecutive steps: allowed but charged `invalid`, so the path
      only takes such a jump when the classifier evidence clearly outweighs it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import RESULTS_DIR

BEST_PARAMS_FILE = RESULTS_DIR / "b_hmm_best.json"
NEG = -1e9


@dataclass(frozen=True)
class DecoderParams:
    alpha: float = 1.0       # emission exponent (weight of classifier evidence)
    beta: float = 1.0        # cost per hop of road-network distance
    invalid: float = 6.0     # cost of a jump to a not-connected arc
    uturn: float = 1.0       # extra cost of reversing onto the opposite arc


def load_best_params() -> DecoderParams:
    """Tuned decoder parameters (from tune_decoder.py) or the defaults."""
    if BEST_PARAMS_FILE.exists():
        return DecoderParams(**json.loads(BEST_PARAMS_FILE.read_text())["params"])
    return DecoderParams()


def params_dict(p: DecoderParams) -> dict:
    return asdict(p)


# ----------------------------------------------------------------------------- road graph
class ArcGraph:
    """Directed arc graph with cached 'how many arcs until I can reach node X' lookups."""

    def __init__(self, arcs: pd.DataFrame, max_hops: int = 5):
        self.frm = arcs["from_node"].to_numpy()
        self.to = arcs["to_node"].to_numpy()
        self.max_hops = max_hops
        order = np.argsort(self.frm, kind="stable")
        n_nodes = int(max(self.frm.max(), self.to.max())) + 1
        off = np.zeros(n_nodes + 1, dtype=np.int64)
        np.add.at(off, self.frm + 1, 1)
        self._off = np.cumsum(off)
        self._out_to = self.to[order]            # heads of outgoing arcs, grouped by from-node
        self._cache: dict[int, dict[int, int]] = {}

    def reach(self, a: int) -> dict[int, int]:
        """{node: d} nodes reachable after leaving arc a; d = extra arcs travelled (0 = a's head)."""
        r = self._cache.get(a)
        if r is None:
            start = int(self.to[a])
            r, frontier = {start: 0}, [start]
            for d in range(1, self.max_hops + 1):
                nxt = []
                for n in frontier:
                    for w in self._out_to[self._off[n]:self._off[n + 1]]:
                        w = int(w)
                        if w not in r:
                            r[w] = d
                            nxt.append(w)
                if not nxt:
                    break
                frontier = nxt
            self._cache[a] = r
        return r


# ----------------------------------------------------------------------------- one trip
@dataclass
class TripSequence:
    """Parameter-independent arrays for one trip (built once, decoded many times)."""
    trip_id: str
    obs_idx: np.ndarray      # (n,)    observations that have candidates
    arcs: np.ndarray         # (n, K)  candidate arc ids, -1 padding
    prob: np.ndarray         # (n, K)  classifier probabilities, 0 padding
    code: np.ndarray         # (n-1, K, K) int16: 0 stay, d+1 reachable, 99 unreachable
    uturn: np.ndarray        # (n-1, K, K) bool
    gap: np.ndarray          # (n-1,)  bool: connectivity switched off for this step
    n_total_obs: int
    n_gaps: int


def build_sequence(scored: pd.DataFrame, trip, graph: ArcGraph, gap_s: float = 30.0) -> TripSequence:
    """scored needs obs_idx, arc_id, prob (candidates of ONE trip). trip supplies timestamps."""
    s = scored.sort_values(["obs_idx", "prob"], ascending=[True, False]).reset_index(drop=True)
    pos = s.groupby("obs_idx").cumcount().to_numpy()
    obs_ids, inv = np.unique(s["obs_idx"].to_numpy(), return_inverse=True)
    n, K = len(obs_ids), int(pos.max()) + 1
    arcs = np.full((n, K), -1, dtype=np.int64)
    prob = np.zeros((n, K))
    arcs[inv, pos] = s["arc_id"].to_numpy()
    prob[inv, pos] = s["prob"].to_numpy()

    t = trip.track.set_index("obs_idx")["t"].reindex(obs_ids).to_numpy()
    # a step is a "gap" (chain restarts, connectivity not enforced) if the clock jumped by more than
    # gap_s OR observation indices are missing in between (no candidate within R, or removed by the
    # training filter): the vehicle moved an unknown distance, so adjacency would be unreliable.
    gap = (np.diff(t) > gap_s) | (np.diff(obs_ids) > 1)
    code = np.full((max(n - 1, 0), K, K), 99, dtype=np.int16)
    uturn = np.zeros((max(n - 1, 0), K, K), dtype=bool)
    frm, to = graph.frm, graph.to
    for i in range(n - 1):
        if gap[i]:
            continue
        for ja in range(K):
            a = arcs[i, ja]
            if a < 0:
                break
            reach = graph.reach(int(a))
            for jb in range(K):
                b = arcs[i + 1, jb]
                if b < 0:
                    break
                if a == b:
                    code[i, ja, jb] = 0
                    continue
                d = reach.get(int(frm[b]))
                if d is not None:
                    code[i, ja, jb] = d + 1
                uturn[i, ja, jb] = (frm[b] == to[a]) and (to[b] == frm[a])
    return TripSequence(trip.trip_id, obs_ids, arcs, prob, code, uturn, gap,
                        int(len(trip.track)), int(gap.sum()))


def viterbi(seq: TripSequence, p: DecoderParams) -> np.ndarray:
    """Most likely arc for each observation in seq.obs_idx (array of arc ids)."""
    n, K = seq.arcs.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    em = p.alpha * np.log(np.clip(seq.prob, 1e-6, 1.0))
    em[seq.arcs < 0] = NEG
    # transition log-probs for every step at once: -cost
    cost = np.where(seq.code == 0, 0.0,
                    np.where(seq.code < 99, p.beta * seq.code, p.invalid)) + p.uturn * seq.uturn
    trans = -cost
    trans[seq.gap] = 0.0                       # gap: connectivity not enforced
    delta = em[0].copy()
    back = np.zeros((n, K), dtype=np.int16)
    for i in range(1, n):
        sc = delta[:, None] + trans[i - 1]
        back[i] = sc.argmax(0)
        delta = em[i] + sc.max(0)
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return seq.arcs[np.arange(n), path]


def decode_scored(scored: pd.DataFrame, trip, p: DecoderParams | None = None,
                  max_hops: int = 5, gap_s: float = 30.0) -> pd.DataFrame:
    """DataFrame version: candidates with prob -> DataFrame[trip_id, obs_idx, pred_arc]
    for EVERY observation (pred_arc = -1 where there were no candidates). Used by the
    offline comparison / tuning scripts."""
    p = p or load_best_params()
    seq = build_sequence(scored, trip, ArcGraph(trip.arcs, max_hops), gap_s)
    out = pd.DataFrame({"trip_id": trip.trip_id, "obs_idx": trip.track["obs_idx"].to_numpy(),
                        "pred_arc": -1})
    out.loc[out["obs_idx"].isin(seq.obs_idx), "pred_arc"] = viterbi(seq, p)
    return out


def decode_trip(trip, hmm_in: dict[int, list[tuple[int, float]]],
                params: DecoderParams | None = None) -> dict[int, int]:
    """API ENTRY POINT.

    trip   : io.Trip loaded with_route=False (needs track timestamps and the arc table)
    hmm_in : to_hmm_input(score_trip(trip)) = {obs_idx: [(arc_id, prob), ...]}
    returns: {obs_idx: predicted arc_id}. Observations with no candidate are omitted.
    Uses the tuned decoder parameters (results/b_hmm_best.json) unless `params` is given.
    """
    if not hmm_in:
        return {}
    rows = [(o, a, p) for o, cands in hmm_in.items() for a, p in cands]
    scored = pd.DataFrame(rows, columns=["obs_idx", "arc_id", "prob"])
    pred = decode_scored(scored.assign(trip_id=trip.trip_id), trip, params)
    pred = pred[pred["pred_arc"] >= 0]
    return {int(o): int(a) for o, a in zip(pred["obs_idx"], pred["pred_arc"])}
