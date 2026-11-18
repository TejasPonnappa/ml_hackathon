"""PathSnap data processing: loading, projection, candidate generation, reference alignment, features.

Problem framing (Model 1 = Logistic Regression):
  For every GPS observation we generate a small set of candidate road arcs (nearest arcs within a radius).
  A binary classifier scores each (observation, candidate arc) pair: 1 = candidate is the reference arc.
  The predicted arc of an observation is the candidate with the highest score (observation-level Top-1).
"""
import glob
import os
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), "student_folder", "Student folder")

# ---- candidate generation settings (justified in EDA via ref_coverage) ----
RADIUS_M = 100.0
MAX_CANDS = 10
SAMPLE_STEP_M = 20.0

FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio_min",
    "frac_along", "arc_len_m", "cos_heading", "abs_cos_heading", "heading_valid",
    "speed_mps", "dt_s", "n_cands", "d_prev_m", "d_next_m", "out_deg_to", "in_deg_from",
]


def list_trips(partition):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(DATA, partition, "*")))


def load_trip(partition, trip_id):
    d = os.path.join(DATA, partition, trip_id)
    p = os.path.join(d, trip_id)
    track = np.loadtxt(p + ".track", ndmin=2)
    nodes = np.loadtxt(p + ".nodes", ndmin=2)
    arcs = np.loadtxt(p + ".arcs", dtype=np.int64, ndmin=2)
    route = np.loadtxt(p + ".route", dtype=np.int64, ndmin=1)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    sx, sy = 111320.0 * np.cos(np.radians(lat0)), 110574.0  # equirectangular, fine at city scale
    proj = lambda ll: np.c_[(ll[:, 0] - lon0) * sx, (ll[:, 1] - lat0) * sy]
    return dict(
        id=trip_id, partition=partition, track_ll=track[:, :2], t=track[:, 2], nodes_ll=nodes,
        P=proj(nodes), O=proj(track[:, :2]), arcs=arcs, route=route,
    )


def seg_dist(p, A, B):
    """distance from points p to segments AB (all arrays (n,2)); returns (dist, frac)."""
    ab = B - A
    L2 = (ab ** 2).sum(1)
    f = np.clip(((p - A) * ab).sum(1) / np.maximum(L2, 1e-9), 0, 1)
    return np.linalg.norm(p - (A + f[:, None] * ab), axis=1), f


def _arc_index(trip):
    P, arcs = trip["P"], trip["arcs"]
    A, B = P[arcs[:, 0]], P[arcs[:, 1]]
    L = np.linalg.norm(B - A, axis=1)
    n = np.maximum(1, np.ceil(L / SAMPLE_STEP_M).astype(int) + 1)
    aid = np.repeat(np.arange(len(arcs)), n)
    off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    s = off / np.repeat(np.maximum(n - 1, 1), n)
    pts = A[aid] + s[:, None] * (B - A)[aid]
    return cKDTree(pts), aid, A, B, L


def _near(tree, aid, O, r):
    out = []
    for lst in tree.query_ball_point(O, r + SAMPLE_STEP_M / 2):
        out.append(np.unique(aid[lst]) if lst else np.empty(0, dtype=np.int64))
    return out


def align_reference(trip, lam=5.0, cap=150.0):
    """Derive one reference arc per GPS observation.
    The .route file only lists arcs in travel order, so each observation is assigned to a route arc with a
    monotone (non-decreasing) dynamic-programming alignment minimising point-to-arc distance (capped at `cap` m).
    Skipping route arcs costs `lam` metres per arc (GPS gaps / short arcs). Returns route-position and arc id per obs."""
    P, arcs, route, O = trip["P"], trip["arcs"], trip["route"], trip["O"]
    A, B = P[arcs[route, 0]], P[arcs[route, 1]]
    T, J = len(O), len(route)
    ar = np.arange(J)
    dp = np.empty(J)
    back = np.zeros((T, J), dtype=np.int32)
    d0, _ = _dist_matrix_row(O[0], A, B)
    dp = np.minimum(d0, cap) + lam * ar
    for t in range(1, T):
        d, _ = _dist_matrix_row(O[t], A, B)
        c = np.minimum(d, cap)
        stay = dp
        adv = np.r_[np.inf, dp[:-1]]
        # skip >=2 arcs forward:  min_{k<=j-2} dp[k] + lam*(j-1-k)
        g = dp - lam * ar
        cm = np.minimum.accumulate(g)
        ci = np.maximum.accumulate(np.where(g == cm, ar, 0))
        skip = np.r_[np.inf, np.inf, cm[:-2] + lam * (ar[2:] - 1)]
        skip_i = np.r_[0, 0, ci[:-2]]
        stack = np.vstack([stay, adv, skip])
        k = stack.argmin(0)
        best = stack[k, ar]
        back[t] = np.where(k == 0, ar, np.where(k == 1, ar - 1, skip_i))
        dp = c + best
    pos = np.empty(T, dtype=np.int64)
    pos[-1] = dp.argmin()
    for t in range(T - 1, 0, -1):
        pos[t - 1] = back[t, pos[t]]
    return pos, route[pos]


def _dist_matrix_row(p, A, B):
    return seg_dist(np.broadcast_to(p, A.shape), A, B)


def ref_coverage(trip, radius=200.0, kmax=30):
    """For EDA: rank & distance of the reference arc among arcs ordered by distance (rank=-1 if outside radius)."""
    tree, aid, A, B, L = _arc_index(trip)
    _, ref = align_reference(trip)
    near = _near(tree, aid, trip["O"], radius)
    rank = np.full(len(ref), -1)
    dist = np.zeros(len(ref))
    for i, a in enumerate(near):
        pa = np.broadcast_to(trip["O"][i], (len(a), 2))
        d, _ = seg_dist(pa, A[a], B[a]) if len(a) else (np.empty(0), None)
        o = np.argsort(d)
        a, d = a[o], d[o]
        w = np.where(a == ref[i])[0]
        if len(w):
            rank[i] = w[0]
            dist[i] = d[w[0]]
    return rank, dist


def build_candidates(trip, radius=RADIUS_M, kmax=MAX_CANDS, with_labels=True):
    """Return a dict of candidate-level feature arrays for a trip (one row per (observation, candidate arc))."""
    tree, aid, A, B, L = _arc_index(trip)
    O, t, arcs = trip["O"], trip["t"], trip["arcs"]
    T = len(O)
    out_deg = np.bincount(arcs[:, 0], minlength=len(trip["P"]))
    in_deg = np.bincount(arcs[:, 1], minlength=len(trip["P"]))
    near = _near(tree, aid, O, radius)

    prev_i, next_i = np.maximum(np.arange(T) - 1, 0), np.minimum(np.arange(T) + 1, T - 1)
    v = O[next_i] - O[prev_i]
    dtw = np.maximum(t[next_i] - t[prev_i], 1e-3)
    vnorm = np.linalg.norm(v, axis=1)
    speed = vnorm / dtw
    hvalid = (vnorm > 3.0).astype(float)
    dt_s = np.r_[1.0, np.diff(t)]

    obs, cand, dist, frac = [], [], [], []
    for i, a in enumerate(near):
        if len(a) == 0:
            continue
        d, f = seg_dist(np.broadcast_to(O[i], (len(a), 2)), A[a], B[a])
        keep = d <= radius
        a, d, f = a[keep], d[keep], f[keep]
        o = np.argsort(d, kind="stable")[:kmax]
        obs.append(np.full(len(o), i)); cand.append(a[o]); dist.append(d[o]); frac.append(f[o])
    if not obs:
        return None
    obs, cand, dist, frac = map(np.concatenate, (obs, cand, dist, frac))
    n_c = np.bincount(obs, minlength=T)
    starts = np.r_[0, np.cumsum(n_c)[:-1]]
    rank = np.arange(len(obs)) - starts[obs]
    dmin = dist[starts[obs]]
    ab = B[cand] - A[cand]
    abn = np.maximum(np.linalg.norm(ab, axis=1), 1e-9)
    cosh = (v[obs] * ab).sum(1) / (np.maximum(vnorm[obs], 1e-9) * abn) * hvalid[obs]
    dprev, _ = seg_dist(O[prev_i][obs], A[cand], B[cand])
    dnext, _ = seg_dist(O[next_i][obs], A[cand], B[cand])
    X = np.c_[
        dist, rank, dist - dmin, dist / (dmin + 1.0), frac, L[cand], cosh, np.abs(cosh), hvalid[obs],
        speed[obs], dt_s[obs], n_c[obs], dprev, dnext, out_deg[arcs[cand, 1]], in_deg[arcs[cand, 0]],
    ].astype(np.float64)
    res = dict(trip=trip["id"], obs=obs, cand=cand, X=X, n_obs=T)
    if with_labels:
        _, ref = align_reference(trip)
        res["ref_obs"] = ref
        res["y"] = (cand == ref[obs]).astype(np.int8)
    return res


def route_continuity(pred_arcs, arcs):
    """Share of consecutive observation pairs whose predicted arcs are identical or directly connected."""
    a, b = pred_arcs[:-1], pred_arcs[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return ok.mean()
