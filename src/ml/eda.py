"""Exploratory data analysis (Dataset Exploration PDF, section 4). Owner: Tejas.

Partitions analysed: train (80 trips) and student_test (10 trips). Every table
has a `partition` column and every plot title states the partition(s).
The evaluator-test partition is never touched.

Outputs
  results/a_eda_inventory.csv      per trip: file sizes, row counts, missing files
  results/a_eda_dtypes.csv         column names / dtypes per file type
  results/a_eda_quality.csv        per trip: missing / invalid / duplicate values
  results/a_eda_time.csv           per trip: timestamp ordering and gaps
  results/a_eda_extent.csv         per trip: location, bbox size, length, speed
  results/a_eda_network.csv        per trip: nodes, arcs, components, isolated nodes...
  results/a_eda_route.csv          per trip: route length, connectivity, GPS-route distance
  results/a_eda_anomalies.csv      trips flagged by simple rules
  results/a_eda_summary.json       partition-level summary numbers
  results/a_eda_*.png              plots
Run:  python -m src.ml.eda      (needs results/a_eda_kr_obs.csv from src.ml.choose_kr)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.common.config import RESULTS_DIR
from src.common.geo import arc_geometry, haversine, track_xy
from src.common.io import list_trips, load_trip, trip_files

R = RESULTS_DIR
PARTS = ("train", "student_test")


def trip_stats(trip_id: str, part: str) -> dict:
    files = trip_files(trip_id, part)
    trip = load_trip(trip_id, part)
    tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    s = {"partition": part, "trip_id": trip_id}

    # ---- inventory
    for ext, p in files.items():
        s[f"has_{ext}"] = p.exists()
        s[f"size_kb_{ext}"] = round(p.stat().st_size / 1024, 1) if p.exists() else np.nan
    s.update(n_obs=len(tr), n_nodes=len(nodes), n_arcs=len(arcs),
             n_route=len(route) if route is not None else 0)

    # ---- quality (missing / invalid)
    s["track_nan"] = int(tr[["lon", "lat", "t"]].isna().sum().sum())
    s["nodes_nan"] = int(nodes[["lon", "lat"]].isna().sum().sum())
    s["track_invalid_range"] = int(((tr.lon.abs() > 180) | (tr.lat.abs() > 90)).sum())
    s["nodes_invalid_range"] = int(((nodes.lon.abs() > 180) | (nodes.lat.abs() > 90)).sum())
    s["track_zero_coord"] = int(((tr.lon == 0) & (tr.lat == 0)).sum())
    s["track_dup_consecutive_pos"] = int(((tr.lon.diff() == 0) & (tr.lat.diff() == 0)).sum())
    s["arcs_bad_node_ref"] = int(((arcs[["from_node", "to_node"]] < 0) |
                                  (arcs[["from_node", "to_node"]] >= len(nodes))).any(axis=1).sum())
    s["route_bad_arc_ref"] = int(((route.arc_id < 0) | (route.arc_id >= len(arcs))).sum()) if route is not None else 0

    # ---- time
    dt = np.diff(tr.t.to_numpy())
    s.update(t_monotonic=bool((dt > 0).all()), n_dt_nonpositive=int((dt <= 0).sum()),
             frac_dt_1s=float((dt == 1).mean()), dt_median=float(np.median(dt)),
             dt_max=float(dt.max()), n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt30s=int((dt > 30).sum()),
             duration_min=float((tr.t.iloc[-1] - tr.t.iloc[0]) / 60))

    # ---- extent / motion (metric, per-trip projection)
    x, y = track_xy(trip)
    step = np.hypot(np.diff(x), np.diff(y))
    hav = haversine(tr.lon.to_numpy()[:-1], tr.lat.to_numpy()[:-1], tr.lon.to_numpy()[1:], tr.lat.to_numpy()[1:])
    speed = step / np.where(dt > 0, dt, np.nan)
    s.update(center_lon=float(tr.lon.median()), center_lat=float(tr.lat.median()),
             bbox_w_km=float((x.max() - x.min()) / 1000), bbox_h_km=float((y.max() - y.min()) / 1000),
             track_len_km=float(step.sum() / 1000),
             proj_vs_haversine_max_rel_err=float(np.nanmax(np.abs(step - hav) / np.maximum(hav, 1.0))),
             speed_median_mps=float(np.nanmedian(speed)), speed_p99_mps=float(np.nanpercentile(speed, 99)),
             n_jumps_gt60mps=int(np.nansum(speed > 60)))

    # ---- network
    g = arc_geometry(trip)
    u, v = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    n = len(nodes)
    A = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    ncomp, comp = connected_components(A, directed=True, connection="weak")
    used = np.zeros(n, bool); used[u] = True; used[v] = True
    sizes = np.bincount(comp[used]) if used.any() else np.array([0])
    pairs = pd.Series(list(zip(u, v)))
    pair_set = set(zip(u, v))
    s.update(n_isolated_nodes=int((~used).sum()),
             n_components_excl_isolated=int((sizes > 0).sum()),
             largest_component_frac=float(sizes.max() / used.sum()) if used.any() else np.nan,
             n_self_loops=int((u == v).sum()), n_duplicate_arcs=int(pairs.duplicated().sum()),
             n_zero_length_arcs=int((g.length_m < 0.01).sum()),
             frac_two_way=float(np.mean([(b, a) in pair_set for a, b in zip(u, v)])),
             arc_len_median_m=float(g.length_m.median()), arc_len_max_m=float(g.length_m.max()),
             network_bbox_km=float(max(g.ax.max() - g.ax.min(), g.ay.max() - g.ay.min()) / 1000))

    # ---- route structure
    if route is not None and len(route):
        ra = route.arc_id.to_numpy()
        conn = arcs.to_node.to_numpy()[ra[:-1]] == arcs.from_node.to_numpy()[ra[1:]]
        rlen = g.length_m.to_numpy()[ra].sum() / 1000
        s.update(route_frac_consecutive_connected=float(conn.mean()) if len(conn) else np.nan,
                 route_n_unique=int(len(np.unique(ra))), route_len_km=float(rlen),
                 route_len_over_track_len=float(rlen / max(s["track_len_km"], 1e-9)),
                 route_arcs_per_obs=float(len(ra) / len(tr)))
    return s


def _plots(df: pd.DataFrame, kr: pd.DataFrame, grid: pd.DataFrame):
    col = {"train": "tab:blue", "student_test": "tab:orange"}

    # 1. GPS coordinate distribution: trip locations on a world scatter
    plt.figure(figsize=(9, 4.5))
    for p, d in df.groupby("partition"):
        plt.scatter(d.center_lon, d.center_lat, s=25, c=col[p], label=p, alpha=.8)
    plt.xlabel("longitude"); plt.ylabel("latitude"); plt.xlim(-180, 180); plt.ylim(-60, 75); plt.grid(alpha=.3)
    plt.title("Trip locations (median GPS position) - train + student_test"); plt.legend(); plt.tight_layout()
    plt.savefig(R / "a_eda_trip_locations.png", dpi=120); plt.close()

    # 2. observations per trip / duration / length
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for p, d in df.groupby("partition"):
        ax[0].hist(d.n_obs, bins=30, alpha=.6, color=col[p], label=p)
        ax[1].hist(d.duration_min, bins=30, alpha=.6, color=col[p], label=p)
        ax[2].hist(d.track_len_km, bins=30, alpha=.6, color=col[p], label=p)
    for a, t in zip(ax, ["GPS observations per trip", "trip duration (min)", "track length (km)"]):
        a.set_title(t); a.legend()
    fig.suptitle("Trajectory size - train + student_test"); plt.tight_layout()
    plt.savefig(R / "a_eda_trip_sizes.png", dpi=120); plt.close()

    # 3. sampling interval distribution (all steps, train)
    dts = []
    for tid in list_trips("train"):
        dts.append(np.diff(load_trip(tid, "train", with_route=False).track.t.to_numpy()))
    dts = np.concatenate(dts)
    vals, cnts = np.unique(np.clip(dts, None, 31), return_counts=True)
    plt.figure(figsize=(7, 3.6))
    plt.bar(vals, cnts); plt.yscale("log"); plt.xlabel("time step (s), clipped at 31"); plt.ylabel("count (log)")
    plt.title(f"GPS sampling interval - train ({(dts == 1).mean():.1%} of steps exactly 1 s)")
    plt.tight_layout(); plt.savefig(R / "a_eda_time_gaps.png", dpi=120); plt.close()

    # 4. missing / invalid value analysis
    qcols = ["track_nan", "nodes_nan", "track_invalid_range", "nodes_invalid_range", "track_zero_coord",
             "track_dup_consecutive_pos", "arcs_bad_node_ref", "route_bad_arc_ref", "n_dt_nonpositive"]
    q = df.groupby("partition")[qcols].sum().T
    q.plot.barh(figsize=(8, 4.2))
    plt.title("Missing / invalid value counts (summed over trips)"); plt.xlabel("count")
    plt.tight_layout(); plt.savefig(R / "a_eda_missing_invalid.png", dpi=120); plt.close()

    # 5. road-network size and connectivity
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p, d in df.groupby("partition"):
        ax[0].scatter(d.n_nodes, d.n_arcs, s=18, c=col[p], label=p)
        ax[1].scatter(d.n_arcs, d.largest_component_frac, s=18, c=col[p], label=p)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs")
    ax[0].set_title("Network size per trip"); ax[0].legend()
    ax[1].set_xscale("log"); ax[1].set_xlabel("arcs"); ax[1].set_ylabel("largest weak component / used nodes")
    ax[1].set_title("Network connectivity"); ax[1].legend()
    plt.tight_layout(); plt.savefig(R / "a_eda_network.png", dpi=120); plt.close()

    # 6. GPS noise: distance from GPS point to its reference arc
    plt.figure(figsize=(7, 3.6))
    for p, d in kr.groupby("partition"):
        plt.hist(np.clip(d.ref_dist_m, 0, 100), bins=100, alpha=.6, density=True, color=col[p], label=p)
    plt.xlabel("distance GPS point -> reference arc (m), clipped at 100"); plt.ylabel("density")
    plt.title("GPS-to-reference-route distance (labels from monotone alignment)")
    plt.legend(); plt.tight_layout(); plt.savefig(R / "a_eda_gps_noise.png", dpi=120); plt.close()

    # 7. candidate miss rate vs K and R (train, anomalous trips excluded)
    plt.figure(figsize=(7, 4))
    gt = grid[grid.partition == "train"]
    for r_, d in gt.groupby("R"):
        plt.plot(d.K, d.miss_rate, marker="o", label=f"R={r_} m")
    plt.xlabel("K (max candidates)"); plt.ylabel("candidate miss rate"); plt.yscale("log")
    plt.title("Candidate miss rate vs K, R - train (excl. anomalous trips)"); plt.legend(fontsize=8)
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(R / "a_eda_candidate_miss.png", dpi=120); plt.close()

    # 8. speed distribution
    plt.figure(figsize=(7, 3.6))
    for p, d in df.groupby("partition"):
        plt.hist(d.speed_median_mps * 3.6, bins=25, alpha=.6, color=col[p], label=p)
    plt.xlabel("median speed per trip (km/h)"); plt.ylabel("trips"); plt.legend()
    plt.title("Typical vehicle speed - train + student_test"); plt.tight_layout()
    plt.savefig(R / "a_eda_speed.png", dpi=120); plt.close()


def _trajectory_plots(trip_ids):
    """GPS trajectory + nearby road network + reference route, in local metres (train)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, tid in zip(axes.ravel(), trip_ids):
        trip = load_trip(tid, "train")
        g = arc_geometry(trip)
        x, y = track_xy(trip)
        pad = 200
        m = ((g[["ax", "bx"]].max(axis=1) > x.min() - pad) & (g[["ax", "bx"]].min(axis=1) < x.max() + pad) &
             (g[["ay", "by"]].max(axis=1) > y.min() - pad) & (g[["ay", "by"]].min(axis=1) < y.max() + pad))
        gg = g[m]
        ax.plot(np.c_[gg.ax, gg.bx, np.full(len(gg), np.nan)].ravel(),
                np.c_[gg.ay, gg.by, np.full(len(gg), np.nan)].ravel(), color="0.8", lw=.6)
        rg = g.loc[trip.route.arc_id.to_numpy()]
        ax.plot(np.c_[rg.ax, rg.bx, np.full(len(rg), np.nan)].ravel(),
                np.c_[rg.ay, rg.by, np.full(len(rg), np.nan)].ravel(), color="green", lw=2.2)
        ax.scatter(x, y, s=2, c="blue")
        ax.set_aspect("equal"); ax.set_title(f"train {tid} ({len(x)} obs)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Trajectories (blue GPS), road network (grey), reference route (green) - train, local metres")
    plt.tight_layout(); plt.savefig(R / "a_eda_trajectories.png", dpi=120); plt.close()


def main():
    rows = []
    for part in PARTS:
        rows += Parallel(n_jobs=-1)(delayed(trip_stats)(t, part) for t in list_trips(part))
    df = pd.DataFrame(rows)

    kr = pd.read_csv(R / "a_eda_kr_obs.csv", dtype={"trip_id": str})
    rd = kr.groupby(["partition", "trip_id"]).ref_dist_m.agg(
        ref_dist_median_m="median", ref_dist_p90_m=lambda s: s.quantile(.9),
        frac_obs_gt50m_from_route=lambda s: (s > 50).mean()).reset_index()
    df = df.merge(rd, on=["partition", "trip_id"], how="left")

    ids = ["partition", "trip_id"]
    df[ids + [c for c in df if c.startswith(("has_", "size_kb_"))] + ["n_obs", "n_nodes", "n_arcs", "n_route"]] \
        .to_csv(R / "a_eda_inventory.csv", index=False)
    df[ids + ["track_nan", "nodes_nan", "track_invalid_range", "nodes_invalid_range", "track_zero_coord",
              "track_dup_consecutive_pos", "arcs_bad_node_ref", "route_bad_arc_ref"]] \
        .to_csv(R / "a_eda_quality.csv", index=False)
    df[ids + ["t_monotonic", "n_dt_nonpositive", "frac_dt_1s", "dt_median", "dt_max", "n_gaps_gt5s",
              "n_gaps_gt30s", "duration_min"]].to_csv(R / "a_eda_time.csv", index=False)
    df[ids + ["center_lon", "center_lat", "bbox_w_km", "bbox_h_km", "track_len_km", "speed_median_mps",
              "speed_p99_mps", "n_jumps_gt60mps", "proj_vs_haversine_max_rel_err"]] \
        .to_csv(R / "a_eda_extent.csv", index=False)
    df[ids + ["n_nodes", "n_arcs", "n_isolated_nodes", "n_components_excl_isolated", "largest_component_frac",
              "n_self_loops", "n_duplicate_arcs", "n_zero_length_arcs", "frac_two_way", "arc_len_median_m",
              "arc_len_max_m", "network_bbox_km"]].to_csv(R / "a_eda_network.csv", index=False)
    df[ids + ["n_route", "route_n_unique", "route_frac_consecutive_connected", "route_len_km",
              "route_len_over_track_len", "route_arcs_per_obs", "ref_dist_median_m", "ref_dist_p90_m",
              "frac_obs_gt50m_from_route"]].to_csv(R / "a_eda_route.csv", index=False)

    # dtypes as actually parsed
    t0 = load_trip(list_trips("train")[0], "train")
    pd.DataFrame([{"file": f, "column": c, "dtype": str(d)} for f, frame in
                  (("track", t0.track), ("nodes", t0.nodes), ("arcs", t0.arcs), ("route", t0.route))
                  for c, d in frame.dtypes.items()]).to_csv(R / "a_eda_dtypes.csv", index=False)

    # anomaly rules
    an = []
    for _, r in df.iterrows():
        why = []
        if r.ref_dist_median_m > 30: why.append(f"route far from GPS (median {r.ref_dist_median_m:.0f} m)")
        if r.n_gaps_gt30s > 0: why.append(f"{r.n_gaps_gt30s} gaps > 30 s")
        if r.n_jumps_gt60mps > 0: why.append(f"{r.n_jumps_gt60mps} jumps > 60 m/s")
        if not r.t_monotonic: why.append("non-increasing timestamps")
        if r.route_frac_consecutive_connected < 1: why.append("route not fully connected")
        if r.n_obs > 10000: why.append(f"very long trip ({r.n_obs} obs)")
        if why: an.append({"partition": r.partition, "trip_id": r.trip_id, "issues": "; ".join(why)})
    pd.DataFrame(an).to_csv(R / "a_eda_anomalies.csv", index=False)

    summ = {}
    for p, d in df.groupby("partition"):
        summ[p] = {
            "trips": len(d), "obs_total": int(d.n_obs.sum()), "obs_per_trip_median": float(d.n_obs.median()),
            "obs_per_trip_min": int(d.n_obs.min()), "obs_per_trip_max": int(d.n_obs.max()),
            "missing_files": int((~d[[c for c in d if c.startswith("has_")]]).sum().sum()),
            "nan_values": int(d.track_nan.sum() + d.nodes_nan.sum()),
            "frac_dt_1s_mean": float(d.frac_dt_1s.mean()), "trips_non_monotonic_time": int((~d.t_monotonic).sum()),
            "trips_with_gap_gt30s": int((d.n_gaps_gt30s > 0).sum()),
            "nodes_median": float(d.n_nodes.median()), "arcs_median": float(d.n_arcs.median()),
            "arcs_max": int(d.n_arcs.max()), "isolated_nodes_total": int(d.n_isolated_nodes.sum()),
            "components_median": float(d.n_components_excl_isolated.median()),
            "frac_two_way_mean": float(d.frac_two_way.mean()),
            "route_fully_connected_trips": int((d.route_frac_consecutive_connected == 1).sum()),
            "route_arcs_per_obs_median": float(d.route_arcs_per_obs.median()),
            "ref_dist_median_m_median": float(d.ref_dist_median_m.median()),
            "lon_range": [float(d.center_lon.min()), float(d.center_lon.max())],
            "lat_range": [float(d.center_lat.min()), float(d.center_lat.max())],
            "proj_vs_haversine_max_rel_err": float(d.proj_vs_haversine_max_rel_err.max()),
        }
    (R / "a_eda_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(pd.DataFrame(an).to_string())

    _plots(df, kr, pd.read_csv(R / "a_eda_kr_grid.csv"))
    _trajectory_plots(["00000001", "00000002", "00000019", "00000046", "00000054", "00000033"])


if __name__ == "__main__":
    main()
