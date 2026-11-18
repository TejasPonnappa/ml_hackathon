"""Model 1 - Logistic Regression candidate classifier for PathSnap.
Run:  python src/run_model1_logreg.py      (from pathsnap_lr/)   -> writes everything to results/
Uses ONLY the 80 training trips (fit + group CV) and the 10 student-testing trips (development comparison).
The evaluator-testing partition is never touched.
"""
import json, os, sys, time, platform
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathsnap_data as D

SEED = 42
np.random.seed(SEED)
OUT = os.path.join(D.ROOT, "results")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "cache_candidates.joblib")

SQ_COLS = [D.FEATURES.index(c) for c in ("dist_m", "dist_minus_min", "d_prev_m", "d_next_m")]


def expand(X):
    """Linear model needs explicit nonlinearity: a Gaussian emission is quadratic in distance, so add scaled squares."""
    return np.c_[X, (X[:, SQ_COLS] / 100.0) ** 2]


FEAT_NAMES = D.FEATURES + [D.FEATURES[i] + "^2" for i in SQ_COLS]


def make_model():
    return make_pipeline(
        FunctionTransformer(expand), StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight=None, random_state=SEED),
    )


# ------------------------------------------------------------------ data
def process(part, tid):
    trip = D.load_trip(part, tid)
    c = D.build_candidates(trip)
    t = trip["t"]; dt = np.diff(t)
    ll = trip["track_ll"]
    inv = dict(partition=part, trip=tid, n_gps=len(t), n_nodes=len(trip["P"]), n_arcs=len(trip["arcs"]),
               n_route_arcs=len(trip["route"]), pct_dt_not_1s=100 * float((dt != 1).mean()), max_gap_s=float(dt.max()),
               nonmonotonic_ts=int((dt <= 0).sum()), nan_rows=int(np.isnan(trip["track_ll"]).any(1).sum()),
               lon_min=ll[:, 0].min(), lon_max=ll[:, 0].max(), lat_min=ll[:, 1].min(), lat_max=ll[:, 1].max(),
               median_arc_len_m=float(np.median(np.linalg.norm(trip["P"][trip["arcs"][:, 1]] - trip["P"][trip["arcs"][:, 0]], axis=1))),
               isolated_nodes=int(len(trip["P"]) - len(np.unique(trip["arcs"]))))
    cov = None
    if part == "train":
        rank, dist = D.ref_coverage(trip)
        cov = (rank, dist)
    _, ref = D.align_reference(trip)
    inv["pct_route_continuity_ref"] = 100 * D.route_continuity(ref, trip["arcs"])
    return c, inv, cov


def load_all():
    if os.path.exists(CACHE):
        return joblib.load(CACHE)
    jobs = [(p, t) for p in ("train", "student_test") for t in D.list_trips(p)]
    res = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(process)(p, t) for p, t in jobs)
    obj = dict(jobs=jobs, res=res)
    joblib.dump(obj, CACHE)
    return obj


# ------------------------------------------------------------------ evaluation
def trip_scores(model, c, trips):
    t0 = time.perf_counter()
    p = model.predict_proba(c["X"])[:, 1]
    return p, time.perf_counter() - t0


def evaluate(model, caches, trips):
    """caches: list of candidate dicts; trips: dict id -> loaded trip (for arcs / route). Returns metrics dict."""
    ys, ps, infer = [], [], 0.0
    per_trip, obs_total, obs_hit_total, obs_correct, cont_ok, cont_n = [], 0, 0, 0, 0, 0
    pred_all = {}
    for c in caches:
        p, dt = trip_scores(model, c, trips); infer += dt
        ys.append(c["y"]); ps.append(p)
        tr = trips[c["trip"]]
        n = c["n_obs"]
        # arg-max candidate per observation
        order = np.lexsort((-p, c["obs"]))
        first = np.r_[True, np.diff(c["obs"][order]) != 0]
        sel = order[first]
        pred = np.full(n, -1, dtype=np.int64)
        pred[c["obs"][sel]] = c["cand"][sel]
        ref = c["ref_obs"]
        hit = np.zeros(n, bool)
        pos = np.where(c["y"] == 1)[0]
        hit[c["obs"][pos]] = True
        correct = pred == ref
        k = pred >= 0
        cont = D.route_continuity(pred[k], tr["arcs"]) if k.sum() > 1 else np.nan
        # sequence level: de-duplicated predicted route vs the supplied reference route (arc sets)
        pr = pred[k]; pr = pr[np.r_[True, pr[1:] != pr[:-1]]] if len(pr) else pr
        ps_, rs_ = set(pr.tolist()), set(tr["route"].tolist())
        tp = len(ps_ & rs_)
        sp, sr = tp / max(len(ps_), 1), tp / max(len(rs_), 1)
        sf = 2 * sp * sr / (sp + sr) if sp + sr > 0 else 0.0
        per_trip.append(dict(trip=c["trip"], n_obs=n, top1_acc=correct.mean(), candidate_miss_rate=1 - hit.mean(),
                             top1_given_hit=correct[hit].mean() if hit.any() else np.nan, route_continuity=cont,
                             seq_precision=sp, seq_recall=sr, seq_f1=sf, n_pred_route_arcs=len(pr), n_ref_route_arcs=len(rs_)))
        obs_total += n; obs_hit_total += hit.sum(); obs_correct += correct.sum()
        if k.sum() > 1:
            a, b = pred[k][:-1], pred[k][1:]
            cont_ok += ((a == b) | (tr["arcs"][a, 1] == tr["arcs"][b, 0])).sum(); cont_n += len(a)
        pred_all[c["trip"]] = pred
    y, p = np.concatenate(ys), np.concatenate(ps)
    yh = (p >= 0.5).astype(int)
    pt = pd.DataFrame(per_trip)
    m = dict(
        n_candidate_rows=int(len(y)), positive_rate=float(y.mean()),
        accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
        recall=recall_score(y, yh, zero_division=0), f1_positive=f1_score(y, yh, zero_division=0),
        macro_f1=f1_score(y, yh, average="macro"), log_loss=log_loss(y, p, labels=[0, 1]),
        confusion_matrix=confusion_matrix(y, yh, labels=[0, 1]).tolist(),
        top1_acc_all_obs=obs_correct / obs_total, top1_acc_given_candidate_hit=obs_correct / max(obs_hit_total, 1),
        candidate_miss_rate=1 - obs_hit_total / obs_total,
        route_continuity=cont_ok / max(cont_n, 1),
        seq_precision_mean=pt.seq_precision.mean(), seq_recall_mean=pt.seq_recall.mean(), seq_f1_mean=pt.seq_f1.mean(),
        inference_time_total_s=infer, inference_time_us_per_candidate=1e6 * infer / len(y),
        inference_time_ms_per_obs=1e3 * infer / obs_total,
    )
    return m, pt, pred_all


def main():
    t_start = time.time()
    obj = load_all()
    inv = pd.DataFrame([r[1] for r in obj["res"]])
    inv.to_csv(f"{OUT}/dataset_inventory.csv", index=False)
    trips = {t: D.load_trip(p, t) for p, t in obj["jobs"]}
    tr_ids = [t for p, t in obj["jobs"] if p == "train"]
    te_ids = [t for p, t in obj["jobs"] if p == "student_test"]
    C = {t: r[0] for (p, t), r in zip(obj["jobs"], obj["res"])}
    cov = {t: r[2] for (p, t), r in zip(obj["jobs"], obj["res"]) if r[2] is not None}

    # ---------- candidate coverage study (train only): supports radius / K choice ----------
    rank = np.concatenate([cov[t][0] for t in tr_ids]); dist = np.concatenate([cov[t][1] for t in tr_ids])
    rows = []
    for R in (25, 50, 75, 100, 150, 200):
        for K in (1, 3, 5, 10, 15, 30):
            rows.append(dict(radius_m=R, max_candidates=K, reference_coverage=float(np.mean((rank >= 0) & (rank < K) & (dist <= R)))))
    covdf = pd.DataFrame(rows); covdf.to_csv(f"{OUT}/candidate_coverage_train.csv", index=False)

    # ---------- 5-fold GROUP cross-validation on the 80 training trips ----------
    gkf = GroupKFold(n_splits=5)
    folds = []
    ids = np.array(tr_ids)
    for k, (a, b) in enumerate(gkf.split(ids, groups=ids)):
        tr_c = [C[t] for t in ids[a] if C[t] is not None]
        va_c = [C[t] for t in ids[b] if C[t] is not None]
        X = np.concatenate([c["X"] for c in tr_c]); y = np.concatenate([c["y"] for c in tr_c])
        m = make_model(); t0 = time.perf_counter(); m.fit(X, y); ft = time.perf_counter() - t0
        met, _, _ = evaluate(m, va_c, trips)
        met.update(fold=k + 1, n_val_trips=len(b), fit_time_s=ft)
        folds.append(met)
        print(f"fold {k+1}: macroF1={met['macro_f1']:.4f} top1={met['top1_acc_all_obs']:.4f} logloss={met['log_loss']:.4f}")
    fdf = pd.DataFrame(folds)
    fdf.drop(columns=["confusion_matrix"]).to_csv(f"{OUT}/model1_cv_folds.csv", index=False)

    # ---------- final fit on all 80 training trips ----------
    tr_c = [C[t] for t in tr_ids if C[t] is not None]
    X = np.concatenate([c["X"] for c in tr_c]); y = np.concatenate([c["y"] for c in tr_c])
    model = make_model(); t0 = time.perf_counter(); model.fit(X, y); fit_time = time.perf_counter() - t0
    n_iter = int(model[-1].n_iter_[0])
    train_m, train_pt, _ = evaluate(model, tr_c, trips)
    te_c = [C[t] for t in te_ids if C[t] is not None]
    test_m, test_pt, test_pred = evaluate(model, te_c, trips)
    test_pt.to_csv(f"{OUT}/model1_student_test_per_trip.csv", index=False)
    joblib.dump(model, f"{OUT}/model1_logreg.joblib")

    coef = pd.DataFrame(dict(feature=FEAT_NAMES, std_coef=model[-1].coef_[0])).sort_values("std_coef")
    coef.to_csv(f"{OUT}/model1_coefficients.csv", index=False)

    cvm = {k: (float(fdf[k].mean()), float(fdf[k].std(ddof=1))) for k in
           ("macro_f1", "top1_acc_all_obs", "route_continuity", "log_loss", "f1_positive", "precision", "recall", "accuracy", "candidate_miss_rate", "seq_f1_mean")}
    summary = dict(
        model="Model 1: Logistic Regression (L2) on (observation, candidate-arc) pairs",
        config=dict(candidate_radius_m=D.RADIUS_M, max_candidates=D.MAX_CANDS, features=FEAT_NAMES, scaler="StandardScaler",
                    C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, class_weight=None, threshold=0.5, seed=SEED,
                    cv="GroupKFold(5) grouped by trip on 80 training trips"),
        n_train_candidate_rows=int(len(y)), lbfgs_iterations=n_iter, fit_time_s=fit_time,
        cv_mean_std=cvm, train=train_m, student_test=test_m,
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__),
        total_runtime_s=time.time() - t_start,
    )
    json.dump(summary, open(f"{OUT}/model1_results.json", "w"), indent=2, default=float)

    comp = pd.DataFrame([dict(
        Model="Model 1: Logistic Regression", Algorithm_Family="Linear / probabilistic",
        Key_Configuration="L2, C=1.0, lbfgs, standardised + squared-distance features, threshold 0.5",
        CV_Macro_F1=f"{cvm['macro_f1'][0]:.4f} +/- {cvm['macro_f1'][1]:.4f}",
        Top1_Accuracy_StudentTest=round(test_m["top1_acc_all_obs"], 4),
        CV_Top1_Accuracy=f"{cvm['top1_acc_all_obs'][0]:.4f}",
        Route_Continuity_StudentTest=round(test_m["route_continuity"], 4),
        StudentTest_Macro_F1=round(test_m["macro_f1"], 4), StudentTest_Log_Loss=round(test_m["log_loss"], 4),
        Inference_Time=f"{test_m['inference_time_ms_per_obs']:.4f} ms/obs ({test_m['inference_time_total_s']:.3f}s total)",
        Train_Time_s=round(fit_time, 2))])
    comp.to_csv(f"{OUT}/model1_comparison_row.csv", index=False)

    make_plots(fdf, cvm, train_m, test_m, test_pt, coef, covdf)
    for tid in ("00000007", "00000052"):
        make_map(trips[tid], C[tid], test_pred[tid])
    print(json.dumps({k: summary[k] for k in ("cv_mean_std", "fit_time_s", "lbfgs_iterations")}, indent=1, default=float))
    print("STUDENT TEST:", json.dumps({k: v for k, v in test_m.items()}, indent=1, default=float))
    print("TRAIN:", {k: round(train_m[k], 4) for k in ("macro_f1", "log_loss", "top1_acc_all_obs", "route_continuity")})


# ------------------------------------------------------------------ plots / map
def make_plots(fdf, cvm, train_m, test_m, test_pt, coef, covdf):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar([f"Fold {i}" for i in fdf.fold], fdf.macro_f1, color="#4c78a8")
    ax.axhline(cvm["macro_f1"][0], color="k", ls="--", lw=1, label=f"mean {cvm['macro_f1'][0]:.3f}")
    ax.set_ylim(min(fdf.macro_f1) - 0.02, max(fdf.macro_f1) + 0.02); ax.set_ylabel("Macro F1 (candidate level)")
    ax.set_title("Model 1 (LogReg): 5-fold group CV Macro F1"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cv_macro_f1.png", dpi=160); plt.close(fig)

    cm = np.array(test_m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%} of row)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["wrong arc", "correct arc"]); ax.set_yticks([0, 1], ["wrong arc", "correct arc"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Student-test confusion matrix"); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_confusion_matrix_student_test.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(test_pt)); w = 0.4
    ax.bar(x - w / 2, test_pt.top1_acc, w, label="Top-1 accuracy", color="#4c78a8")
    ax.bar(x + w / 2, test_pt.route_continuity, w, label="Route continuity", color="#f58518")
    ax.set_xticks(x, test_pt.trip, rotation=45, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_title("Model 1: per-trip results (student-testing)"); ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_student_test_per_trip.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labs = ["Train (80 trips)", "CV (mean)", "Student-test"]
    f1 = [train_m["macro_f1"], cvm["macro_f1"][0], test_m["macro_f1"]]
    t1 = [train_m["top1_acc_all_obs"], cvm["top1_acc_all_obs"][0], test_m["top1_acc_all_obs"]]
    x = np.arange(3)
    ax.bar(x - 0.2, f1, 0.4, label="Macro F1", color="#4c78a8"); ax.bar(x + 0.2, t1, 0.4, label="Top-1 acc.", color="#54a24b")
    for i in range(3):
        ax.text(i - 0.2, f1[i] + 0.005, f"{f1[i]:.3f}", ha="center", fontsize=8); ax.text(i + 0.2, t1[i] + 0.005, f"{t1[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labs); ax.set_ylim(0, 1.08); ax.legend(loc="lower right"); ax.set_title("Over/under-fitting check (same metrics, 3 splits)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_train_cv_test_gap.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.barh(coef.feature, coef.std_coef, color=["#e45756" if v < 0 else "#4c78a8" for v in coef.std_coef])
    ax.set_xlabel("Standardised coefficient (log-odds per 1 SD)"); ax.set_title("Model 1 coefficients"); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_coefficients.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for K in (1, 3, 5, 10, 30):
        s = covdf[covdf.max_candidates == K]
        ax.plot(s.radius_m, s.reference_coverage, marker="o", label=f"K={K}")
    ax.axvline(D.RADIUS_M, color="gray", ls=":"); ax.set_xlabel("Search radius (m)"); ax.set_ylabel("Reference arc in candidate set")
    ax.set_title("Candidate coverage on the 80 training trips"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_candidate_coverage.png", dpi=160); plt.close(fig)


def make_map(trip, c, pred):
    import folium
    ll = lambda i: [[trip["nodes_ll"][i, 1], trip["nodes_ll"][i, 0]]]
    arc_ll = lambda a: [[trip["nodes_ll"][trip["arcs"][a, 0], 1], trip["nodes_ll"][trip["arcs"][a, 0], 0]],
                        [trip["nodes_ll"][trip["arcs"][a, 1], 1], trip["nodes_ll"][trip["arcs"][a, 1], 0]]]
    g = trip["track_ll"]
    m = folium.Map(location=[g[:, 1].mean(), g[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    net = folium.FeatureGroup("Road network (arcs within 100 m of GPS)", show=True)
    for a in np.unique(c["cand"]):
        folium.PolyLine(arc_ll(a), color="#888888", weight=2, opacity=0.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route (green)", show=True)
    for a in trip["route"]:
        folium.PolyLine(arc_ll(a), color="#2ca02c", weight=6, opacity=0.7).add_to(ref)
    ref.add_to(m)
    pr = folium.FeatureGroup("Model 1 predicted route (red)", show=True)
    seen = []
    for a in pred[pred >= 0]:
        if not seen or seen[-1] != a:
            seen.append(a)
    for a in seen:
        folium.PolyLine(arc_ll(a), color="#d62728", weight=3, opacity=0.95).add_to(pr)
    pr.add_to(m)
    gps = folium.FeatureGroup("Raw GPS observations (blue)", show=True)
    ref_obs = c["ref_obs"]
    for i in range(0, len(g), 2):
        folium.CircleMarker([g[i, 1], g[i, 0]], radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=0).add_to(gps)
    gps.add_to(m)
    wrong = folium.FeatureGroup("Mis-matched observations (orange)", show=True)
    bad = np.where(pred != ref_obs)[0]
    for i in bad:
        folium.CircleMarker([g[i, 1], g[i, 0]], radius=4, color="#ff7f0e", fill=True, fill_opacity=0.9, weight=1,
                            tooltip=f"obs {i}: predicted arc {pred[i]}, reference arc {ref_obs[i]}").add_to(wrong)
    wrong.add_to(m)
    top1 = (pred == ref_obs).mean()
    legend = (f'<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;border:1px solid #888;'
              f'font-size:12px;line-height:1.5"><b>Trip {trip["id"]} (student-test) - Model 1 LogReg</b><br>'
              f'<span style="color:#1f77b4">&#9679;</span> raw GPS &nbsp;<span style="color:#2ca02c">&#9632;</span> reference route<br>'
              f'<span style="color:#d62728">&#9632;</span> predicted route &nbsp;<span style="color:#ff7f0e">&#9679;</span> wrong obs<br>'
              f'<span style="color:#888">&#9632;</span> candidate road arcs<br>Top-1 accuracy: {top1:.1%}</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    b = [[g[:, 1].min(), g[:, 0].min()], [g[:, 1].max(), g[:, 0].max()]]
    m.fit_bounds(b)
    m.save(f"{OUT}/map_model1_trip_{trip['id']}.html")


if __name__ == "__main__":
    main()
