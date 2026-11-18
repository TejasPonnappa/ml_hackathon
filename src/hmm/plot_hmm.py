"""Report section 5 figures, drawn ONLY from saved result files (no model is run).

Inputs : results/b_hmm_comparison.csv, b_hmm_pertrip.csv, b_hmm_best.json,
         b_hmm_tuning_grid1.csv, b_hmm_tuning_grid2.csv, b_hmm_tuning.csv (grid 3)
Outputs: results/b_hmm_pertrip.png   per-trip Top-1, classifier-only vs classifier + HMM
         results/b_hmm_gain.png      Top-1 gain on train OOF (tuned on) vs student_test (held out)
         results/b_hmm_tuning.png    every decoder setting tried, per grid, vs the classifier baseline
Run:  python -m src.hmm.plot_hmm
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import RESULTS_DIR as R

# validated 2-slot categorical palette (blue, orange) + recessive inks
CLF, HMM = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def _clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pertrip():
    pt = pd.read_csv(R / "b_hmm_pertrip.csv", dtype={"trip_id": str}).iloc[::-1]
    y = np.arange(len(pt))
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hlines(y, pt["classifier_top1"], pt["hmm_top1"], color=GRID, linewidth=2, zorder=1)
    ax.scatter(pt["classifier_top1"], y, s=46, color=CLF, edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="classifier only")
    ax.scatter(pt["hmm_top1"], y, s=46, color=HMM, edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="classifier + HMM")
    for yi, (_, r) in zip(y, pt.iterrows()):
        x = max(r["classifier_top1"], r["hmm_top1"])
        ax.text(x + 0.006, yi, f"{r['delta']:+.3f}", va="center", fontsize=9,
                color=INK if r["delta"] >= 0 else "#d03b3b",
                fontweight="normal" if r["delta"] >= 0 else "bold")
    ax.set_yticks(y, [f"trip {t}" for t in pt["trip_id"]])
    ax.set_xlim(0.72, 1.03)
    ax.set_xlabel("observation-level Top-1 accuracy")
    ax.set_title("student_test: Top-1 per trip, classifier-only vs classifier + HMM\n"
                 "(label = change; trip 00000065 is the only one the HMM made worse)",
                 fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(R / "b_hmm_pertrip.png", dpi=150)
    plt.close(fig)


def gain():
    best = json.loads((R / "b_hmm_best.json").read_text())
    cmp_ = pd.read_csv(R / "b_hmm_comparison.csv", index_col=0)
    groups = ["train OOF\n(78 trips, tuned on)", "student_test\n(10 trips, held out)"]
    clf = [best["classifier_only_top1_same_trips"], cmp_.loc["classifier_only", "top1_accuracy"]]
    hmm = [best["tuning_top1"], cmp_.loc["classifier_plus_hmm", "top1_accuracy"]]
    x, w = np.arange(2), 0.34
    fig, ax = plt.subplots(figsize=(6, 4.4))
    b1 = ax.bar(x - w / 2 - 0.01, clf, w, color=CLF, label="classifier only")
    b2 = ax.bar(x + w / 2 + 0.01, hmm, w, color=HMM, label="classifier + HMM")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.4f}",
                    ha="center", fontsize=9, color=INK)
    for xi, c, h in zip(x, clf, hmm):
        ax.text(xi, 0.97, f"gain {h - c:+.4f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, groups)
    ax.set_ylim(0.85, 0.99)
    ax.set_ylabel("Top-1 accuracy (axis starts at 0.85)")
    ax.set_title("HMM gain on the data it was tuned on vs held-out data", fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    _clean(ax, "y")
    fig.tight_layout()
    fig.savefig(R / "b_hmm_gain.png", dpi=150)
    plt.close(fig)


def tuning():
    best = json.loads((R / "b_hmm_best.json").read_text())
    grids = [("grid 1\n54 settings", "b_hmm_tuning_grid1.csv"),
             ("grid 2\n81 settings", "b_hmm_tuning_grid2.csv"),
             ("grid 3 (final)\n81 settings", "b_hmm_tuning.csv")]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (label, f) in enumerate(grids):
        d = pd.read_csv(R / f)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d["top1"], s=14, color=HMM, alpha=0.55,
                   edgecolor="none")
        top = d["top1"].max()
        ax.scatter([i], [top], s=60, color=HMM, edgecolor=INK, linewidth=1, zorder=3)
        ax.text(i + 0.24, top, f"best {top:.4f}", va="center", fontsize=9, color=INK)
    base = best["classifier_only_top1_same_trips"]
    ax.axhline(base, color=CLF, linewidth=2, linestyle="--")
    ax.text(2.45, base + 0.002, f"classifier only {base:.4f}", ha="right", fontsize=9, color=INK2)
    ax.set_xticks(range(3), [g for g, _ in grids])
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylabel("Top-1 accuracy (train OOF, 78 trips)")
    p = best["params"]
    ax.set_title("Decoder tuning: every setting tried (one dot each)\n"
                 f"final: alpha={p['alpha']}, beta={p['beta']}, invalid={p['invalid']}, uturn={p['uturn']}",
                 fontsize=10, loc="left")
    _clean(ax, "y")
    fig.tight_layout()
    fig.savefig(R / "b_hmm_tuning.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    pertrip()
    gain()
    tuning()
    print("[plot] saved results/b_hmm_pertrip.png, b_hmm_gain.png, b_hmm_tuning.png")
