#!/usr/bin/env python3
"""Figure 1: why linear interpolation between minority neighbors can place a
synthetic instance inside the majority region. Synthetic 2-D illustration with
a fixed seed; no benchmark data involved.

Usage:
    python make_fig1.py [--out figs/fig1.pdf]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, GRAY, INK = "#2A78D6", "#E34948", "#898781", "#222222"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/fig1.pdf", type=Path)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 8.5, "axes.linewidth": 0.8})
    rng = np.random.default_rng(7)

    # majority class: one broad blob
    maj = rng.normal(loc=(0.0, 0.0), scale=(1.15, 0.75), size=(70, 2))
    # minority class: two small disjuncts on either side of the majority blob
    min_left = rng.normal(loc=(-2.9, 0.9), scale=(0.28, 0.28), size=(6, 2))
    min_right = rng.normal(loc=(2.9, 0.7), scale=(0.28, 0.28), size=(7, 2))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.scatter(maj[:, 0], maj[:, 1], s=16, facecolor=BLUE, edgecolor="white",
               linewidth=0.5, zorder=2, label="majority")
    for grp in (min_left, min_right):
        ax.scatter(grp[:, 0], grp[:, 1], s=26, marker="^", facecolor=RED,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter([], [], s=26, marker="^", facecolor=RED, edgecolor="white",
               linewidth=0.5, label="minority")

    # a 'safe' interpolation inside the right disjunct
    p, q = min_right[0], min_right[3]
    s_ok = p + 0.5 * (q - p)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=GRAY, lw=0.9, ls="--", zorder=1)
    ax.scatter(s_ok[0], s_ok[1], s=48, marker="*", facecolor="white",
               edgecolor=RED, linewidth=1.0, zorder=4)
    ax.annotate("interpolation within\na minority region", xy=s_ok,
                xytext=(1.25, 2.25), fontsize=7.2, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY))

    # an interpolation across the majority region (seed on the left, partner on the right)
    u, v = min_left[1], min_right[2]
    s_bad = u + 0.5 * (v - u)
    ax.plot([u[0], v[0]], [u[1], v[1]], color=GRAY, lw=0.9, ls="--", zorder=1)
    ax.scatter(s_bad[0], s_bad[1], s=48, marker="*", facecolor="white",
               edgecolor=RED, linewidth=1.0, zorder=4)
    ax.annotate("interpolation across\nthe majority region", xy=s_bad,
                xytext=(-3.6, -2.2), fontsize=7.2, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY))
    ax.scatter([], [], s=48, marker="*", facecolor="white", edgecolor=RED,
               linewidth=1.0, label="synthetic")

    ax.set_xlim(-4.2, 4.6)
    ax.set_ylim(-2.7, 2.9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.legend(loc="lower right", fontsize=7.2, frameon=True, framealpha=1.0,
              edgecolor="black", handletextpad=0.4, borderpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
