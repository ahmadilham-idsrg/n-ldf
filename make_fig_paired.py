#!/usr/bin/env python3
"""Figure: per-dataset paired F1 differences, N-LDF minus each comparison strategy.

Reads results/dataset_method_means.csv and writes figs/fig_paired_diff.pdf.
No model is retrained; the figure re-renders data already in the archive.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- palette: diverging blue <-> red, neutral gray midpoint -------------------
POS   = "#2a78d6"   # N-LDF higher
NEG   = "#e34948"   # comparison higher
TIE   = "#898781"   # exact tie (neutral midpoint)
INK   = "#0b0b0b"
INK2  = "#52514e"
MUTED = "#898781"
GRID  = "#e1e0d9"
SURF  = "#ffffff"

ROWS = [("ADASYN", "ADASYN"),
        ("B-SMOTE", "Borderline-SMOTE"),
        ("SMOTE", "SMOTE"),
        ("SMOTE-Tomek", "SMOTE-Tomek"),
        ("Baseline", "No resampling")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/dataset_method_means.csv", type=Path)
    ap.add_argument("--out", default="figs/fig_paired_diff.pdf", type=Path)
    a = ap.parse_args()

    piv = (pd.read_csv(a.input)
             .pivot(index="Dataset", columns="Method", values="F1"))
    rng = np.random.default_rng(42)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.65, 2.95))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

    # rows top->bottom; a wider gap isolates the baseline row
    ypos = {"ADASYN": 4.0, "B-SMOTE": 3.0, "SMOTE": 2.0, "SMOTE-Tomek": 1.0,
            "Baseline": -0.35}

    for key, label in ROWS:
        diff = (piv["N-LDF"] - piv[key]).dropna().values
        y = ypos[key]
        jit = rng.uniform(-0.22, 0.22, size=diff.size)
        col = np.where(diff > 1e-9, POS, np.where(diff < -1e-9, NEG, TIE))
        ax.scatter(diff, y + jit, s=26, c=col, alpha=0.80,
                   edgecolors=SURF, linewidths=0.9, zorder=3, clip_on=False)
        # median tick: the statistic the Wilcoxon test operates on
        med = float(np.median(diff))
        ax.plot([med, med], [y - 0.42, y + 0.42], color=INK, lw=2.2,
                solid_capstyle="butt", zorder=5)
        # selective direct labels: win-tie-loss, at a fixed right-hand column
        w = int((diff > 1e-9).sum()); l = int((diff < -1e-9).sum())
        t = int(diff.size - w - l)
        tr = ax.get_yaxis_transform()
        ax.text(1.035, y, f"{w}–{t}–{l}", transform=tr, va="center", ha="left",
                fontsize=8.5, color=INK if key == "Baseline" else INK2,
                fontweight="bold" if key == "Baseline" else "normal")
        ax.text(1.205, y, f"{med:+.4f}".replace("-", "−"), transform=tr,
                va="center", ha="left", fontsize=8, color=MUTED)

    # zero reference: the line the whole figure is read against
    ax.axvline(0, color="#c3c2b7", lw=1.0, zorder=2)

    ax.set_yticks([ypos[k] for k, _ in ROWS])
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=9, color=INK)
    ax.set_ylim(-1.05, 4.75)
    ax.set_xlim(-0.28, 0.42)
    ax.set_xticks(np.arange(-0.2, 0.41, 0.1))
    ax.set_xlabel("F1 difference per dataset:  N-LDF $-$ comparison strategy",
                  fontsize=9, color=INK, labelpad=6)

    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7"); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors=MUTED, length=3, width=0.8, labelsize=8)
    ax.tick_params(axis="y", length=0)

    # column headers for the direct-label block
    tr = ax.get_yaxis_transform()
    ax.text(1.035, 4.62, "win–tie–loss", transform=tr, va="center", ha="left",
            fontsize=7.5, color=MUTED)
    ax.text(1.205, 4.62, "median", transform=tr, va="center", ha="left",
            fontsize=7.5, color=MUTED)

    # separator above the baseline row
    ax.plot([-0.28, 0.42], [0.35, 0.35], color=GRID, lw=0.8, zorder=1)

    # compact legend: colour is redundant with side of the zero line
    h = [plt.Line2D([], [], marker="o", ls="", ms=5.2, mfc=POS, mec=SURF, mew=0.9),
         plt.Line2D([], [], marker="o", ls="", ms=5.2, mfc=TIE, mec=SURF, mew=0.9),
         plt.Line2D([], [], marker="o", ls="", ms=5.2, mfc=NEG, mec=SURF, mew=0.9),
         plt.Line2D([], [], color=INK, lw=1.8)]
    ax.legend(h, ["N-LDF higher", "tie", "comparison higher", "median"],
              loc="lower center", bbox_to_anchor=(0.44, -0.30), ncol=4,
              frameon=False, fontsize=8, handletextpad=0.5, columnspacing=1.9,
              labelcolor=INK2)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURF)
    fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight",
                facecolor=SURF)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
