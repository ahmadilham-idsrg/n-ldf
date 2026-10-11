#!/usr/bin/env python3
"""Figure 2 in OriginLab house style, rendered with matplotlib so it runs in
Google Colab.

Why not the originpro package: every OriginLab Python package (originpro,
OriginExt, OrglabExt) is Windows-only and requires a local installation of
Origin 2021 or later. Google Colab runs on Linux with no Origin installed, so
none of them can be imported there. This script reproduces the Origin default
graph style -- full frame box, inward major and minor ticks on all four sides,
Arial, no grid, framed legend -- with matplotlib alone.

Usage:
    python make_fig_paired_origin.py
    python make_fig_paired_origin.py --input results/dataset_method_means.csv \
                                     --out figs/fig_paired_diff_origin.pdf

Writes the figure as .pdf, .png and .tif (300 dpi), and writes a tidy CSV of
the plotted values for importing into Origin itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

# --- colours -----------------------------------------------------------------
# Diverging pair, validated for colour-vision deficiency (CVD dE 21.6, normal
# vision 32.3, contrast >= 3:1 on white). Origin's stock pure red #FF0000 and
# pure blue #0000FF are kept out because they do not separate under deuteranopia.
POS = "#2A78D6"   # N-LDF higher
NEG = "#E34948"   # comparison strategy higher
TIE = "#898781"   # exact tie: the neutral midpoint of the diverging scale
BLACK = "#000000"

ROWS = [("ADASYN", "ADASYN"),
        ("B-SMOTE", "Borderline-SMOTE"),
        ("SMOTE", "SMOTE"),
        ("SMOTE-Tomek", "SMOTE-Tomek"),
        ("Baseline", "No resampling")]

YPOS = {"ADASYN": 4.0, "B-SMOTE": 3.0, "SMOTE": 2.0, "SMOTE-Tomek": 1.0,
        "Baseline": -0.35}


def origin_rcparams() -> None:
    """Origin's default graph appearance, expressed as matplotlib rcParams.

    Arial is Origin's default face. Colab ships Liberation Sans, which is
    metrically identical to Arial, so the fallback chain changes nothing
    measurable. Run ``!apt-get -qq install -y fonts-liberation`` in Colab if
    the fallback list comes up empty.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica",
                            "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 1.0,          # Origin frame weight
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "axes.grid": False,             # Origin draws no grid by default
        "text.color": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5.0, "xtick.minor.size": 2.6,
        "ytick.major.size": 5.0, "ytick.minor.size": 2.6,
        "xtick.major.width": 1.0, "xtick.minor.width": 0.8,
        "ytick.major.width": 1.0, "ytick.minor.width": 0.8,
        "legend.frameon": True, "legend.fancybox": False,
        "legend.edgecolor": BLACK, "legend.framealpha": 1.0,
        "legend.borderpad": 0.45,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed as TrueType, not Type 3
    })


def paired_table(piv: pd.DataFrame) -> pd.DataFrame:
    """Long-format table of every plotted point, for import into Origin."""
    out = []
    for key, label in ROWS:
        d = (piv["N-LDF"] - piv[key]).dropna()
        for ds, v in d.items():
            out.append({"Comparison": label, "Dataset": ds,
                        "Diff_F1": round(float(v), 6),
                        "Outcome": "Win" if v > 1e-9
                                   else ("Loss" if v < -1e-9 else "Tie")})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/dataset_method_means.csv",
                    type=Path)
    ap.add_argument("--out", default="figs/fig_paired_diff_origin.pdf",
                    type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    piv = (pd.read_csv(a.input)
             .pivot(index="Dataset", columns="Method", values="F1"))
    rng = np.random.default_rng(42)     # fixed jitter, so the figure is stable

    origin_rcparams()
    fig, ax = plt.subplots(figsize=(5.9, 3.15))

    for key, label in ROWS:
        diff = (piv["N-LDF"] - piv[key]).dropna().values
        y = YPOS[key]
        jit = rng.uniform(-0.21, 0.21, size=diff.size)
        col = np.where(diff > 1e-9, POS, np.where(diff < -1e-9, NEG, TIE))
        ax.scatter(diff, y + jit, s=25, c=col, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=4, clip_on=False)
        med = float(np.median(diff))
        ax.plot([med, med], [y - 0.40, y + 0.40], color=BLACK, lw=2.0,
                solid_capstyle="butt", zorder=6)

        w = int((diff > 1e-9).sum())
        l = int((diff < -1e-9).sum())
        t = int(diff.size - w - l)
        tr = ax.get_yaxis_transform()
        ax.text(1.030, y, f"{w}–{t}–{l}", transform=tr, va="center",
                ha="left", fontsize=8.5,
                fontweight="bold" if key == "Baseline" else "normal")
        ax.text(1.190, y, f"{med:+.4f}".replace("-", "−"), transform=tr,
                va="center", ha="left", fontsize=8)

    ax.axvline(0, color="#9A9A9A", lw=0.9, ls=(0, (4, 2.5)), zorder=2)

    ax.set_yticks([YPOS[k] for k, _ in ROWS])
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=9)
    ax.set_ylim(-1.05, 4.75)
    ax.set_xlim(-0.28, 0.42)
    ax.set_xticks(np.arange(-0.2, 0.41, 0.1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="both", length=0)   # categorical: no ticks
    ax.tick_params(axis="x", which="major", labelsize=8.5)
    ax.set_xlabel("F1 difference per dataset:  N-LDF $-$ comparison strategy",
                  fontsize=9, labelpad=5)

    tr = ax.get_yaxis_transform()
    ax.text(1.030, 4.60, "win–tie–loss", transform=tr, va="center",
            ha="left", fontsize=7.5)
    ax.text(1.190, 4.60, "median", transform=tr, va="center", ha="left",
            fontsize=7.5)
    ax.plot([-0.28, 0.42], [0.35, 0.35], color="#B4B4B4", lw=0.8, zorder=3)

    h = [plt.Line2D([], [], marker="o", ls="", ms=5.0, mfc=c, mec="white",
                    mew=0.8) for c in (POS, TIE, NEG)]
    h.append(plt.Line2D([], [], color=BLACK, lw=1.8))
    ax.legend(h, ["N-LDF higher", "tie", "comparison higher", "median"],
              loc="lower center", bbox_to_anchor=(0.44, -0.335), ncol=4,
              fontsize=8, handletextpad=0.45, columnspacing=1.5)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png", ".tif"):
        fig.savefig(a.out.with_suffix(ext), dpi=a.dpi, bbox_inches="tight")

    tidy = a.out.with_name(a.out.stem + "_data.csv")
    paired_table(piv).to_csv(tidy, index=False)

    print("wrote", a.out.with_suffix(".pdf"))
    print("wrote", a.out.with_suffix(".png"))
    print("wrote", a.out.with_suffix(".tif"), f"({a.dpi} dpi)")
    print("wrote", tidy, "-- import this into Origin to redraw it there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
