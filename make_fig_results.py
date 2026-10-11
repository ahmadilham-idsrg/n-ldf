#!/usr/bin/env python3
"""Result figures of the N-LDF manuscript, drawn from the archived results.

  figs/fig_f1_severity.pdf  (Figure 3 of the MLJ revision) minority-class F1 by
                            imbalance severity, with source-cluster percentile-
                            bootstrap 95% intervals (resampling unit = source
                            dataset, so same-source dependence is preserved)
  figs/fig_cd_diagram.pdf   only with --cd-diagram: a Nemenyi critical-difference
                            diagram over the six strategies. It is not part of
                            the manuscript, which does not use mean-rank post-hoc
                            procedures (Sect. 4; Benavoli et al. 2016), and it is
                            not in the repository.

Both read results/fold_level_results.csv and retrain nothing. The Nemenyi
critical difference uses q_0.05(k=6) = 2.850 (Demsar 2006, Table 5).

Next to each figure the script writes the values it plotted, so the figures
can be redrawn in any other tool (Origin, R, Excel) without re-deriving them:

  figs/fig_dataset_level_f1.csv      246 rows: Dataset, Severity, IR, Method,
                                     F1 (dataset-level mean) and Rank (1 = best
                                     of the six strategies on that dataset).
                                     This one table feeds both figures.
  figs/fig_cd_diagram_data.csv       only with --cd-diagram: 6 rows: Method,
                                     MeanRank, plus N, k, CD and the group each
                                     method falls in.
  figs/fig_f1_severity_data.csv      18 rows: Severity, Method, n, n_sources,
                                     Mean, CI_low, CI_high (source-cluster
                                     percentile bootstrap, 10,000 resamples,
                                     seed 42).

Usage:
    python make_fig_results.py [--input results/fold_level_results.csv] [--outdir figs] [--cd-diagram]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

METHODS = ["Baseline", "SMOTE", "ADASYN", "B-SMOTE", "SMOTE-Tomek", "N-LDF"]
LABELS = {"Baseline": "No resampling", "SMOTE": "SMOTE", "ADASYN": "ADASYN",
          "B-SMOTE": "Borderline-SMOTE", "SMOTE-Tomek": "SMOTE-Tomek",
          "N-LDF": "N-LDF"}
STRATA = [("Mild", r"Mild  (IR < 9)"), ("High", r"High  (9 $\leq$ IR < 40)"),
          ("Extreme", r"Extreme  (IR $\geq$ 40)")]
Q_ALPHA_K6 = 2.850          # Nemenyi critical value, alpha = 0.05, k = 6

BLACK = "#000000"
HIGHLIGHT = "#2A78D6"       # N-LDF, the method under study
NEUTRAL = "#52514E"         # every comparison strategy
GRAYLINE = "#B4B4B4"


def origin_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica",
                            "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 1.0, "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK, "axes.grid": False, "text.color": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 5.0, "xtick.minor.size": 2.6,
        "ytick.major.size": 5.0, "ytick.minor.size": 2.6,
        "xtick.major.width": 1.0, "xtick.minor.width": 0.8,
        "ytick.major.width": 1.0, "ytick.minor.width": 0.8,
        "legend.frameon": True, "legend.fancybox": False,
        "legend.edgecolor": BLACK, "legend.framealpha": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def dataset_method_means(fold: pd.DataFrame) -> pd.DataFrame:
    return fold.groupby(["Dataset", "Method"])["F1"].mean().reset_index()


def dataset_level_table(fold: pd.DataFrame, dmm: pd.DataFrame) -> pd.DataFrame:
    """The 41 x 6 dataset-level F1 values with their within-dataset ranks."""
    meta = fold[["Dataset", "Severity", "IR"]].drop_duplicates("Dataset")
    f1 = dmm.pivot(index="Dataset", columns="Method", values="F1")[METHODS]
    rank = f1.rank(axis=1, ascending=False)
    long = (f1.stack().rename("F1").reset_index()
              .merge(rank.stack().rename("Rank").reset_index(),
                     on=["Dataset", "Method"])
              .merge(meta, on="Dataset"))
    long["Rank"] = long["Rank"].round(1)
    long["F1"] = long["F1"].round(6)
    long["IR"] = long["IR"].round(2)
    order = {m: i for i, m in enumerate(METHODS)}
    long["_m"] = long["Method"].map(order)
    long = (long.sort_values(["IR", "_m"]).drop(columns="_m")
                [["Dataset", "Severity", "IR", "Method", "F1", "Rank"]])
    return long.reset_index(drop=True)


# --------------------------------------------------------------------------
# Figure 3: critical-difference diagram
# --------------------------------------------------------------------------
def fig_cd(dmm: pd.DataFrame, out: Path) -> dict:
    f1 = dmm.pivot(index="Dataset", columns="Method", values="F1")[METHODS].dropna()
    n, k = f1.shape
    ranks = f1.rank(axis=1, ascending=False).mean().sort_values()
    cd = Q_ALPHA_K6 * np.sqrt(k * (k + 1) / (6.0 * n))

    fig, ax = plt.subplots(figsize=(5.9, 2.15))
    lo, hi = 1.0, float(k)
    ax.set_xlim(hi + 0.15, lo - 0.15)                 # best rank on the right
    ax.set_ylim(-2.75, 1.75)
    ax.spines[["left", "right", "bottom"]].set_visible(False)
    ax.spines["top"].set_position(("data", 0.0))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(np.arange(1, k + 1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis="x", which="both", direction="out", pad=2,
                   top=True, bottom=False, labelsize=8.5)
    ax.set_yticks([])
    ax.text((lo + hi) / 2, -2.62,
            "mean rank of dataset-level minority-class F1  (lower is better)",
            ha="center", va="bottom", fontsize=8.5)

    # CD scale bar
    ax.plot([hi, hi - cd], [1.05, 1.05], color=BLACK, lw=1.6,
            solid_capstyle="butt", clip_on=False)
    for x in (hi, hi - cd):
        ax.plot([x, x], [0.92, 1.18], color=BLACK, lw=1.0, clip_on=False)
    ax.text(hi - cd / 2, 1.28, f"CD = {cd:.3f}", ha="center", va="bottom",
            fontsize=8, clip_on=False)

    # Group bars: maximal sets of adjacent methods within CD of each other
    order = list(ranks.index)
    vals = ranks.values
    groups = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            groups.append((i, j))
        i = j + 1 if j > i else i + 1
    for gi, (a, b) in enumerate(groups):
        y = -0.42 - 0.30 * gi
        ax.plot([vals[a] - 0.04, vals[b] + 0.04], [y, y], color=BLACK, lw=3.2,
                solid_capstyle="butt")

    # Method labels: best half on the right, rest on the left
    half = (k + 1) // 2
    ybase = -0.95 - 0.30 * max(0, len(groups) - 1)
    for idx, (name, r) in enumerate(ranks.items()):
        col = HIGHLIGHT if name == "N-LDF" else NEUTRAL
        if idx < half:                                   # right side
            slot = idx
            xt, ha = lo - 0.12, "left"
        else:                                            # left side
            slot = idx - half
            xt, ha = hi + 0.12, "right"
        yt = ybase - 0.55 * slot
        ax.plot([r, r, xt], [0, yt, yt], color=col, lw=0.9)
        ax.text(xt + (-0.03 if ha == "left" else 0.03), yt,
                f"{LABELS[name]}  ({r:.3f})", ha=ha, va="center", fontsize=8.5,
                color=col, fontweight="bold" if name == "N-LDF" else "normal")

    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    group_of = {}
    for gi, (a, b) in enumerate(groups, start=1):
        for idx in range(a, b + 1):
            group_of[order[idx]] = gi
    data = pd.DataFrame({
        "Method": order,
        "Label": [LABELS[m] for m in order],
        "MeanRank": np.round(vals, 4),
        "N_datasets": n, "k_methods": k, "CD_alpha05": round(cd, 4),
        "Group": [group_of.get(m, "") for m in order],
    })
    data.to_csv(out.with_name(out.stem + "_data.csv"), index=False)
    return {"n": n, "k": k, "cd": cd, "ranks": ranks.round(4).to_dict(),
            "groups": [[order[a], order[b]] for a, b in groups]}


# --------------------------------------------------------------------------
# Figure 4: F1 by severity with bootstrap intervals
# --------------------------------------------------------------------------
def source_of(name: str) -> str:
    """Source dataset of a KEEL task, by name prefix (Section 5.1)."""
    for pfx in ["yeast", "abalone", "ecoli", "kddcup", "winequality", "glass",
                "poker", "kr-vs-k"]:
        if name.startswith(pfx):
            return pfx
    return name


def fig_severity(fold: pd.DataFrame, dmm: pd.DataFrame, out: Path,
                 n_boot: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Stratum means with source-cluster percentile-bootstrap 95% intervals.

    Within each stratum the resampling unit is the source dataset: every
    task of a drawn source enters together, so same-source dependence is
    preserved in the interval (revision, Section 6.5).
    """
    sev = fold[["Dataset", "Severity"]].drop_duplicates("Dataset")
    d = dmm.merge(sev, on="Dataset")
    rng = np.random.default_rng(seed)
    rows = []
    for s, _ in STRATA:
        ds = d[d.Severity == s]
        tasks = sorted(ds.Dataset.unique())
        groups = {}
        for t in tasks:
            groups.setdefault(source_of(t), []).append(t)
        names = list(groups)
        f1 = ds.pivot(index="Dataset", columns="Method", values="F1")
        for m in METHODS:
            v = f1[m]
            boot = np.empty(n_boot)
            for i in range(n_boot):
                pick = rng.choice(names, size=len(names), replace=True)
                sel = [t for k in pick for t in groups[k]]
                boot[i] = v.loc[sel].mean()
            rows.append({"Severity": s, "Method": m, "n": len(tasks),
                         "n_sources": len(names), "mean": float(v.mean()),
                         "lo": float(np.percentile(boot, 2.5)),
                         "hi": float(np.percentile(boot, 97.5))})
    tab = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharey=True)
    ypos = {m: i for i, m in enumerate(reversed(METHODS))}
    for ax, (s, title), letter in zip(axes, STRATA, "abc"):
        # Panel letter (Springer Nature style: bold lowercase, referenced as
        # "a", "b", "c" in the caption).
        ax.text(0.0, 1.12, letter, transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="left", va="bottom")
        t = tab[tab.Severity == s].set_index("Method")
        for m in METHODS:
            col = HIGHLIGHT if m == "N-LDF" else NEUTRAL
            y = ypos[m]
            ax.plot([t.loc[m, "lo"], t.loc[m, "hi"]], [y, y], color=col,
                    lw=1.4, solid_capstyle="butt", zorder=2)
            ax.plot(t.loc[m, "mean"], y, marker="o", ms=5.6, mfc=col, mec="white",
                    mew=0.8, ls="", zorder=3)
        n, ns = int(t["n"].iloc[0]), int(t["n_sources"].iloc[0])
        ax.set_title(f"{title}\n$n$ = {n} tasks from {ns} sources", fontsize=8.0, pad=4)
        ax.set_xlim(0.10, 0.90)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_ylim(-0.6, len(METHODS) - 0.4)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", which="both", length=0)
    axes[0].set_yticks([ypos[m] for m in METHODS])
    axes[0].set_yticklabels([LABELS[m] for m in METHODS], fontsize=8.5)
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == "N-LDF":
            lab.set_color(HIGHLIGHT); lab.set_fontweight("bold")
    axes[1].set_xlabel("dataset-level minority-class F1  (mean and source-cluster bootstrap 95% interval)",
                       fontsize=8.5, labelpad=5)
    fig.subplots_adjust(wspace=0.10)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    tidy = tab.rename(columns={"mean": "Mean", "lo": "CI_low", "hi": "CI_high"})
    tidy["Label"] = tidy["Method"].map(LABELS)
    tidy["n_boot"] = n_boot
    tidy["seed"] = seed
    tidy["resampling_unit"] = "source dataset"
    tidy = tidy[["Severity", "Method", "Label", "n", "n_sources", "Mean", "CI_low",
                 "CI_high", "n_boot", "seed", "resampling_unit"]].round(6)
    tidy.to_csv(out.with_name(out.stem + "_data.csv"), index=False)
    return tab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/fold_level_results.csv", type=Path)
    ap.add_argument("--outdir", default="figs", type=Path)
    ap.add_argument("--cd-diagram", action="store_true",
                    help="also draw the Nemenyi CD diagram (not used in the manuscript)")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    origin_rcparams()
    fold = pd.read_csv(a.input, low_memory=False)
    dmm = dataset_method_means(fold)

    long = dataset_level_table(fold, dmm)
    long.to_csv(a.outdir / "fig_dataset_level_f1.csv", index=False)
    print(f"wrote {a.outdir / 'fig_dataset_level_f1.csv'}  ({len(long)} rows)")

    if a.cd_diagram:
        info = fig_cd(dmm, a.outdir / "fig_cd_diagram.pdf")
        print(f"CD diagram (not used in the manuscript)  N={info['n']}  k={info['k']}  CD={info['cd']:.4f}")
        for m, r in info["ranks"].items():
            print(f"    {m:<12} {r:.4f}")
        print("    groups within CD:", info["groups"])

    tab = fig_severity(fold, dmm, a.outdir / "fig_f1_severity.pdf")
    print("\nFigure 4  (mean, source-cluster bootstrap 95% interval)")
    print(tab.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
