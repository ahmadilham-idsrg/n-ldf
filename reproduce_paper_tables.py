#!/usr/bin/env python3
"""Regenerate every number reported in the N-LDF manuscript from the raw
fold-level records.

This script retrains nothing. It reads ``results/fold_level_results.csv`` and
prints Tables 5, 6 and 7 of the paper, the statistical tests and paired
differences of Section 6.2, the variance decomposition of Section 6.1, the
observed eta diagnostics of Section 4.3 (Table 2), the acceptance-rate
accounting of Section 5.2, and the fit accounting of Section 5.3 (table numbers
refer to the MLJ revision; supplementary_analysis.py covers the analyses added
at revision), so that any
reader can check the manuscript against the archived data in one command:

    python reproduce_paper_tables.py

Use ``--check`` to compare the computed values against the values printed in
the manuscript; the script then exits non-zero if any of them disagree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["Baseline", "SMOTE", "ADASYN", "B-SMOTE", "SMOTE-Tomek", "N-LDF"]
OVERSAMPLERS = [m for m in METHODS if m != "Baseline"]
STRATA = ["Mild", "High", "Extreme"]

# Values as printed in the manuscript, for --check.
PAPER = {
    "n_fits": 21911,
    "table3": {
        "Baseline":    (0.4940, 0.5858, 0.5366, 0.8759, 0.6392, 4.9512),
        "SMOTE":       (0.5777, 0.5603, 0.7466, 0.8904, 0.6369, 3.0976),
        "ADASYN":      (0.5672, 0.5430, 0.7438, 0.8891, 0.6276, 3.5610),
        "B-SMOTE":     (0.5792, 0.5724, 0.7239, 0.8873, 0.6359, 3.1829),
        "SMOTE-Tomek": (0.5776, 0.5606, 0.7457, 0.8909, 0.6366, 3.0732),
        "N-LDF":       (0.5821, 0.6307, 0.6739, 0.8868, 0.6522, 3.1341),
    },
    "table6": {
        "Mild":    (0.6707, 0.7289, 0.7229, 0.7222, 0.7306, 0.7181),
        "High":    (0.4599, 0.5523, 0.5240, 0.5453, 0.5508, 0.5857),
        "Extreme": (0.4225, 0.5142, 0.5126, 0.5254, 0.5140, 0.5083),
    },
    "friedman_all": (33.6485, 2.80e-06),
    "friedman_oversamplers": (1.9294, 0.7487),
    "holm_baseline": 3.355e-07,        # Wilcoxon, zero_method="pratt", Holm over 5
    # Figure 2: win-tie-loss and median of the per-dataset F1 difference
    "paired": {
        "SMOTE":       (19, 3, 19,  0.0000),
        "ADASYN":      (20, 2, 19,  0.0000),
        "B-SMOTE":     (17, 2, 22, -0.0025),
        "SMOTE-Tomek": (19, 3, 19,  0.0000),
        "Baseline":    (36, 3,  2,  0.0701),
    },
    "paired_range": (-0.24, 0.39),
    # Section 6.1: share of the total F1 variation, in per cent
    "variance": {"dataset": 98.88, "method": 0.03, "interaction": 1.10},
    # Section 4.3, Table 2: observed per-call mean of eta, averaged per dataset
    "eta_observed": {          # stratum: (datasets, mean, n >= 0.8, n <= 0)
        "Mild":    (10, 0.47, 2, 1),
        "High":    (12, 0.32, 1, 3),
        "Extreme": (19, 0.17, 7, 8),
    },
    "eta_ir9": (31, 8, 11),            # IR >= 9: datasets, n >= 0.8, n <= 0
    "eta_spearman": -0.12,             # Spearman(IR, dataset mean eta)
    "eta_extremes": (21, 26),          # datasets with max >= 0.99 / min <= -0.99
    "n_calls": 914,
    # Section 5.2: acceptance and the resampling amount actually generated
    "acceptance_floor_others": 0.74,   # every dataset except the two chess tasks
    "zero_acceptance": ["kr-vs-k-zero_vs_eight", "kr-vs-k-zero_vs_fifteen"],
    "max_accepted_chess": 4,
    "attempt_totals": {"attempted": 646754, "accepted": 524557,
                       "distance": 106178, "ratio": 7054, "factor": 4048,
                       "no_candidate": 4917},
    "datasets_at_target": 39,          # accepted == floor(0.75 n_maj) - n_min
    "datasets_over_20x": 21,           # accepted > 20 n_min: no cap was applied
    "attempt_multiplier_chess": 5,     # attempts == 5 * requested on the chess tasks
    # Section 6.2, exploratory family: Holm over all 15 pairs
    "exploratory_baseline_max_holm": 1.24e-03,
    "exploratory_oversampler_min_holm": 0.42,
}


def dataset_method_means(fold: pd.DataFrame) -> pd.DataFrame:
    """One value per dataset-method pair: the direct mean of every available
    fold-level score, across classifiers, repeats and folds (Section 5.3)."""
    metrics = ["F1", "Precision", "G-Mean", "AUC-ROC", "AUPRC"]
    return fold.groupby(["Dataset", "Method"])[metrics].mean().reset_index()


def table3(dmm: pd.DataFrame) -> pd.DataFrame:
    metrics = ["F1", "Precision", "G-Mean", "AUC-ROC", "AUPRC"]
    out = dmm.groupby("Method")[metrics].mean()
    f1 = dmm.pivot(index="Dataset", columns="Method", values="F1")[METHODS]
    out["Mean F1 Rank"] = f1.rank(axis=1, ascending=False).mean()
    return out.loc[METHODS].round(4)


def table5(fold: pd.DataFrame) -> pd.DataFrame:
    """Per-classifier means: dataset-level first, then equal weight per dataset."""
    per = (fold.groupby(["Classifier", "Dataset", "Method"])[["F1", "G-Mean"]]
               .mean().reset_index())
    out = (per.groupby(["Classifier", "Method"])[["F1", "G-Mean"]]
              .mean().round(4))
    return out.unstack("Method").swaplevel(axis=1).sort_index(axis=1)


def table6(fold: pd.DataFrame, dmm: pd.DataFrame) -> pd.DataFrame:
    sev = fold[["Dataset", "Severity"]].drop_duplicates("Dataset")
    merged = dmm.merge(sev, on="Dataset")
    out = merged.pivot_table(index="Severity", columns="Method",
                             values="F1", aggfunc="mean")
    return out.reindex(STRATA)[METHODS].round(4)


def statistics(dmm: pd.DataFrame) -> dict:
    f1 = dmm.pivot(index="Dataset", columns="Method", values="F1")[METHODS].dropna()
    chi_all, p_all = stats.friedmanchisquare(*[f1[m] for m in METHODS])
    chi_os, p_os = stats.friedmanchisquare(*[f1[m] for m in OVERSAMPLERS])

    pairs = []
    for m in METHODS:
        if m == "N-LDF":
            continue
        w, p = stats.wilcoxon(f1["N-LDF"], f1[m], zero_method="pratt",
                              alternative="two-sided")
        pairs.append({"vs": m, "W": w, "raw_p": p})
    pairs.sort(key=lambda r: r["raw_p"])
    k, running = len(pairs), 0.0
    for i, row in enumerate(pairs):          # Holm step-down, monotone-enforced
        running = max(running, min(1.0, row["raw_p"] * (k - i)))
        row["holm_p"] = running

    # exploratory family: every pair among the six strategies, Holm over 15
    import itertools
    expl = []
    for a, b in itertools.combinations(METHODS, 2):
        w, p = stats.wilcoxon(f1[a], f1[b], zero_method="pratt",
                              alternative="two-sided")
        expl.append({"a": a, "b": b, "W": w, "raw_p": p})
    expl.sort(key=lambda r: r["raw_p"])
    k, running = len(expl), 0.0
    for i, row in enumerate(expl):
        running = max(running, min(1.0, row["raw_p"] * (k - i)))
        row["holm_p"] = running

    return {
        "n_datasets": len(f1),
        "friedman_all": (chi_all, p_all),
        "friedman_oversamplers": (chi_os, p_os),
        "pairwise": pairs,
        "exploratory": expl,
    }


def paired_differences(dmm: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset F1 differences, N-LDF minus each comparison strategy.

    These are the values the Wilcoxon signed-rank tests operate on and the
    values plotted in Figure 2. A difference is counted as a tie only when the
    two dataset-level means agree to within 1e-9.
    """
    f1 = dmm.pivot(index="Dataset", columns="Method", values="F1")
    rows = {}
    for m in ["ADASYN", "B-SMOTE", "SMOTE", "SMOTE-Tomek", "Baseline"]:
        d = (f1["N-LDF"] - f1[m]).dropna()
        rows[m] = {
            "n": len(d),
            "win": int((d > 1e-9).sum()),
            "tie": int((d.abs() <= 1e-9).sum()),
            "loss": int((d < -1e-9).sum()),
            "median": float(d.median()),
            "min": float(d.min()),
            "max": float(d.max()),
        }
    return pd.DataFrame(rows).T


def variance_decomposition(dmm: pd.DataFrame) -> dict:
    """Two-way decomposition of dataset-level F1 over datasets x oversamplers.

    One observation per cell, so the residual carries the dataset-method
    interaction. Returns each term's share of the total sum of squares.
    """
    x = dmm.pivot(index="Dataset", columns="Method",
                  values="F1")[OVERSAMPLERS].dropna()
    v = x.to_numpy()
    n, k = v.shape
    grand = v.mean()
    ss_total = ((v - grand) ** 2).sum()
    ss_dataset = k * ((v.mean(axis=1) - grand) ** 2).sum()
    ss_method = n * ((v.mean(axis=0) - grand) ** 2).sum()
    return {
        "n_datasets": n,
        "n_methods": k,
        "dataset": 100 * ss_dataset / ss_total,
        "method": 100 * ss_method / ss_total,
        "interaction": 100 * (ss_total - ss_dataset - ss_method) / ss_total,
    }


def eta_diagnostics(fold: pd.DataFrame) -> dict:
    """Observed eta logged by the sampler at each N-LDF resampling call.

    ``eta_mean`` is the mean of eta_i over minority instances after the
    IR**lambda_p adjustment; one row per (dataset, repeat, fold) call.
    """
    calls = (fold[fold.Method == "N-LDF"]
                 .drop_duplicates(["Dataset", "Repeat", "Fold"]))
    per = calls.groupby("Dataset").agg(
        Severity=("Severity", "first"), IR=("IR", "first"),
        eta_mean=("eta_mean", "mean"), eta_min=("eta_min", "mean"),
        eta_max=("eta_max", "mean"), acceptance=("acceptance_rate", "mean"),
        n_accepted_max=("n_accepted", "max"))
    strata = {}
    for s in STRATA:
        t = per[per.Severity == s]
        strata[s] = (len(t), t.eta_mean.mean(),
                     int((t.eta_mean >= 0.8).sum()), int((t.eta_mean <= 0).sum()))
    hi = per[per.IR >= 9]
    rho = stats.spearmanr(per.IR, per.eta_mean).statistic
    return {
        "per_dataset": per, "n_calls": len(calls), "strata": strata,
        "ir9": (len(hi), int((hi.eta_mean >= 0.8).sum()),
                int((hi.eta_mean <= 0).sum())),
        "spearman": float(rho),
        "extremes": (int((per.eta_max >= 0.99).sum()),
                     int((per.eta_min <= -0.99).sum())),
    }


def generation_accounting(fold: pd.DataFrame) -> dict:
    """What N-LDF actually generated: requested vs accepted, and the attempt log.

    Establishes from the archive alone that no 20 * n_min cap was applied and
    that the attempt cap was 5 * N_target (Section 5.2 and the provenance note
    of Section 5.4).
    """
    calls = (fold[fold.Method == "N-LDF"]
                 .drop_duplicates(["Dataset", "Repeat", "Fold"]).copy())
    calls["n_min"] = (calls.NTrain / (1 + calls.IR)).round().astype(int)
    calls["n_maj"] = calls.NTrain - calls.n_min
    calls["requested"] = (np.floor(0.75 * calls.n_maj) - calls.n_min).clip(lower=0)
    calls["at_target"] = calls.n_accepted == calls.requested
    calls["over_20x"] = calls.n_accepted > 20 * calls.n_min
    per = calls.groupby("Dataset").agg(at_target=("at_target", "all"),
                                       over_20x=("over_20x", "any"),
                                       attempts=("n_attempted", "sum"),
                                       requested=("requested", "sum"))
    chess = per.loc[["kr-vs-k-zero_vs_eight", "kr-vs-k-zero_vs_fifteen"]]
    tot = calls[["n_attempted", "n_accepted", "n_rejected_distance",
                 "n_rejected_maj", "n_rejected_density"]].sum().astype(int)
    return {
        "attempted": int(tot.n_attempted), "accepted": int(tot.n_accepted),
        "distance": int(tot.n_rejected_distance), "ratio": int(tot.n_rejected_maj),
        "factor": int(tot.n_rejected_density),
        "no_candidate": int(tot.n_attempted - tot.n_accepted - tot.n_rejected_distance
                            - tot.n_rejected_maj - tot.n_rejected_density),
        "datasets_at_target": int(per.at_target.sum()),
        "datasets_over_20x": int(per.over_20x.sum()),
        "chess_multiplier": float((chess.attempts / chess.requested).mean()),
    }


def fit_accounting(fold: pd.DataFrame) -> dict:
    g = fold.groupby("Dataset").agg(repeats=("Repeat", "nunique"),
                                    folds=("Fold", "nunique"),
                                    n=("F1", "size"))
    g["expected"] = g.repeats * g.folds * len(METHODS) * fold.Classifier.nunique()
    full = (fold.Dataset.nunique() * len(METHODS) * fold.Classifier.nunique() * 25)
    return {
        "completed": int(g.n.sum()),
        "full_design": int(full),
        "absent": int(full - g.n.sum()),
        "structural": int(full - g.expected.sum()),
        "runtime_failures": int(g.expected.sum() - g.n.sum()),
        "reduced_folds": g[g.folds != 5].index.tolist(),
        "reduced_repeats": g[g.repeats != 5].index.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/fold_level_results.csv", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="compare against the values printed in the manuscript")
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2

    fold = pd.read_csv(args.input)
    dmm = dataset_method_means(fold)
    pd.set_option("display.width", 130)

    acct = fit_accounting(fold)
    print("=" * 74)
    print("Section 5.3  --  fit accounting")
    print("=" * 74)
    print(f"  completed fits                  : {acct['completed']:,}")
    print(f"  full factorial design           : {acct['full_design']:,}")
    print(f"  absent                          : {acct['absent']:,}")
    print(f"    from reduced fold/repeat plan : {acct['structural']:,}")
    print(f"    individual runtime failures   : {acct['runtime_failures']:,}")
    print(f"  datasets with fewer than 5 folds  : {acct['reduced_folds']}")
    print(f"  datasets with fewer than 5 repeats: {acct['reduced_repeats']}")

    t3 = table3(dmm)
    print("\n" + "=" * 74)
    print("Table 8  --  global performance summary (equal weight per dataset)")
    print("=" * 74)
    print(t3.to_string())
    print(f"\n  highest F1        : {t3.F1.idxmax()} ({t3.F1.max():.4f})")
    print(f"  lowest mean rank  : {t3['Mean F1 Rank'].idxmin()} "
          f"({t3['Mean F1 Rank'].min():.4f})")
    os_f1 = t3.loc[OVERSAMPLERS, "F1"].sort_values(ascending=False)
    print(f"  best two on F1    : {os_f1.index[0]} - {os_f1.index[1]} "
          f"= {os_f1.iloc[0] - os_f1.iloc[1]:.4f}")
    print(f"  widest F1 spread among oversamplers = "
          f"{os_f1.max() - os_f1.min():.4f}")

    st = statistics(dmm)
    print("\n" + "=" * 74)
    print(f"Section 6.2  --  F1 tests (N = {st['n_datasets']} datasets)")
    print("=" * 74)
    c, p = st["friedman_all"]
    print(f"  Friedman, six strategies   : chi2 = {c:.4f}   p = {p:.4g}")
    c, p = st["friedman_oversamplers"]
    print(f"  Friedman, five oversamplers: chi2 = {c:.4f}   p = {p:.4f}")
    print(f"\n  planned family (N-LDF vs each; Wilcoxon, Pratt zeros, Holm over 5)")
    print(f"  {'N-LDF vs':<14}{'W':>9}{'raw p':>14}{'Holm p':>14}")
    for row in st["pairwise"]:
        print(f"  {row['vs']:<14}{row['W']:>9.1f}"
              f"{row['raw_p']:>14.4g}{row['holm_p']:>14.4g}")
    print(f"\n  exploratory family (all 15 pairs; Holm over 15)")
    print(f"  {'pair':<26}{'W':>9}{'raw p':>14}{'Holm p':>14}")
    for row in st["exploratory"]:
        print(f"  {row['a'] + ' vs ' + row['b']:<26}{row['W']:>9.1f}"
              f"{row['raw_p']:>14.4g}{row['holm_p']:>14.4g}")

    ga = generation_accounting(fold)
    print("\n" + "=" * 74)
    print("Section 5.2  --  what N-LDF generated (914 resampling calls)")
    print("=" * 74)
    print(f"  attempts {ga['attempted']:,}  accepted {ga['accepted']:,}  "
          f"rejected: distance {ga['distance']:,}, ratio {ga['ratio']:,}, "
          f"factor {ga['factor']:,}; no candidate {ga['no_candidate']:,}")
    print(f"  datasets where accepted == floor(0.75 n_maj) - n_min : {ga['datasets_at_target']}")
    print(f"  datasets where accepted  >  20 n_min (no cap applied) : {ga['datasets_over_20x']}")
    print(f"  attempts / requested on the chess tasks               : {ga['chess_multiplier']:.2f}")

    pd_ = paired_differences(dmm)
    print("\n" + "=" * 74)
    print("Figure 2  --  per-dataset paired F1 differences (N-LDF minus each)")
    print("=" * 74)
    print(f"  {'vs':<14}{'n':>4}{'win':>5}{'tie':>5}{'loss':>6}"
          f"{'median':>11}{'min':>10}{'max':>10}")
    for m, r in pd_.iterrows():
        print(f"  {m:<14}{int(r.n):>4}{int(r.win):>5}{int(r.tie):>5}"
              f"{int(r.loss):>6}{r['median']:>+11.4f}{r['min']:>+10.4f}"
              f"{r['max']:>+10.4f}")
    osr = pd_.drop(index="Baseline")
    print(f"  spread across the four oversampler comparisons: "
          f"{osr['min'].min():+.4f} .. {osr['max'].max():+.4f}")

    ed = eta_diagnostics(fold)
    print("\n" + "=" * 74)
    print(f"Section 4.3 / Table 2  --  observed eta ({ed['n_calls']} resampling calls)")
    print("=" * 74)
    print(f"  {'stratum':<9}{'datasets':>9}{'mean eta':>10}{'>=0.8':>7}{'<=0':>6}")
    for s, (n, m, a, b) in ed["strata"].items():
        print(f"  {s:<9}{n:>9}{m:>10.2f}{a:>7}{b:>6}")
    n, a, b = ed["ir9"]
    print(f"  IR >= 9   : {n} datasets, {a} with mean >= 0.8, {b} with mean <= 0")
    print(f"  Spearman(IR, mean eta) = {ed['spearman']:+.2f}")
    print(f"  datasets with per-call max >= 0.99: {ed['extremes'][0]}, "
          f"min <= -0.99: {ed['extremes'][1]}")
    per = ed["per_dataset"]
    low = per[per.acceptance < PAPER["acceptance_floor_others"]]
    print(f"  acceptance rate below {PAPER['acceptance_floor_others']}: "
          f"{low.index.tolist()}  (max accepted per fold: "
          f"{low.n_accepted_max.astype(int).tolist()})")

    vd = variance_decomposition(dmm)
    print("\n" + "=" * 74)
    print(f"Section 6.1  --  variance decomposition "
          f"({vd['n_datasets']} datasets x {vd['n_methods']} oversamplers)")
    print("=" * 74)
    for key in ("dataset", "method", "interaction"):
        print(f"  {key:<14}{vd[key]:>7.2f}%  of the total F1 sum of squares")

    print("\n" + "=" * 74)
    print("Table 21 (Appendix D)  --  per base classifier")
    print("=" * 74)
    print(table5(fold).to_string())

    t6 = table6(fold, dmm)
    print("\n" + "=" * 74)
    print("Table 11  --  mean dataset-level F1 by imbalance severity")
    print("=" * 74)
    print(t6.to_string())
    for s in STRATA:
        order = t6.loc[s, OVERSAMPLERS].sort_values(ascending=False)
        pos = list(order.index).index("N-LDF") + 1
        print(f"  {s:<8} best = {order.index[0]:<12} "
              f"N-LDF places {pos} of {len(OVERSAMPLERS)}")

    if not args.check:
        return 0

    print("\n" + "=" * 74)
    print(f"--check  --  agreement with the manuscript (tolerance {args.tol})")
    print("=" * 74)
    bad = []
    if acct["completed"] != PAPER["n_fits"]:
        bad.append(f"fit count {acct['completed']} != {PAPER['n_fits']}")
    cols = ["F1", "Precision", "G-Mean", "AUC-ROC", "AUPRC", "Mean F1 Rank"]
    for m, vals in PAPER["table3"].items():
        for col, want in zip(cols, vals):
            got = t3.loc[m, col]
            if abs(got - want) > args.tol:
                bad.append(f"Table 8 {m}/{col}: {got:.4f} != {want:.4f}")
    for s, vals in PAPER["table6"].items():
        for m, want in zip(METHODS, vals):
            got = t6.loc[s, m]
            if abs(got - want) > args.tol:
                bad.append(f"Table 11 {s}/{m}: {got:.4f} != {want:.4f}")
    if abs(st["friedman_all"][0] - PAPER["friedman_all"][0]) > 1e-3:
        bad.append("Friedman (six strategies) statistic disagrees")
    if abs(st["friedman_oversamplers"][0]
           - PAPER["friedman_oversamplers"][0]) > 1e-3:
        bad.append("Friedman (five oversamplers) statistic disagrees")
    holm = {r["vs"]: r["holm_p"] for r in st["pairwise"]}
    if abs(holm["Baseline"] - PAPER["holm_baseline"]) > 1e-9:
        bad.append(f"Holm p for N-LDF vs Baseline {holm['Baseline']:.4g} "
                   f"!= {PAPER['holm_baseline']:.4g}")
    base = [r for r in st["exploratory"] if "Baseline" in (r["a"], r["b"])]
    ovs = [r for r in st["exploratory"] if "Baseline" not in (r["a"], r["b"])]
    if abs(max(r["holm_p"] for r in base) - PAPER["exploratory_baseline_max_holm"]) > 5e-6:
        bad.append("exploratory family: largest baseline Holm p disagrees")
    if abs(min(r["holm_p"] for r in ovs) - PAPER["exploratory_oversampler_min_holm"]) > 5e-3:
        bad.append("exploratory family: smallest oversampler Holm p disagrees")
    for key, want in PAPER["attempt_totals"].items():
        if ga[key] != want:
            bad.append(f"Section 5.2 {key}: {ga[key]:,} != {want:,}")
    if ga["datasets_at_target"] != PAPER["datasets_at_target"]:
        bad.append(f"Section 5.2 datasets at target {ga['datasets_at_target']} != {PAPER['datasets_at_target']}")
    if ga["datasets_over_20x"] != PAPER["datasets_over_20x"]:
        bad.append(f"Section 5.2 datasets over 20x {ga['datasets_over_20x']} != {PAPER['datasets_over_20x']}")
    if abs(ga["chess_multiplier"] - PAPER["attempt_multiplier_chess"]) > 0.01:
        bad.append(f"Section 5.2 chess attempt multiplier {ga['chess_multiplier']:.2f} != 5")
    for m, (w, t, l, med) in PAPER["paired"].items():
        r = pd_.loc[m]
        if (int(r.win), int(r.tie), int(r.loss)) != (w, t, l):
            bad.append(f"Figure 2 {m}: win-tie-loss "
                       f"{int(r.win)}-{int(r.tie)}-{int(r.loss)} "
                       f"!= {w}-{t}-{l}")
        if abs(r["median"] - med) > args.tol:
            bad.append(f"Figure 2 {m}: median {r['median']:+.4f} != {med:+.4f}")
    lo, hi = PAPER["paired_range"]
    if abs(osr["min"].min() - lo) > 5e-3 or abs(osr["max"].max() - hi) > 5e-3:
        bad.append(f"Figure 2 spread {osr['min'].min():+.4f} .. "
                   f"{osr['max'].max():+.4f} != {lo:+.2f} .. {hi:+.2f}")
    for key, want in PAPER["variance"].items():
        if abs(vd[key] - want) > 0.005:
            bad.append(f"Section 6.1 variance share, {key}: "
                       f"{vd[key]:.2f}% != {want:.2f}%")
    for s, (n, m, a, b) in PAPER["eta_observed"].items():
        got = ed["strata"][s]
        if got[0] != n or abs(got[1] - m) > 0.005 or got[2] != a or got[3] != b:
            bad.append(f"Table 2 {s}: {got[0]}, {got[1]:.2f}, {got[2]}, {got[3]} "
                       f"!= {n}, {m:.2f}, {a}, {b}")
    if ed["ir9"] != PAPER["eta_ir9"]:
        bad.append(f"Section 4.3 IR>=9 counts {ed['ir9']} != {PAPER['eta_ir9']}")
    if abs(ed["spearman"] - PAPER["eta_spearman"]) > 0.005:
        bad.append(f"Section 4.3 Spearman {ed['spearman']:+.2f} "
                   f"!= {PAPER['eta_spearman']:+.2f}")
    if ed["extremes"] != PAPER["eta_extremes"]:
        bad.append(f"Section 4.3 extremes {ed['extremes']} != {PAPER['eta_extremes']}")
    if ed["n_calls"] != PAPER["n_calls"]:
        bad.append(f"Section 4.3 call count {ed['n_calls']} != {PAPER['n_calls']}")
    if sorted(low.index) != sorted(PAPER["zero_acceptance"]):
        bad.append(f"Section 5.2 low-acceptance datasets {low.index.tolist()} "
                   f"!= {PAPER['zero_acceptance']}")
    elif int(low.n_accepted_max.max()) > PAPER["max_accepted_chess"]:
        bad.append(f"Section 5.2 chess datasets accepted up to "
                   f"{int(low.n_accepted_max.max())} > {PAPER['max_accepted_chess']}")

    if bad:
        print("  MISMATCH:")
        for b in bad:
            print("   -", b)
        return 1
    print("  every checked value in the manuscript is reproduced by this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
