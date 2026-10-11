#!/usr/bin/env python3
"""Analysis of the extension experiments (matched amount, threshold-tuned
baseline, ablations) together with the archived benchmark records.

The archived strategies keep their archived rows; the extension packages add
new strategies evaluated on the same partitions. Aggregation and inference
follow the manuscript: one score per dataset-strategy block (direct mean of
all fold-level fits across the four classifiers), paired Wilcoxon-Pratt tests
with Holm correction within each planned family, and a source-cluster
bootstrap (15 KEEL source datasets, 10,000 resamples) for every contrast.

Planned families (F1 is the primary outcome; the other metrics are reported
with their own Holm correction):

  amount      parity minus r = 0.75 within a mechanism:
              N-LDF-r1.0 - N-LDF, SMOTE - SMOTE-r0.75, ROS-r1.0 - ROS-r0.75
  mechanism   at matched amount:
              r = 0.75: N-LDF - SMOTE-r0.75, N-LDF - ROS-r0.75, SMOTE-r0.75 - ROS-r0.75
              parity  : N-LDF-r1.0 - SMOTE, N-LDF-r1.0 - ROS-r1.0, SMOTE - ROS-r1.0
  threshold   Baseline-thr - Baseline, N-LDF - Baseline-thr, SMOTE - Baseline-thr,
              N-LDF-r1.0 - Baseline-thr
  ablation    each ABL-* variant minus N-LDF (Holm over the variants present)

Usage
  python extension_analysis.py --archived results/fold_level_results.csv \
      nldf_results_41_matched/fold_level_results.csv [more package CSVs] \
      --outdir results/extension
  python extension_analysis.py --selftest
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["F1", "Precision", "Recall", "Specificity", "G-Mean", "AUC-ROC", "AUPRC"]
SEED = 42
B_BOOT = 10_000
ARCHIVED = ["Baseline", "ADASYN", "B-SMOTE", "SMOTE", "SMOTE-Tomek", "N-LDF"]

FAMILIES = {
    "amount": [("N-LDF-r1.0", "N-LDF"), ("SMOTE", "SMOTE-r0.75"), ("ROS-r1.0", "ROS-r0.75")],
    "mechanism_r0.75": [("N-LDF", "SMOTE-r0.75"), ("N-LDF", "ROS-r0.75"), ("SMOTE-r0.75", "ROS-r0.75")],
    "mechanism_parity": [("N-LDF-r1.0", "SMOTE"), ("N-LDF-r1.0", "ROS-r1.0"), ("SMOTE", "ROS-r1.0")],
    "threshold": [("Baseline-thr", "Baseline"), ("N-LDF", "Baseline-thr"), ("SMOTE", "Baseline-thr"),
                  ("N-LDF-r1.0", "Baseline-thr")],
    # class weighting without resampling (package "comparators")
    "weighting": [("Baseline-cw", "Baseline"), ("N-LDF", "Baseline-cw"), ("SMOTE", "Baseline-cw"),
                  ("Baseline-thr", "Baseline-cw")],
    # the nearest safety-, boundary- and density-guided oversamplers, all at parity
    "comparators": [("N-LDF", "SLS"), ("N-LDF", "SVM-SMOTE"), ("N-LDF", "KMeans-SMOTE"), ("SLS", "SMOTE")],
    # the same comparators against N-LDF at the same (parity) target, added before
    # any extension result was seen (3 Oct 2026), since N-LDF itself targets r = 0.75
    "comparators_parity": [("N-LDF-r1.0", "SLS"), ("N-LDF-r1.0", "SVM-SMOTE"), ("N-LDF-r1.0", "KMeans-SMOTE")],
    # every other oversampler at both targets (package "matched_all"; added 3 Oct 2026,
    # before any extension result was seen)
    "amount_all": [("B-SMOTE", "B-SMOTE-r0.75"), ("ADASYN", "ADASYN-r0.75"), ("SMOTE-Tomek", "SMOTE-Tomek-r0.75"),
                   ("SLS", "SLS-r0.75"), ("SVM-SMOTE", "SVM-SMOTE-r0.75"), ("KMeans-SMOTE", "KMeans-SMOTE-r0.75")],
    "mechanism_r0.75_all": [("N-LDF", "B-SMOTE-r0.75"), ("N-LDF", "ADASYN-r0.75"), ("N-LDF", "SMOTE-Tomek-r0.75"),
                            ("N-LDF", "SLS-r0.75"), ("N-LDF", "SVM-SMOTE-r0.75"), ("N-LDF", "KMeans-SMOTE-r0.75")],
    # N-LDF against all eight other mechanisms at the same target, one Holm family per
    # target. Added after the first matched-amount results were available, as the
    # union of the prespecified mechanism and comparator contrasts plus N-LDF-r1.0
    # against the three remaining archived oversamplers; used for the mechanism table.
    "mechanism_r0.75_complete": [("N-LDF", m) for m in
                                 ["ROS-r0.75", "SMOTE-r0.75", "B-SMOTE-r0.75", "ADASYN-r0.75", "SMOTE-Tomek-r0.75",
                                  "SLS-r0.75", "SVM-SMOTE-r0.75", "KMeans-SMOTE-r0.75"]],
    "amount_complete": [("ROS-r1.0", "ROS-r0.75"), ("SMOTE", "SMOTE-r0.75"), ("B-SMOTE", "B-SMOTE-r0.75"),
                        ("ADASYN", "ADASYN-r0.75"), ("SMOTE-Tomek", "SMOTE-Tomek-r0.75"), ("SLS", "SLS-r0.75"),
                        ("SVM-SMOTE", "SVM-SMOTE-r0.75"), ("KMeans-SMOTE", "KMeans-SMOTE-r0.75"),
                        ("N-LDF-r1.0", "N-LDF")],
    "mechanism_parity_complete": [("N-LDF-r1.0", m) for m in
                                  ["ROS-r1.0", "SMOTE", "B-SMOTE", "ADASYN", "SMOTE-Tomek",
                                   "SLS", "SVM-SMOTE", "KMeans-SMOTE"]],
}
ABLATION_PREFIX = "ABL-"
SENSITIVITY_PREFIX = "SENS-"


# --------------------------------------------------------------------------
def source_of(name: str) -> str:
    for pfx in ["yeast", "abalone", "ecoli", "kddcup", "winequality", "glass", "poker", "kr-vs-k",
                "car-", "new_thyroid", "page-blocks"]:   # last three: held-out sources
        if name.startswith(pfx):
            return {"car-": "car", "new_thyroid": "new-thyroid"}.get(pfx, pfx)
    return name


def holm(rows: list[dict], key: str = "raw_p") -> list[dict]:
    rows = sorted(rows, key=lambda r: r[key])
    k, running = len(rows), 0.0
    for i, r in enumerate(rows):
        running = max(running, min(1.0, r[key] * (k - i)))
        r["holm_p"] = running
    return rows


def wilcoxon(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 0.0, 1.0
    w, p = stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided", correction=False)
    return float(w), float(p)


def dataset_method_means(fold: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METRICS + ["AchievedMinMajRatio"] if c in fold.columns]
    return fold.groupby(["Dataset", "Method"], as_index=False)[cols].mean()


def wide(dmm: pd.DataFrame, metric: str) -> pd.DataFrame:
    return dmm.pivot(index="Dataset", columns="Method", values=metric)


def contrast(x: pd.DataFrame, a: str, b: str, boot: int, seed: int) -> dict:
    """Paired contrast a - b on the datasets where both exist: task-level and
    source-block Wilcoxon-Pratt, source-cluster bootstrap interval."""
    d = (x[a] - x[b]).dropna()
    src = pd.Series({t: source_of(t) for t in d.index})
    _, p_task = wilcoxon(d, np.zeros(len(d)))
    d_src = d.groupby(src).mean()
    _, p_src = wilcoxon(d_src, np.zeros(len(d_src)))
    groups = {s: d.index[src == s].tolist() for s in src.unique()}
    names = list(groups)
    # per-source sums and task counts: the mean over the tasks of the drawn sources
    # (each draw entering with all its tasks) is sum(draws * sums) / sum(draws * counts)
    sums = np.array([d.loc[groups[s]].sum() for s in names])
    counts = np.array([len(groups[s]) for s in names], dtype=float)
    index = {s: i for i, s in enumerate(names)}
    rng = np.random.default_rng(seed)
    bs = np.empty(boot)
    for i in range(boot):
        pick = rng.choice(names, size=len(names), replace=True)
        w = np.bincount([index[s] for s in pick], minlength=len(names))
        bs[i] = (w * sums).sum() / (w * counts).sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"a": a, "b": b, "n": int(len(d)), "mean_diff": float(d.mean()),
            "median_diff": float(d.median()),
            "win": int((d > 1e-12).sum()), "tie": int((np.abs(d) <= 1e-12).sum()), "loss": int((d < -1e-12).sum()),
            "raw_p": p_task, "source_raw_p": p_src, "n_sources": int(len(d_src)),
            "ci_lo": float(lo), "ci_hi": float(hi), "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def run_family(x: pd.DataFrame, pairs: list[tuple[str, str]], boot: int, seed: int) -> list[dict]:
    rows = [contrast(x, a, b, boot, seed) for a, b in pairs if a in x.columns and b in x.columns]
    if not rows:
        return rows
    rows = holm(rows, "raw_p")
    # Holm for the source-level p as well
    tmp = holm([{"raw_p": r["source_raw_p"], "_i": i} for i, r in enumerate(rows)], "raw_p")
    for t in tmp:
        rows[t["_i"]]["source_holm_p"] = t["holm_p"]
    return rows


def fmt_p(p: float) -> str:
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


# --------------------------------------------------------------------------
def analyse(fold: pd.DataFrame, outdir: Path, boot: int = B_BOOT, seed: int = SEED) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    dmm = dataset_method_means(fold)
    present = sorted(dmm.Method.unique())
    results: dict = {"strategies": present, "n_rows": int(len(fold))}

    # ---- means per strategy (all metrics + realized amount)
    means = dmm.groupby("Method")[[c for c in METRICS + ["AchievedMinMajRatio"] if c in dmm.columns]].mean()
    means["n_datasets"] = dmm.groupby("Method").Dataset.nunique()
    means.round(4).to_csv(outdir / "strategy_means.csv")
    results["means"] = means.round(6).to_dict(orient="index")

    # ---- realized amounts and rejection accounting for N-LDF variants
    if "AchievedMinMajRatio" in fold.columns:
        amt = fold.drop_duplicates(["Dataset", "Repeat", "Fold", "Method"]).groupby("Method")[
            ["AchievedMinMajRatio"]].agg(["mean", "min", "max"])
        amt.round(4).to_csv(outdir / "realized_amount.csv")
    rej_cols = [c for c in ["n_attempted", "n_accepted", "n_rejected_distance", "n_rejected_density",
                            "n_rejected_maj"] if c in fold.columns]
    if rej_cols:
        calls = fold.drop_duplicates(["Dataset", "Repeat", "Fold", "Method"])
        rej = calls[calls.Method.str.startswith(("N-LDF", "ABL-", "SENS-"))].groupby("Method")[rej_cols].sum()
        rej.to_csv(outdir / "rejections_by_variant.csv")
        results["rejections_by_variant"] = rej.to_dict(orient="index")
    if "SamplerStatus" in fold.columns:
        calls = fold.drop_duplicates(["Dataset", "Repeat", "Fold", "Method"])
        fb = calls.groupby("Method").SamplerStatus.apply(lambda s: int((s == "fallback").sum())).rename("n_fallback")
        fb = pd.concat([fb, calls.groupby("Method").size().rename("n_calls")], axis=1)
        fb.to_csv(outdir / "fallbacks_by_strategy.csv")
        results["fallbacks_by_strategy"] = fb.to_dict(orient="index")

    # ---- planned families, per metric
    families = {}
    for fam, pairs in FAMILIES.items():
        families[fam] = {}
        for metric in METRICS:
            x = wide(dmm, metric)
            rows = run_family(x, pairs, boot, seed)
            if rows:
                families[fam][metric] = rows
    abl = [m for m in present if m.startswith(ABLATION_PREFIX)]
    if abl and "N-LDF" in present:
        families["ablation"] = {}
        for metric in METRICS:
            x = wide(dmm, metric)
            families["ablation"][metric] = run_family(x, [(v, "N-LDF") for v in abl], boot, seed)
    sens = [m for m in present if m.startswith(SENSITIVITY_PREFIX)]
    if sens and "N-LDF" in present:
        families["sensitivity"] = {}
        for metric in METRICS:
            x = wide(dmm, metric)
            families["sensitivity"][metric] = run_family(x, [(v, "N-LDF") for v in sens], boot, seed)
    results["families"] = families

    # ---- flat CSV of every contrast
    flat = []
    for fam, per_metric in families.items():
        for metric, rows in per_metric.items():
            for r in rows:
                flat.append({"family": fam, "metric": metric, **r})
    if flat:
        pd.DataFrame(flat).round(6).to_csv(outdir / "contrasts.csv", index=False)

    # ---- markdown summary
    lines = ["# Extension experiments: summary", "",
             f"Strategies present: {', '.join(present)}  (rows: {len(fold):,})", "",
             "## Strategy means (dataset-level, equal weight over datasets)", "",
             means.round(4).to_markdown(), ""]
    for fam, per_metric in families.items():
        if "F1" not in per_metric:
            continue
        lines += [f"## Family `{fam}`", ""]
        for metric in ["F1", "Recall", "Specificity", "AUPRC"]:
            rows = per_metric.get(metric, [])
            if not rows:
                continue
            lines += [f"**{metric}**", "",
                      "| contrast (a − b) | n | mean diff | W–T–L | task Holm p | source Holm p | cluster 95% interval |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                lines.append(f"| {r['a']} − {r['b']} | {r['n']} | {r['mean_diff']:+.4f} | {r['win']}–{r['tie']}–{r['loss']} | "
                             f"{fmt_p(r['holm_p'])} | {fmt_p(r['source_holm_p'])} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
                             f"{' *' if r['ci_excludes_zero'] else ''} |")
            lines.append("")
    lines += ["Holm correction is within each family and metric. `*` marks a source-cluster",
              "bootstrap interval that excludes zero (no multiplicity correction).", ""]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (outdir / "extension_results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    return results


def load_inputs(archived: Path, packages: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(archived, low_memory=False)]
    frames[0] = frames[0][frames[0].Method.isin(ARCHIVED)]
    for p in packages:
        f = pd.read_csv(p, low_memory=False)
        f = f[~f.Method.isin(ARCHIVED)]  # archived strategies keep their archived rows
        frames.append(f)
    fold = pd.concat(frames, ignore_index=True, sort=False)
    # realized minority/majority ratio for the archived strategies: the archived
    # N-LDF logged it (achieved_ratio); the parity oversamplers are 1.0 by
    # construction before cleaning; the baseline and SMOTE-Tomek are left NaN
    if "AchievedMinMajRatio" not in fold.columns:
        fold["AchievedMinMajRatio"] = np.nan
    arch = fold.Method.isin(ARCHIVED) & fold.AchievedMinMajRatio.isna()
    if "achieved_ratio" in fold.columns:
        m = arch & (fold.Method == "N-LDF")
        fold.loc[m, "AchievedMinMajRatio"] = fold.loc[m, "achieved_ratio"]
    fold.loc[arch & fold.Method.isin(["SMOTE", "B-SMOTE", "ADASYN"]), "AchievedMinMajRatio"] = 1.0
    return fold


# --------------------------------------------------------------------------
def selftest(outdir: Path) -> None:
    """Synthetic fold-level records with the archive's column layout, to
    exercise every code path (not a result)."""
    rng = np.random.default_rng(0)
    tasks = ["glass1", "pima", "iris0", "yeast1", "yeast3", "ecoli1", "ecoli2", "abalone9-18",
             "abalone19", "kddcup-land_vs_satan", "kr-vs-k-zero_vs_eight", "poker-8_vs_6",
             "winequality-red-3_vs_5", "shuttle-2_vs_5", "vowel0", "segment0"]
    strategies = ARCHIVED + ["N-LDF-r1.0", "SMOTE-r0.75", "ROS-r0.75", "ROS-r1.0", "Baseline-thr",
                             "ABL-lp0", "ABL-lp1", "ABL-localprop", "ABL-uniform",
                             "ABL-nojitter", "ABL-nodist", "ABL-nosupport", "ABL-noratio",
                             "Baseline-cw", "SLS", "SVM-SMOTE", "KMeans-SMOTE",
                             "SENS-lp0.5", "SENS-kd11", "SENS-acc+0.1"]
    rows = []
    for t in tasks:
        base = rng.uniform(0.3, 0.8)
        for m in strategies:
            shift = {"Baseline": -0.08, "Baseline-thr": -0.02, "Baseline-cw": -0.03}.get(m, 0.0) + rng.normal(0, 0.01)
            for rep_ in range(2):
                for fold_ in range(3):
                    for clf in ["RF", "SVM-Linear", "SVM-RBF", "XGBoost"]:
                        f1 = float(np.clip(base + shift + rng.normal(0, 0.05), 0, 1))
                        row = {"Dataset": t, "Severity": "Mild", "IR": 5.0, "Repeat": rep_, "Fold": fold_,
                               "Method": m, "Classifier": clf, "NTrain": 100, "NTest": 25, "NResampled": 150,
                               "F1": f1, "Precision": f1, "Recall": f1, "Specificity": 1 - f1 / 3,
                               "G-Mean": f1, "AUC-ROC": 0.5 + f1 / 2, "AUPRC": f1,
                               "AchievedMinMajRatio": 1.0 if "r1.0" in m or m in ("SMOTE", "ADASYN", "B-SMOTE", "SMOTE-Tomek", "SLS", "SVM-SMOTE", "KMeans-SMOTE") else 0.75,
                               "SamplerStatus": "fallback" if (m == "KMeans-SMOTE" and t.startswith("abalone")) else "success"}
                        if m.startswith(("N-LDF", "ABL-", "SENS-")):
                            row.update({"n_attempted": 120, "n_accepted": 100, "n_rejected_distance": 5,
                                        "n_rejected_density": 10, "n_rejected_maj": 5})
                        rows.append(row)
    fold = pd.DataFrame(rows)
    res = analyse(fold, outdir, boot=200, seed=1)
    assert "amount" in res["families"] and "ablation" in res["families"]
    assert "weighting" in res["families"] and "comparators" in res["families"]
    assert "sensitivity" in res["families"] and len(res["families"]["sensitivity"]["F1"]) == 3
    assert res["fallbacks_by_strategy"]["KMeans-SMOTE"]["n_fallback"] > 0
    assert len(res["families"]["ablation"]["F1"]) == 8
    assert (outdir / "summary.md").exists() and (outdir / "contrasts.csv").exists()
    print("extension_analysis selftest: OK ->", outdir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packages", nargs="*", type=Path, help="fold_level_results.csv of each extension package")
    ap.add_argument("--archived", type=Path, default=Path("results/fold_level_results.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("results/extension"))
    ap.add_argument("--boot", type=int, default=B_BOOT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(Path("/tmp/extension_selftest") if a.outdir == Path("results/extension") else a.outdir)
        return 0
    if not a.archived.exists():
        print(f"error: {a.archived} not found", file=sys.stderr)
        return 2
    if not a.packages:
        print("error: give at least one package fold_level_results.csv", file=sys.stderr)
        return 2
    fold = load_inputs(a.archived, a.packages)
    res = analyse(fold, a.outdir, boot=a.boot)
    print((a.outdir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
