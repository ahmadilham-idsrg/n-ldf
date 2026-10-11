#!/usr/bin/env python3
"""Analysis of the held-out source evaluation, exactly as fixed in
heldout/PROTOCOL.md (SHA-256 in heldout/protocol.sha256).

    python heldout_analysis.py runs/heldout/fold_level_results.csv --outdir results/heldout

Unit of inference: the source dataset (task scores averaged within source).
Primary: paired Wilcoxon-Pratt on source-level values, Holm within family;
source-cluster bootstrap 95% interval of the mean task-level difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import extension_analysis as ea

ARCHIVED_FAMILY = [("N-LDF", m) for m in ["SMOTE", "B-SMOTE", "ADASYN", "SMOTE-Tomek", "Baseline"]]
FAMILIES = {"archived": ARCHIVED_FAMILY,
            **{k: ea.FAMILIES[k] for k in ["amount", "mechanism_r0.75", "mechanism_parity", "threshold"]}}
SIX = ["Baseline", "SMOTE", "B-SMOTE", "ADASYN", "SMOTE-Tomek", "N-LDF"]


def analyse(fold: pd.DataFrame, outdir: Path, boot: int = ea.B_BOOT, seed: int = ea.SEED) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    proto = Path(__file__).resolve().parent / "heldout" / "PROTOCOL.md"
    res: dict = {"protocol_sha256": hashlib.sha256(proto.read_bytes()).hexdigest() if proto.exists() else None}
    dmm = ea.dataset_method_means(fold)
    dmm["Source"] = dmm.Dataset.map(ea.source_of)
    res["n_tasks"] = int(dmm.Dataset.nunique())
    res["n_sources"] = int(dmm.Source.nunique())
    res["records"] = int(len(fold))

    task_means = dmm.groupby("Method")[[m for m in ea.METRICS if m in dmm]].mean()
    src = dmm.groupby(["Source", "Method"])[[m for m in ea.METRICS if m in dmm]].mean().reset_index()
    src_means = src.groupby("Method")[[m for m in ea.METRICS if m in src]].mean()
    task_means.round(4).to_csv(outdir / "strategy_means_task_level.csv")
    src_means.round(4).to_csv(outdir / "strategy_means_source_level.csv")
    res["source_level_means"] = src_means.round(6).to_dict(orient="index")
    res["task_level_means"] = task_means.round(6).to_dict(orient="index")

    # Friedman over the six archived strategies on source-level F1
    w = src.pivot(index="Source", columns="Method", values="F1")[SIX].dropna()
    chi, p = stats.friedmanchisquare(*[w[m] for m in SIX])
    w5 = w[[m for m in SIX if m != "Baseline"]]
    chi5, p5 = stats.friedmanchisquare(*[w5[m] for m in w5.columns])
    ranks = w.rank(axis=1, ascending=False).mean()
    res["friedman_six"] = {"chi2": float(chi), "p": float(p), "n_sources": int(len(w))}
    res["friedman_oversamplers"] = {"chi2": float(chi5), "p": float(p5)}
    res["mean_rank_source_level"] = ranks.round(3).to_dict()

    families = {}
    flat = []
    for fam, pairs in FAMILIES.items():
        families[fam] = {}
        for metric in ["F1", "AUPRC", "Recall", "Specificity"]:
            x = ea.wide(dmm, metric)
            rows = ea.run_family(x, pairs, boot, seed)
            families[fam][metric] = rows
            for r in rows:
                flat.append({"family": fam, "metric": metric, **r})
    res["families"] = families
    pd.DataFrame(flat).round(6).to_csv(outdir / "contrasts.csv", index=False)

    lines = ["# Held-out source evaluation", "",
             f"Protocol SHA-256: `{res['protocol_sha256']}`", "",
             f"{res['n_tasks']} tasks, {res['n_sources']} sources, {res['records']:,} fit records.", "",
             "## Source-level means", "", src_means.round(4).to_markdown(), "",
             f"Friedman (six archived strategies, source-level F1): chi2 = {chi:.2f}, p = {p:.3g}; "
             f"five oversamplers: chi2 = {chi5:.2f}, p = {p5:.3g}.", "",
             "Mean F1 rank (source level): " + ", ".join(f"{k} {v:.2f}" for k, v in ranks.sort_values().items()), ""]
    for fam, per in families.items():
        lines += [f"## Family `{fam}`", "",
                  "| metric | contrast (a − b) | mean diff | W–T–L (tasks) | source Holm p | task Holm p | cluster 95% interval |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for metric in ["F1", "AUPRC"]:
            for r in per.get(metric, []):
                lines.append(f"| {metric} | {r['a']} − {r['b']} | {r['mean_diff']:+.4f} | {r['win']}–{r['tie']}–{r['loss']} | "
                             f"{ea.fmt_p(r['source_holm_p'])} | {ea.fmt_p(r['holm_p'])} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
        lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (outdir / "heldout_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("results/heldout"))
    ap.add_argument("--boot", type=int, default=ea.B_BOOT)
    a = ap.parse_args()
    fold = pd.read_csv(a.records, low_memory=False)
    analyse(fold, a.outdir, boot=a.boot)
    print((a.outdir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
