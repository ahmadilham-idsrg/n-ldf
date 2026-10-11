#!/usr/bin/env python3
"""Compare re-run records of the six archived strategies with the archive.

    python verify_reproduction.py runs/reference/units                     # unit files
    python verify_reproduction.py results/rerun/fold_level_results.csv.gz  # combined records
    python verify_reproduction.py results/rerun/fold_level_results.csv.gz --check-paper
        # exit 0 only if the comparison gives exactly the counts quoted in Sect. 5.6

For every (task, repeat, fold, strategy, classifier) present in both the
re-run unit files and results/fold_level_results.csv, reports the largest
absolute difference in each metric and the number of records that differ.
A re-run that matches with difference 0 on every record shows that the data
files, the environment and the harness reproduce the archived computation
exactly, so new strategies run in the same setting can be paired with the
archived records.
"""
from __future__ import annotations

import glob
import json
import sys

CHECK_PAPER = "--check-paper" in sys.argv
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["Dataset", "Repeat", "Fold", "Method", "Classifier"]
METRICS = ["F1", "Precision", "Recall", "Specificity", "G-Mean", "AUC-ROC", "AUPRC", "NResampled"]


def main(unit_dir: str) -> int:
    repo = Path(__file__).resolve().parent
    arch_path = repo / "results" / "fold_level_results.csv"
    if not arch_path.exists():
        arch_path = repo / "fold_level_results.csv"
    arch = pd.read_csv(arch_path)
    src = Path(unit_dir)
    if src.is_file():                       # a combined record file (csv or csv.gz)
        new = pd.read_csv(src, low_memory=False)
        out_dir = src.parent
    else:                                   # a directory of unit files written by run_units.py
        files = sorted(glob.glob(str(src / "*.csv")))
        new = pd.concat([pd.read_csv(f) for f in files if Path(f).stat().st_size > 2], ignore_index=True)
        out_dir = src.parent
    new = new[new["Method"].isin(arch["Method"].unique())]
    m = new.merge(arch, on=KEY, how="outer", suffixes=("_new", "_arch"), indicator=True)
    units = set(map(tuple, new[["Dataset", "Method"]].drop_duplicates().to_numpy()))
    in_scope = m[[(d, s) in units for d, s in zip(m["Dataset"], m["Method"])]]
    both = in_scope[in_scope["_merge"] == "both"]
    report = {"units": len(units), "records_rerun": int(len(new)),
              "records_matched": int(len(both)),
              "only_in_rerun": int((in_scope["_merge"] == "left_only").sum()),
              "only_in_archive": int((in_scope["_merge"] == "right_only").sum()),
              "metrics": {}}
    for c in METRICS:
        d = (both[c + "_new"].astype(float) - both[c + "_arch"].astype(float)).abs()
        report["metrics"][c] = {"max_abs_diff": float(d.max()) if len(d) else None,
                                "n_differ": int((d > 0).sum())}
    values_identical = all(v["n_differ"] == 0 for v in report["metrics"].values())
    exact = report["only_in_rerun"] == 0 and report["only_in_archive"] == 0 and values_identical
    report["values_identical_on_shared_records"] = bool(values_identical)
    report["exact"] = bool(exact)
    # slots present in only one of the two runs (expected: omitted fits, Appendix A)
    only = in_scope[in_scope["_merge"] != "both"][KEY].copy()
    only["present_in"] = in_scope.loc[only.index, "_merge"].astype(str).map(
        {"left_only": "rerun_only", "right_only": "archive_only"})
    only.to_csv(out_dir / "presence_differences.csv", index=False)
    print(json.dumps(report, indent=1))
    by = []
    for (ds, meth), g in both.groupby(["Dataset", "Method"]):
        diff = max(float((g[c + "_new"] - g[c + "_arch"]).abs().max()) for c in METRICS)
        by.append({"Dataset": ds, "Method": meth, "records": len(g), "max_abs_diff": diff})
    out = out_dir / "reproduction_check.csv"
    pd.DataFrame(by).to_csv(out, index=False)
    (out_dir / "reproduction_check.json").write_text(json.dumps(report, indent=1))
    print(f"per-unit table: {out}")
    # record-level identity (all metrics at once)
    diff_any = np.zeros(len(both), dtype=bool)
    for c in METRICS:
        if c == "NResampled":
            continue
        diff_any |= (both[c + "_new"].astype(float) - both[c + "_arch"].astype(float)).abs().to_numpy() > 0
    report["records_identical_in_every_metric"] = int((~diff_any).sum())
    report["records_differing"] = int(diff_any.sum())
    report["units_with_differing_records"] = sorted({f"{d}/{m}" for d, m in
                                                     both.loc[diff_any, ["Dataset", "Method"]].itertuples(index=False)})
    (out_dir / "reproduction_check.json").write_text(json.dumps(report, indent=1))
    if CHECK_PAPER:
        # the values quoted in Sect. 5.6 of the manuscript
        ok = (report["records_matched"] == 21885 and report["records_identical_in_every_metric"] == 21871
              and report["units_with_differing_records"] == ["segment0/N-LDF"]
              and max(v["max_abs_diff"] for k, v in report["metrics"].items() if k != "NResampled") < 0.015
              and report["only_in_rerun"] == 25 and report["only_in_archive"] == 26)
        print(f"re-run vs archive: {report['records_identical_in_every_metric']} of {report['records_matched']} shared "
              f"records identical in every metric; differing units {report['units_with_differing_records']}; "
              f"presence differences {report['only_in_rerun']}/{report['only_in_archive']} -> "
              f"{'agrees with the manuscript' if ok else 'DOES NOT agree with the manuscript'}")
        return 0 if ok else 1
    print("values identical on every shared record:", values_identical,
          f"| presence differences: {report['only_in_rerun']} re-run only, {report['only_in_archive']} archive only"
          " (see presence_differences.csv; replay them with replay_slots.py)")
    return 0 if values_identical else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0] if args else "runs/reference/units"))
