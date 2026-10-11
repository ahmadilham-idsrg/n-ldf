#!/usr/bin/env python3
"""Collect the unit files written by run_units.py into one record file per run.

    python combine_units.py runs/matched            # -> runs/matched/fold_level_results.csv
                                                    #    runs/matched/unit_log.csv

unit_log.csv lists every (task, strategy) unit with its status, number of
records and wall-clock seconds (two single-threaded worker processes on a
2-vCPU machine; the times are for orientation, not a benchmark).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd


def combine(run: Path) -> pd.DataFrame:
    units = run / "units"
    frames, log = [], []
    for meta_path in sorted(glob.glob(str(units / "*.json"))):
        meta = json.loads(Path(meta_path).read_text())
        csv = Path(meta_path).with_suffix(".csv")
        if csv.exists() and csv.stat().st_size > 2:
            frames.append(pd.read_csv(csv, low_memory=False))
        log.append({k: meta.get(k) for k in ("task", "method", "status", "rows", "seconds", "sha256", "error")})
    fold = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    fold.to_csv(run / "fold_level_results.csv", index=False)
    lg = pd.DataFrame(log)
    lg.to_csv(run / "unit_log.csv", index=False)
    return lg


if __name__ == "__main__":
    for r in sys.argv[1:]:
        lg = combine(Path(r))
        print(f"{r}: {len(lg)} units, {int(lg.rows.sum()) if len(lg) else 0} records, "
              f"status {lg.status.value_counts().to_dict() if len(lg) else {}}, "
              f"{lg.seconds.sum() / 3600 if len(lg) else 0:.2f} worker-hours")
