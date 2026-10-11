#!/usr/bin/env python3
"""Run one extension package headlessly, outside Colab.

    python run_extension.py --package matched            # full 41-task run, resumable
    python run_extension.py --package comparators --smoke # three tasks, two repeats
    python run_extension.py --package matched --synthetic # code check on random data

It executes the code cells of notebooks/nldf_benchmark_extension.ipynb (Cells
1-5 verbatim, then Cell 6 with PACKAGE / SMOKE_TEST set from the command
line) in a fresh namespace, so a run from here is the same computation as a
run in Colab: same loader (KEEL zips are read from keel_cache/, downloaded if
absent), same partitions, preprocessing, classifiers, seeds and checkpoint.
Output: nldf_results_41_<package>/fold_level_results.csv (+ dataset_audit.csv
with the zip hashes); checkpoint_<package>.pkl lets an interrupted run resume.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
NB = REPO / "notebooks" / "nldf_benchmark_extension.ipynb"
if not NB.exists():
    NB = REPO / "nldf_benchmark_extension.ipynb"
SMOKE_LINE = 'SMOKE_DATASETS = ["glass1", "winequality-white-3_vs_7", "kddcup-land_vs_portsweep"]'


def code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def run_cell(ns: dict, text: str, name: str) -> None:
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("!") and "= !" not in l]
    text = "\n".join(lines)
    ast.parse(text)
    exec(compile(text, name, "exec"), ns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True,
                    choices=["matched", "matched_all", "ablation_statistic", "ablation_components", "comparators", "sensitivity"])
    ap.add_argument("--smoke", action="store_true", help="three smoke-test tasks, two repeats")
    ap.add_argument("--clean-start", action="store_true", help="delete the package's checkpoint and output first")
    ap.add_argument("--synthetic", action="store_true", help="replace the KEEL loader by random data (code check only)")
    args = ap.parse_args()

    import os
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    cells = code_cells(NB)
    assert len(cells) == 8 and cells[6].startswith('"""Cell 6: run one package'), "unexpected notebook layout"
    ns = {"__name__": "__main__"}
    t0 = time.time()
    for i in (1, 2, 3, 4, 5):
        run_cell(ns, cells[i], f"cell{i}")
        if "CheckpointState" in ns:
            sys.modules["__main__"].CheckpointState = ns["CheckpointState"]   # picklable checkpoint
    print(f"[run_extension] harness loaded in {time.time() - t0:.1f}s; package={args.package} smoke={args.smoke}")

    run = cells[6]
    run = run.replace('PACKAGE = "matched"', f'PACKAGE = "{args.package}"')
    if args.smoke:
        run = run.replace("SMOKE_TEST = False", "SMOKE_TEST = True")
    if args.clean_start:
        run = run.replace("CLEAN_START = False", "CLEAN_START = True")
    if args.synthetic:
        import numpy as np
        import pandas as pd
        LoadedDataset, sev = ns["LoadedDataset"], ns["severity_from_ir"]

        def fake(name, url, session, cache_dir):
            rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
            n, n_min, d = 240, 40, 4
            X = rng.normal(size=(n, d)); y = np.zeros(n, dtype=int); y[:n_min] = 1; X[:n_min] += 1.5
            ir = (n - n_min) / n_min
            return LoadedDataset(name=name, X=pd.DataFrame(X), y=y, numeric_columns=list(range(d)),
                                 categorical_columns=[], n_samples=n, n_features=d, n_minority=n_min,
                                 n_majority=n - n_min, imbalance_ratio=ir, severity=sev(ir),
                                 source_url="synthetic", sha256="", is_problematic=False)
        ns["load_keel_dataset"] = fake
        run = run.replace(SMOKE_LINE, 'SMOKE_DATASETS = ["glass1", "pima", "yeast1"]')
        run = run.replace('"n_folds": 5,', '"n_folds": 2,').replace('"n_repeats": 2 if SMOKE_TEST else 5,', '"n_repeats": 1,')
        run = run.replace('f"nldf_results_41_{PACKAGE}"', 'f"synthetic_results_{PACKAGE}"')
        run = run.replace('f"checkpoint_{PACKAGE}.pkl"', 'f"synthetic_checkpoint_{PACKAGE}.pkl"')
        if "SMOKE_TEST = True" not in run:
            run = run.replace("SMOKE_TEST = False", "SMOKE_TEST = True")
    run_cell(ns, run, "cell6_run")
    print(f"[run_extension] done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
