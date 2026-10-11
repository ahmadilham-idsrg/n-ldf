#!/usr/bin/env python3
"""End-to-end test of the two re-runnable notebooks on synthetic data.

    python tests/test_notebooks_synthetic.py            # from the repository root
    python tests/test_notebooks_synthetic.py --quick    # skip extension_analysis

It executes the code cells of ``notebooks/nldf_benchmark_reproducible.ipynb``
and ``notebooks/nldf_benchmark_extension.ipynb`` headlessly, with the KEEL
loader replaced by three small synthetic datasets, and asserts that

1. the clean notebook runs the six archived strategies and writes fold-level
   records with the archived column set;
2. each extension package (``matched``, ``matched_all``, ``ablation_statistic``,
   ``ablation_components``, ``comparators``) runs on the same datasets;
3. every package reproduces the clean notebook's partitions exactly (same
   NTrain/NTest per dataset, repeat and fold), so extension results are paired
   with the archived ones;
4. the tuned baseline logs a threshold in [0, 1], ``N-LDF-r1.0`` runs at
   sampling ratio 1.0, random oversampling reaches its requested ratio, and the
   component ablations switch off the test they are named after, the
   comparators reach parity or record a fallback, and the weighted baseline
   runs on the unresampled partition;
5. ``extension_analysis.py`` merges the records and writes its outputs.

This is a test of the code, not an experiment: the datasets are random and the
numbers it produces mean nothing. It takes a few minutes on a laptop and does
not touch the network.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks" if (REPO / "notebooks").is_dir() else REPO
POST = REPO / "src" / "nldf_postprocessing.py"
if not POST.exists():
    POST = REPO / "nldf_postprocessing.py"
QUICK = "--quick" in sys.argv

SMOKE_LINE = 'SMOKE_DATASETS = ["glass1", "winequality-white-3_vs_7", "kddcup-land_vs_portsweep"]'
SYNTH_LINE = 'SMOKE_DATASETS = ["yeast1", "abalone19", "glass1"]'


def code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def run_cell(ns: dict, text: str, name: str) -> None:
    """Execute one code cell, dropping Colab shell lines (``!pip``, ``x = !cmd``)."""
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("!") and "= !" not in l]
    text = "\n".join(lines)
    ast.parse(text)
    exec(compile(text, name, "exec"), ns)


def shrink(cell: str) -> str:
    """Smoke-test settings with 2 folds x 1 repeat on the three synthetic tasks."""
    assert SMOKE_LINE in cell, "smoke-test dataset list not found in the run cell"
    cell = cell.replace("SMOKE_TEST = False", "SMOKE_TEST = True").replace(SMOKE_LINE, SYNTH_LINE)
    cell = cell.replace('"n_folds": 5,', '"n_folds": 2,')
    cell = cell.replace('"n_repeats": 2 if SMOKE_TEST else 5,', '"n_repeats": 1,')
    return cell


def synthetic_datasets(ns: dict) -> dict:
    LoadedDataset, severity = ns["LoadedDataset"], ns["severity_from_ir"]

    def make(name, n, n_min, d, seed, categorical):
        rng = np.random.default_rng(seed)
        X = rng.normal(0.0, 1.0, (n, d))
        y = np.zeros(n, dtype=int)
        y[:n_min] = 1
        X[:n_min] += 1.5
        frame = pd.DataFrame(X)
        cat = []
        if categorical:
            frame[d] = rng.choice(["a", "b", "c"], size=n)
            cat = [d]
        ir = (n - n_min) / n_min
        return LoadedDataset(
            name=name, X=frame, y=y, numeric_columns=list(range(d)), categorical_columns=cat,
            n_samples=n, n_features=frame.shape[1], n_minority=n_min, n_majority=n - n_min,
            imbalance_ratio=ir, severity=severity(ir), source_url="synthetic", sha256="",
            is_problematic=False,
        )

    return {
        "yeast1": make("yeast1", 240, 60, 4, 1, False),
        "abalone19": make("abalone19", 400, 12, 5, 2, True),
        "glass1": make("glass1", 200, 70, 3, 3, False),
    }


def report_dropped(df: pd.DataFrame, n_methods: int, n_datasets: int = 3, n_folds: int = 2,
                   n_classifiers: int = 4, tolerance: int = 2) -> None:
    """The harness silently drops a comparison fit whose classifier raises (the
    archive lost 25 of 24,600 slots this way; manuscript, Sect. 5.4). Report
    any such slot here instead of hiding it, and fail if more than a couple go."""
    expected = n_datasets * n_folds * n_methods * n_classifiers
    dropped = expected - len(df)
    if dropped:
        full = pd.MultiIndex.from_product(
            [sorted(df.Dataset.unique()), [0], range(n_folds), sorted(df.Method.unique()), sorted(df.Classifier.unique())],
            names=["Dataset", "Repeat", "Fold", "Method", "Classifier"])
        have = pd.MultiIndex.from_frame(df[["Dataset", "Repeat", "Fold", "Method", "Classifier"]])
        print(f"   note: {dropped} of {expected} fit slots dropped by the harness: {list(full.difference(have))}")
    assert dropped <= tolerance, f"{dropped} fits dropped"


def prepare_namespace(cells: list[str], indices: tuple[int, ...]) -> dict:
    ns = {"__name__": "__main__"}
    for i in indices:
        run_cell(ns, cells[i], f"cell{i}")
        if "CheckpointState" in ns:
            # the harness pickles its checkpoint; in a notebook the class lives in __main__
            sys.modules["__main__"].CheckpointState = ns["CheckpointState"]
    synth = synthetic_datasets(ns)
    ns["load_keel_dataset"] = lambda name, url, session, cache_dir: synth[name]
    ns["DATASETS"] = {k: "synthetic" for k in synth}
    return ns


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="nldf_nbtest_"))
    for f in ["nldf.py", "nldf_ablation.py", "nldf_comparators.py", "extension_analysis.py"]:
        shutil.copy(REPO / f, work / f)
    (work / "src").mkdir()
    shutil.copy(POST, work / "src" / "nldf_postprocessing.py")
    os.chdir(work)
    sys.path.insert(0, str(work))
    print("working directory:", work)

    # ---- 1. clean notebook: archived strategies
    clean = code_cells(NB_DIR / "nldf_benchmark_reproducible.ipynb")
    assert len(clean) == 12, f"expected 12 code cells in the clean notebook, found {len(clean)}"
    assert clean[7].startswith('"""Cell 6: FULL BENCHMARK'), "run cell moved"
    ns1 = prepare_namespace(clean, (1, 2, 3, 4, 5, 6))   # Cell 0 (pip/clone) and Cells 7-10 are not run
    run_cell(ns1, shrink(clean[7]), "clean_run")
    archived = ns1["results"]
    assert set(archived.Method) == {"Baseline", "ADASYN", "B-SMOTE", "SMOTE", "SMOTE-Tomek", "N-LDF"}, set(archived.Method)
    for col in ["Dataset", "Repeat", "Fold", "Classifier", "F1", "AUPRC", "NTrain", "NTest", "SamplerStatus",
                "n_attempted", "n_rejected_density", "n_rejected_maj", "sampling_ratio", "achieved_ratio"]:
        assert col in archived.columns, col
    Path("results").mkdir()
    archived.to_csv("results/fold_level_results.csv", index=False)
    report_dropped(archived, n_methods=6)
    print(f"1. clean notebook: {len(archived)} rows, 6 strategies            OK")

    # ---- 2. extension notebook: the three packages
    ext = code_cells(NB_DIR / "nldf_benchmark_extension.ipynb")
    assert len(ext) == 8, f"expected 8 code cells in the extension notebook, found {len(ext)}"
    assert ext[6].startswith('"""Cell 6: run one package'), "run cell moved"
    ns2 = prepare_namespace(ext, (1, 2, 3, 4, 5))        # Cell 0 (pip/clone) and Cell 7 (analysis) are not run
    packages = {}
    for pkg in ["matched", "matched_all", "ablation_statistic", "ablation_components", "comparators", "sensitivity"]:
        cell = shrink(ext[6]).replace('PACKAGE = "matched"', f'PACKAGE = "{pkg}"')
        run_cell(ns2, cell, f"ext_run_{pkg}")
        packages[pkg] = ns2["results"].copy()
        assert set(packages[pkg].Method) == set(ns2["PACKAGES"][pkg]), pkg
        report_dropped(packages[pkg], n_methods=len(ns2["PACKAGES"][pkg]))
        print(f"2. package {pkg:<20}: {len(packages[pkg])} rows                   OK")

    # ---- 3. partition identity
    key = ["Dataset", "Repeat", "Fold"]
    ref = archived.drop_duplicates(key)[key + ["NTrain", "NTest"]].set_index(key).sort_index()
    for pkg, df in packages.items():
        got = df.drop_duplicates(key)[key + ["NTrain", "NTest"]].set_index(key).sort_index()
        assert ref.equals(got), f"partitions differ in package {pkg}"
    print("3. partitions identical to the clean notebook                OK")

    # ---- 4. logged quantities
    m = packages["matched"]
    thr = m[m.Method == "Baseline-thr"]
    assert thr.Threshold.notna().all() and thr.Threshold.between(0, 1).all()
    assert (m[m.Method == "N-LDF-r1.0"].sampling_ratio == 1.0).all()
    assert (m[m.Method == "ROS-r1.0"].AchievedMinMajRatio.round(2) == 1.0).all()
    assert (m[m.Method == "SMOTE-r0.75"].AchievedMinMajRatio.round(2) <= 0.76).all()
    assert m.SamplerSeconds.notna().all()
    ab = packages["ablation_components"]
    assert (ab[ab.Method == "ABL-nosupport"].n_rejected_density == 0).all()
    assert (ab[ab.Method == "ABL-noratio"].n_rejected_maj == 0).all()
    cp = packages["comparators"]
    cw = cp[cp.Method == "Baseline-cw"]
    assert (cw.NMinAfter == cw.NMinTrain).all() and (cw.NMajAfter == cw.NMajTrain).all()
    for m in ["SLS", "SVM-SMOTE", "KMeans-SMOTE"]:
        r = cp[cp.Method == m]
        assert set(r.SamplerStatus) <= {"success", "fallback"}, m
        train_ratio = r.NMinTrain / r.NMajTrain
        assert (r.AchievedMinMajRatio >= train_ratio - 1e-9).all() and (r.AchievedMinMajRatio <= 1.05).all(), m
        if m == "SLS":
            assert (r.AchievedMinMajRatio.round(2) == 1.0).all(), "SLS should reach parity"
        if m == "KMeans-SMOTE":   # parity, or a recorded SMOTE fallback (never a silent shortfall)
            ok = (r.AchievedMinMajRatio >= 0.99) | (r.SamplerStatus == "fallback")
            assert ok.all(), "KMeans-SMOTE shortfall without a recorded fallback"
        # SVM-SMOTE may legitimately generate fewer than parity (only danger support
        # vectors seed it); the realized ratio is logged, which is the point
    ma = packages["matched_all"]
    for m in ns2["PACKAGES"]["matched_all"]:
        r = ma[ma.Method == m]
        assert len(r) > 0, m
        # r = 0.75 target: never above 0.76 (SMOTE-Tomek cleaning and SVM-SMOTE may end below)
        assert (r.AchievedMinMajRatio <= 0.76).all(), m
    sn = packages["sensitivity"]
    assert set(sn.Method) == set(ns2["PACKAGES"]["sensitivity"]) and (sn.sampling_ratio == 0.75).all()
    print("4. thresholds, ratios, ablation flags, comparators, sensitivity OK")

    # ---- 5. extension_analysis on the merged records
    if QUICK:
        print("5. extension_analysis skipped (--quick)")
    else:
        for pkg, df in packages.items():
            Path(f"nldf_results_41_{pkg}").mkdir()
            df.to_csv(f"nldf_results_41_{pkg}/fold_level_results.csv", index=False)
        cmd = [sys.executable, "extension_analysis.py", "--archived", "results/fold_level_results.csv",
               "--outdir", "results/extension", "--boot", "300"]
        cmd += [f"nldf_results_41_{p}/fold_level_results.csv" for p in packages]
        out = subprocess.run(cmd, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        res = json.loads(Path("results/extension/extension_results.json").read_text())
        for fam in ["amount", "mechanism_r0.75", "mechanism_parity", "threshold", "weighting", "comparators",
                    "comparators_parity", "amount_all", "mechanism_r0.75_all", "ablation", "sensitivity"]:
            assert fam in res["families"], fam
        for f in ["summary.md", "contrasts.csv", "strategy_means.csv"]:
            assert (Path("results/extension") / f).exists(), f
        print("5. extension_analysis on merged records                      OK")

    print("ALL NOTEBOOK TESTS PASSED (synthetic data; not an experiment)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
