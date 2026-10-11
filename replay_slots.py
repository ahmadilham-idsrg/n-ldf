#!/usr/bin/env python3
"""Replay individual fits (task, repeat, fold, strategy, classifier) and
record what happens, including the exception and traceback of a fit that
fails.

    python replay_slots.py supplementary/failed_fit_slots.csv --out runs/replay_archived.csv
    python replay_slots.py --missing runs/matched/units --out runs/replay_matched.csv

The archived run did not store the error messages of its failed fits. Every
computation in the benchmark is seeded from (task, repeat, fold, strategy) and
(task, repeat, fold, classifier), so a single fit can be recomputed in
isolation: same partition (RepeatedStratifiedKFold with the notebook's adaptive
fold rule and stable_seed(task)), same fold preprocessing, same sampler call,
same classifier. The code below follows Cell 4 of the extension notebook line
for line for one slot; where Cell 4 would swallow an exception, it is recorded.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import run_units


def schedule(ds, n_splits: int = 5, n_repeats: int = 5) -> tuple[int, int]:
    if ds.is_problematic:
        return min(n_splits, max(2, ds.n_minority // 5)), min(n_repeats, 2)
    if ds.n_minority < n_splits * 2:
        return max(2, ds.n_minority // 2), n_repeats
    return n_splits, n_repeats


def replay(ns: dict, task: str, method: str, clf_name: str, repeat: int, fold: int,
           cache_dir: str, threads_note: str) -> dict:
    url = ns["DATASETS"].get(task, f"cache:{cache_dir}")
    ds = ns["load_keel_dataset"](task, url, None, Path(cache_dir))
    folds, repeats = schedule(ds)
    splitter = ns["RepeatedStratifiedKFold"](n_splits=folds, n_repeats=repeats,
                                             random_state=ns["stable_seed"](task))
    stable_seed = ns["stable_seed"]
    rec = {"Dataset": task, "Method": method, "Classifier": clf_name, "Repeat": repeat, "Fold": fold,
           "stage": "", "outcome": "", "error_type": "", "error": "", "traceback": "", "threads": threads_note}
    for split_index, (train_idx, test_idx) in enumerate(splitter.split(ds.X, ds.y)):
        if split_index // folds != repeat or split_index % folds != fold:
            continue
        X_train_raw, X_test_raw = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
        y_train, y_test = ds.y[train_idx], ds.y[test_idx]
        try:
            rec["stage"] = "preprocess"
            pre = ns["FoldPreprocessor"](ds.numeric_columns, ds.categorical_columns, dataset_name=task).fit(X_train_raw)
            X_train, X_test = pre.transform(X_train_raw), pre.transform(X_test_raw)
            rec["stage"] = "sampler"
            seed = stable_seed(task, repeat, fold, ns["SEED_ALIAS"].get(method, method))
            sampler = ns["build_sampler_ext"](method, y_train, pre.project_categorical, seed)
            if sampler is None:
                X_res, y_res = X_train.copy(), y_train.copy()
            else:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                try:
                    X_res = pre.project_categorical(X_res)
                except Exception:
                    pass
            rec["n_min_after"] = int((y_res == 1).sum())
            rec["n_resampled"] = int(len(y_res))
            rec["stage"] = "classifier"
            clf = ns["build_classifier_ext"](clf_name, stable_seed(task, repeat, fold, clf_name), y_res, method)
            threshold = float("nan")
            if method in ns["THRESHOLD_TUNED"]:
                rec["stage"] = "threshold"
                threshold = ns["tune_threshold"](clf, X_res, y_res, stable_seed(task, repeat, fold, clf_name, "thr"))
                rec["stage"] = "classifier"
            clf.fit(X_res, y_res)
            rec["stage"] = "predict"
            y_score = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_score >= threshold).astype(int) if method in ns["THRESHOLD_TUNED"] else clf.predict(X_test)
            rec["stage"] = "metrics"
            m = ns["evaluate_predictions"](y_test, y_pred, y_score)
            rec.update({"outcome": "fit succeeded", **{f"metric_{k}": v for k, v in m.items()}})
        except Exception as exc:  # noqa: BLE001
            rec.update({"outcome": "fit failed", "error_type": type(exc).__name__,
                        "error": str(exc)[:500], "traceback": traceback.format_exc()[-2000:]})
        break
    return rec


def missing_slots(unit_dir: str) -> pd.DataFrame:
    """Slots that a finished unit should contain but does not."""
    ns = run_units.harness()
    rows = []
    for f in sorted(glob.glob(str(Path(unit_dir) / "*.json"))):
        meta = json.loads(Path(f).read_text())
        csv = Path(f).with_suffix(".csv")
        got = pd.read_csv(csv) if csv.stat().st_size > 2 else pd.DataFrame(columns=["Repeat", "Fold", "Classifier"])
        cache = json.loads((Path(unit_dir).parent / "run_config.json").read_text()).get("cache_dir", "keel_cache")
        ds = ns["load_keel_dataset"](meta["task"], ns["DATASETS"].get(meta["task"], "cache"), None, Path(cache))
        folds, repeats = schedule(ds)
        have = set(zip(got.get("Repeat", []), got.get("Fold", []), got.get("Classifier", [])))
        for r in range(repeats):
            for fo in range(folds):
                for c in ns["CLASSIFIERS"]:
                    if (r, fo, c) not in have:
                        rows.append({"Dataset": meta["task"], "Method": meta["method"], "Classifier": c,
                                     "Repeat": r, "Fold": fo, "cache_dir": cache})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slots", nargs="?", help="CSV with Dataset, Method, Classifier, Repeat, Fold")
    ap.add_argument("--missing", help="unit directory: replay the slots missing from its units")
    ap.add_argument("--cache-dir", default="keel_cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ns = run_units.harness()
    slots = missing_slots(a.missing) if a.missing else pd.read_csv(a.slots)
    if "cache_dir" not in slots.columns:
        slots["cache_dir"] = a.cache_dir
    import os
    note = f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, RF/XGB n_jobs=1"
    recs = []
    for s in slots.itertuples(index=False):
        r = replay(ns, s.Dataset, s.Method, s.Classifier, int(s.Repeat), int(s.Fold), s.cache_dir, note)
        recs.append(r)
        print(f"{s.Dataset:<32} {s.Method:<13} {s.Classifier:<10} r{s.Repeat} f{s.Fold}: {r['outcome']:<14} "
              f"{r['error_type']} {r['error'][:110]}", flush=True)
    pd.DataFrame(recs).to_csv(a.out, index=False)
    print(f"{len(recs)} slots -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
