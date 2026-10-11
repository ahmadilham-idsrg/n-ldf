#!/usr/bin/env python3
"""Run benchmark units -- one (task, strategy) pair over all its partitions --
in parallel worker processes, with the notebook's own harness code.

    python run_units.py --methods N-LDF-r1.0 SMOTE-r0.75 --out runs/matched --workers 2
    python run_units.py --methods Baseline SMOTE N-LDF --datasets glass1 ecoli1 --out runs/verify

Each worker executes Cells 1-4 of notebooks/nldf_benchmark_extension.ipynb
verbatim (loader, fold preprocessor, classifiers, run_benchmark_ext) and calls
run_benchmark_ext on a single task with a single strategy. Partitions, seeds,
preprocessing and classifier settings are therefore those of the notebook;
because every seed depends only on (task, repeat, fold, strategy), running a
strategy alone gives the same records as running it inside a package.

The six archived strategies (Baseline, SMOTE, B-SMOTE, ADASYN, SMOTE-Tomek,
N-LDF) are built by the reproducible notebook's build_sampler (Cell 3 of
nldf_benchmark_reproducible.ipynb), so they can be re-run in the same
environment as the extension strategies.

Output: <out>/units/<task>__<strategy>.csv (one file per finished unit, written
atomically, so an interrupted run resumes by skipping finished units) and
<out>/units/<task>__<strategy>.json (wall-clock seconds, status, error).
The timeout guard of the notebook (GLOBAL_TIMEOUT = 3600 s per task) applies
here per unit unless --timeout is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One thread per worker process (BLAS, OpenMP/XGBoost); set before numpy is imported.
# Results do not depend on the thread count: RandomForest is deterministic in n_jobs,
# and the XGBoost / RF n_jobs=1 override below is verified bit-for-bit against the
# archived records (see verify_reproduction.py).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parent
ARCHIVED = ("Baseline", "SMOTE", "B-SMOTE", "ADASYN", "SMOTE-Tomek", "N-LDF")


def _cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _exec(ns: dict, text: str, name: str) -> None:
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("!") and "= !" not in l]
    exec(compile("\n".join(lines), name, "exec"), ns)


_NS: dict | None = None


def harness() -> dict:
    """Notebook namespace, built once per worker process."""
    global _NS
    if _NS is not None:
        return _NS
    import contextlib
    import io
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    nb_dir = REPO / "notebooks" if (REPO / "notebooks").exists() else REPO
    ext = _cells(nb_dir / "nldf_benchmark_extension.ipynb")
    rep = _cells(nb_dir / "nldf_benchmark_reproducible.ipynb")
    ns: dict = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for i in (1, 2, 3, 4):
            _exec(ns, ext[i], f"ext_cell{i}")
            if "CheckpointState" in ns:
                sys.modules["__main__"].CheckpointState = ns["CheckpointState"]
        # archived strategies: the reproducible notebook's build_sampler (Cell 3)
        rep_ns = dict(ns)
        _exec(rep_ns, rep[3], "rep_cell3")
    import traceback as _tb
    ns["traceback"] = _tb
    build_ext = ns["build_sampler_ext"]
    build_rep = rep_ns["build_sampler"]

    def build_any(method, y_train, projector, seed):
        if method in ARCHIVED:
            return build_rep(method, y_train, projector, seed)
        return build_ext(method, y_train, projector, seed)

    ns["build_sampler_ext"] = build_any

    # single-threaded classifiers (n_jobs=-1 -> 1); everything else unchanged
    build_clf = ns["build_classifiers"]

    def build_classifiers_1thread(seed):
        clfs = build_clf(seed)
        clfs["RF"].set_params(n_jobs=1)
        clfs["XGBoost"].set_params(n_jobs=1)
        return clfs

    ns["build_classifiers"] = build_classifiers_1thread

    # log (and re-raise) any exception raised while a classifier is built, e.g.
    # the XGBoost warm-up fit on unseeded random labels inside build_classifiers
    build_clf_ext = ns["build_classifier_ext"]
    ns["_BUILD_FAILURES"] = []

    def build_classifier_ext_logged(clf_name, seed, y_res, method):
        try:
            return build_clf_ext(clf_name, seed, y_res, method)
        except Exception as exc:  # noqa: BLE001
            ns["_BUILD_FAILURES"].append({"Classifier": clf_name, "Method": method, "seed": int(seed),
                                          "error_type": type(exc).__name__, "error": str(exc)[:300]})
            raise

    ns["build_classifier_ext"] = build_classifier_ext_logged
    if os.environ.get("NLDF_NN_CACHE") == "1":     # speed only; see nn_cache.py
        import nn_cache
        nn_cache.install()
    _NS = ns
    return ns


def run_unit(task: str, method: str, out_dir: str, timeout: int, cache_dir: str) -> dict:
    out = Path(out_dir) / "units"
    stem = f"{task}__{method}"
    csv_path, meta_path = out / f"{stem}.csv", out / f"{stem}.json"
    if csv_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())
    import contextlib
    import io
    ns = harness()
    t0 = time.time()
    meta = {"task": task, "method": method, "status": "ok", "error": "", "rows": 0}
    buf = io.StringIO()
    try:
        url = ns["DATASETS"].get(task, f"cache:{cache_dir}")   # held-out tasks: read from the cache only
        ds = ns["load_keel_dataset"](task, url, None, Path(cache_dir))
        ns["GLOBAL_TIMEOUT"] = timeout
        ns["_BUILD_FAILURES"].clear()
        with contextlib.redirect_stdout(buf):
            df, failures = ns["run_benchmark_ext"]({task: ds}, 5, 5, checkpoint_path=None,
                                                   failed_datasets=None, skip_failures=True,
                                                   methods=(method,))
        if failures:
            meta["status"] = "failed"
            meta["error"] = buf.getvalue()[-600:]
        meta["rows"] = int(len(df))
        meta["build_failures"] = list(ns["_BUILD_FAILURES"])
        meta["sha256"] = ds.sha256
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "error"
        meta["error"] = f"{exc!r}\n{traceback.format_exc()[-1500:]}"
        import pandas as pd
        df = pd.DataFrame()
    meta["seconds"] = round(time.time() - t0, 2)
    out.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=None, help="default: all 41 tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--cache-dir", default=str(REPO / "keel_cache"))
    ap.add_argument("--order", choices=["given", "small-first", "large-first"], default="large-first")
    ap.add_argument("--nn-cache", action="store_true", help="memoised NearestNeighbors in N-LDF (nn_cache.py)")
    args = ap.parse_args()
    if args.nn_cache:
        os.environ["NLDF_NN_CACHE"] = "1"

    ns = harness()
    tasks = args.datasets or list(ns["DATASETS"])
    import pandas as pd
    size = {}
    for t in tasks:
        z = Path(args.cache_dir) / f"{t}.zip"
        size[t] = z.stat().st_size if z.exists() else 0     # zip size as a proxy for task size
    units = [(t, m) for t in tasks for m in args.methods]
    if args.order != "given":
        units.sort(key=lambda u: size.get(u[0], 0), reverse=(args.order == "large-first"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "run_config.json").write_text(json.dumps(vars(args), indent=1))
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_unit, t, m, args.out, args.timeout, args.cache_dir): (t, m) for t, m in units}
        for f in as_completed(futs):
            t, m = futs[f]
            done += 1
            try:
                meta = f.result()
                print(f"[{done}/{len(units)} {(time.time() - t0) / 60:6.1f} min] {t:<32} {m:<14} "
                      f"{meta['status']:<7} rows={meta['rows']:<4} {meta.get('seconds', 0):8.1f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[{done}/{len(units)}] {t} {m} WORKER ERROR {exc!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
