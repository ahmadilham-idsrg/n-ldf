#!/usr/bin/env python3
"""Runtime and peak-memory scaling of the resampling call on synthetic data.

    python scaling_benchmark.py                 # full grid, writes results/scaling/
    python scaling_benchmark.py --quick         # small grid, a minute

Measures the wall-clock time and the peak Python-heap allocation
(``tracemalloc``, which numpy reports to) of one ``fit_resample`` call for
N-LDF at the archived benchmark configuration (``nldf.build_sampler``), SMOTE
(imbalanced-learn, k = 5) and random oversampling (ROS), on Gaussian
two-class data with the minority shifted by 1.5 in its first two coordinates
(the same separation at every d), at a fixed imbalance ratio of 9, for a grid of training sizes N (500, 2000, 8000) and dimensions d (5, 20, 50;
N = 8000 at d = 5 and 20 only). Three repetitions with different seeds; the
minimum over repetitions is reported (the usual choice for a timing, since
interference can only add time), the median is kept in the CSV. N-LDF is run at its
benchmark target (r = 0.75) and SMOTE/ROS at parity, as in the benchmark, and
N-LDF is also run at parity so that time per generated instance is comparable.

This is a measurement of the implementation as released, on one machine, not
a result about the method's quality: nothing here is evaluated by a
classifier. The KEEL tasks of the benchmark have N <= 4,174 and d <= 41.
"""
from __future__ import annotations

import argparse
import os

# single-threaded BLAS/OpenMP so that a timing is not at the mercy of thread
# scheduling (set before numpy is imported)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

import nldf

IR = 9.0


def make_data(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_min = max(10, int(round(n / (IR + 1))))
    X = rng.normal(size=(n, d))
    y = np.zeros(n, dtype=int)
    y[:n_min] = 1
    X[:n_min, :2] += 1.5          # the same class separation at every d
    return X, y


def samplers(y: np.ndarray, seed: int) -> dict:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    s = nldf.build_sampler("N-LDF", y, lambda X: X, seed)
    s_parity = nldf.build_sampler("N-LDF", y, lambda X: X, seed)
    s_parity.sampling_ratio = 1.0
    return {"N-LDF (r=0.75)": s, "N-LDF (parity)": s_parity,
            "SMOTE (parity)": SMOTE(k_neighbors=5, random_state=seed),
            "ROS (parity)": RandomOverSampler(random_state=seed)}


def measure(sampler, X, y) -> tuple[float, float, int]:
    tracemalloc.start()
    t0 = time.perf_counter()
    Xr, yr = sampler.fit_resample(X, y)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, peak / 2**20, int((yr == 1).sum() - (y == 1).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="results/scaling")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    grid = [(500, 5), (500, 20), (2000, 5), (2000, 20)] if args.quick else \
           [(500, 5), (500, 20), (500, 50), (2000, 5), (2000, 20), (2000, 50), (8000, 5), (8000, 20)]
    rows = []
    for n, d in grid:
        reps = args.reps
        if True:
            for rep in range(reps):
                X, y = make_data(n, d, seed=100 + rep)
                for name, s in samplers(y, seed=rep).items():
                    dt, peak_mb, n_new = measure(s, X, y)
                    rows.append({"N": n, "d": d, "n_min": int(y.sum()), "rep": rep, "sampler": name,
                                 "seconds": dt, "peak_mb": peak_mb, "n_generated": n_new,
                                 "seconds_per_1000_generated": 1000 * dt / max(n_new, 1)})
                    print(f"N={n:>6} d={d:>3} rep={rep} {name:<16} {dt:8.2f}s  peak {peak_mb:7.1f} MB  generated {n_new}")
    df = pd.DataFrame(rows)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scaling_runs.csv", index=False)
    g = df.groupby(["N", "d", "sampler"])[["seconds", "peak_mb", "n_generated", "seconds_per_1000_generated"]]
    med = g.median().reset_index()
    med.to_csv(out / "scaling_median.csv", index=False)
    mn = g.min().reset_index()
    mn.to_csv(out / "scaling_min.csv", index=False)
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "processor": platform.processor() or "n/a", "cpu_count": os.cpu_count(),
            "numpy": np.__version__, "reps": args.reps, "IR": IR, "threads": 1}
    pd.Series(info).to_json(out / "scaling_environment.json", indent=2)
    print("\nminimum seconds per call over repetitions (rows: N x d; columns: sampler)")
    print(mn.pivot_table(index=["N", "d"], columns="sampler", values="seconds").round(2))
    print("\nminimum seconds per 1000 generated instances")
    print(mn.pivot_table(index=["N", "d"], columns="sampler", values="seconds_per_1000_generated").round(1))
    print("\nmedian peak MB")
    print(med.pivot_table(index=["N", "d"], columns="sampler", values="peak_mb").round(1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
