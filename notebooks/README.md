# Notebooks

Four notebook files live here. Two are **executed records** and must not be
edited; one is the **re-runnable** version.

| File | What it is | Sampler class(es) in the file | Run recorded in its outputs |
|---|---|---|---|
| `R1_2026_N_LDF_2_SMOTE_Q1_v_ilham.ipynb` | The Colab notebook as saved after the archived run, outputs preserved | "Cell 3 (FINAL FIXED)" — the class in `nldf.py` — followed by the superseded "Cell 3 (REVISED COMPLETE)", which also holds `build_sampler` | 2026-07-19, 12:51–15:35 (notebook clock): 41 tasks, 21,911 completed fits. **This is the archived run** (`results/fold_level_results.csv`). |
| `development/R1_2026_N_LDF_2_SMOTE_Q1_v_ilham__run-2026-07-18.ipynb` | The same notebook as saved one day earlier, outputs preserved | "Cell 3 (REVISED COMPLETE)" only | 2026-07-18, 20:00–21:55: 41 tasks, 21,917 fits — the last **development run**; also the three-task smoke test (glass1, winequality-white-3_vs_7, kddcup-land_vs_portsweep; 5 folds × 2 repeats). Its result files were overwritten and are not archived. |
| `nldf_benchmark_reproducible.ipynb` | Clean re-runnable version, no outputs | None inline: Cell 3 imports `nldf.py`, verifies its SHA-256 (`77c5f727…9872`) and builds every sampler through `build_sampler` | — |
| `nldf_benchmark_extension.ipynb` | The matched-amount, threshold-tuned and class-weighted baseline, comparator (Safe-Level-SMOTE, SVM-SMOTE, k-means SMOTE) and ablation experiments on the same partitions (see the repository README, *Extension experiments*) | None inline: `nldf.py` + `nldf_ablation.py` + `nldf_comparators.py` (all self-tested) | not yet run |

## Why two executed notebooks are kept

The executed notebook is the primary evidence of what ran, including its
timestamps and row counts, so it is kept exactly as saved. The development
snapshot is kept because it documents the history that Section 5.4 of the
manuscript discloses:

* The sixteen N-LDF parameter values are **identical** in both executed
  notebooks (`build_sampler` is byte-for-byte the same in both). The
  development run and the archived run differ only in the sampler
  implementation: the development run used the REVISED COMPLETE class
  (synthetic cap 20·n_min, 3·N_target attempts, deterministic interpolation
  coefficient, two-stage acceptance with ratio bound 2.0, jitter scale 0.1);
  the archived run is the first run of the FINAL FIXED class (no cap,
  5·N_target attempts, Beta-distributed coefficient, three-stage acceptance
  with ratio bound 1.5, OAS covariance, projection before acceptance,
  duplicate check against accepted candidates).
* The implementation corrections **lowered** N-LDF's result: dataset-level
  mean F1 0.5908 → 0.5821 and mean rank 2.78 (first of six) → 3.13 (third),
  while the five comparison strategies changed by at most 0.0002 (they are
  seeded and unaffected by the N-LDF code). The reported configuration is
  therefore not the most favourable of the runs that were made.
* Earlier output directories named `nldf_results_41_fixed` and
  `nldf_results_41_fixed_v2` appear in clean-up lists; their contents were
  not retained. Messages in the status cell indicate that at least one of
  them produced no N-LDF rows at all (the harness then still called the
  first-generation class).

## Class defaults are not the benchmark values

`NLDFv2.__init__` in `nldf.py` (and in the FINAL FIXED cell) has defaults
`tau_boundary=0.5` and `acceptance_threshold=-0.10`. The benchmark used
`tau_boundary=0.1` and `acceptance_threshold=0.0`, set explicitly in
`build_sampler`, which is the only authoritative source of the sixteen values.
`nldf.py` is not edited to align the defaults because its hash is quoted in the
manuscript; the re-runnable notebook reads the values from the sampler object
instead of retyping them.

## The inner-CV option

The class contains an inner cross-validation option for the sampling ratio
(`use_inner_cv`, `ratio_grid`). It was **disabled** in every run
(`use_inner_cv=False`; the logged `sampling_ratio` is 0.75 on all 914 calls).
Two reporting cells of the executed notebook list "inner-CV sampling ratio
selection" among the method's components; that wording is wrong and is
corrected in the re-runnable notebook. The manuscript does not make that claim.

## Re-running

Open `nldf_benchmark_reproducible.ipynb` from the repository root (or in Colab,
where Cell 0 clones the repository). Set `SMOKE_TEST = True` in Cell 6 for the
three-task check, `False` for the full 41-task run (about three hours on a
Colab CPU runtime). Results are written to `nldf_results_41_v2_final/`, the
post-processing to `nldf_results_41_v2_final/post_processing/`. Exact numerical
reproduction across library versions is not guaranteed; the archived
environment is listed in `results/analysis_manifest.json`.
