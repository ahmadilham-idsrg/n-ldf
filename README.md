# N-LDF: amount, mechanism and decision threshold in minority oversampling

Implementation, experiment records and analysis code for the paper

> **N-LDF: A Controlled Evaluation of Class-Size-Normalized Local Support Across Oversampling Amounts, Mechanisms, and Decision Thresholds** (submitted to *Machine Learning* https://link.springer.com/journal/10994 )
> Ahmad Ilham, Mustafa Mat Deris, Deshinta Arrova Dewi

Every number, table and figure in the paper is regenerated from the files in
this repository by one command; nothing needs to be retrained.

```bash
pip install -r requirements.txt
bash reproduce_all.sh            # about 12 minutes; --quick skips the notebook test
```

## What the paper found

N-LDF is an oversampler that weights seeds, chooses partners, centres the
interpolation and screens candidates with a bounded contrast of same-class and
opposite-class Cauchy affinities, each divided by its class size raised to
lambda_p = 0.7. The paper uses it as a controlled case for a general question:
how much of an oversampler's effect comes from its mechanism, how much from the
amount of synthetic data it adds, and how much from the decision threshold at
which it is scored.

**Benchmark** (41 KEEL tasks from 15 source datasets, four classifiers, 5x5
repeated stratified CV, every setting fixed in advance, 21,911 fits). N-LDF
differs from the unresampled baseline and from none of SMOTE,
Borderline-SMOTE, ADASYN and SMOTE-Tomek in minority-class F1 (Holm p = 1.00),
also under three source-aware procedures. Every oversampler's F1 gain coincides
with a shift of the operating point and no gain in AUPRC.

**Controlled experiments** (same partitions and seeds; `results/extension/`):

| Question | Result |
|---|---|
| Is resampling needed for the F1 gain? | A threshold tuned on the training partition, no resampling: F1 0.5813, the same as every oversampler (N-LDF 0.5821, SMOTE 0.5777; Holm p = 1.00). Balanced class weights recover about half (0.5427). |
| Does the amount matter? | Each of nine mechanisms at r = 0.75 and at parity: recall up 0.008-0.022 and specificity down 0.006-0.021 at parity (all source-level Holm p <= 0.021), F1 changes by -0.007 to +0.000 (none significant). |
| Does the mechanism matter at a matched amount? | N-LDF differs in F1 from none of eight other mechanisms at either target (source-level Holm p >= 0.40), but has lower recall (by 0.04-0.09) and higher specificity (by 0.01-0.04). Interpolation beats duplication (SMOTE - ROS +0.02 F1, interval excludes zero). |
| Which part of N-LDF matters? | Removing any acceptance test or the perturbation: no measurable change. Raw affinities (lambda_p = 0) or a count-based proportion: F1 -0.026/-0.028 (Holm p <= 0.016). Not using the statistic in generation: F1 +0.009 (n.s.), recall +0.05 -- the conservative operating point comes from the support-guided generation. |
| Parameter sensitivity | Moving lambda_p (0.5, 0.9), the bandwidth neighbourhood k_d (5, 11), the acceptance threshold (+-0.1) or the ratio bound (2.0) changes F1 by at most 0.004 (none significant after Holm). |

**Held-out sources** (13 KEEL tasks from 10 sources not used in development;
protocol `heldout/PROTOCOL.md`, hashed before the run; `results/heldout/`).
N-LDF trails every SMOTE variant by about 0.06 F1 (cluster-bootstrap intervals
exclude zero for three of four; Holm p >= 0.34 with ten sources), because its
duplicate test blocks generation on the four all-categorical tasks; on the
other nine it matches them. The tuned threshold recovers about 60 % of SMOTE's
gain there.

**Reproduction.** A re-run of the six archived strategies in the archived
library versions reproduces 21,871 of the 21,885 shared records bit for bit; the
other 14 (N-LDF on `segment0`) differ by at most 0.015 because the re-run used
single-threaded linear algebra, and match exactly with two threads
(`results/rerun/`). This identifies the input files after the fact. The 25 fits missing from the
archive are explained: an unseeded XGBoost warm-up inside the notebook's
classifier factory fails with probability 2^-10 per call (Appendix A); all 25
succeed when recomputed (`results/replay/`).

## Layout

```
.
├── reproduce_all.sh            runs every check below in order and stops at the first disagreement
├── reproduce_paper_tables.py   benchmark tables and F1 tests (--check against the printed values)
├── supplementary_analysis.py   source-aware inference, secondary metrics, sensitivity, classifier tests (--check)
├── nldf.py                     the sampler class that produced the archived results (SHA-256 77c5f727...9872)
├── nldf_ablation.py            N-LDF variants: parity, statistic and component ablations, parameter changes
├── nldf_comparators.py         Safe-Level-SMOTE, audited SVM-/k-means SMOTE, r = 0.75 versions, class weights
├── run_units.py                runs (task, strategy) units with the notebook's own harness, two workers, resumable
├── run_queue.sh                the queue that produced every package in results/extension
├── nn_cache.py                 memoised nearest-neighbour index for N-LDF (speed only; verified bit-identical)
├── combine_units.py            unit files -> one record file per package
├── replay_slots.py             recomputes single fits and records exceptions (missing-fit diagnosis)
├── verify_reproduction.py      compares a re-run with the archived records
├── build_keel_cache.py         builds keel_cache/ and heldout/keel_cache/ from imbalanced-databases 0.1.1 (PyPI)
├── download_keel.sh            alternative: fetches the 41 benchmark zips from the KEEL site
├── extension_analysis.py       inference for the controlled experiments (prespecified Holm families)
├── heldout_analysis.py         the held-out analysis exactly as fixed in heldout/PROTOCOL.md
├── extension_tables.py         LaTeX tables of Sects. 7-8 from the analysis outputs (+ --check-tex)
├── run_extension.py            runs one extension package through the notebook, outside Colab
├── scaling_benchmark.py        runtime and peak memory vs N and d on synthetic data (Appendix E)
├── make-review-bundle.sh       one zip with the manuscript and this repository at the same version
├── heldout/PROTOCOL.md         held-out protocol (+ protocol.sha256)
├── tests/test_notebooks_synthetic.py   both re-runnable notebooks, headless, on synthetic data
├── make_fig*.py                figures
├── notebooks/                  executed notebook, development snapshot, re-runnable and extension notebooks
└── results/
    ├── fold_level_results.csv  archived benchmark, 21,911 rows (SHA-256 f0c957d5...7d6a4)
    ├── supplementary/          outputs of supplementary_analysis.py
    ├── extension/records/      fold-level records of the six controlled packages (csv.gz) + unit logs
    ├── extension/              their analysis: summary.md, contrasts.csv, strategy_means.csv, ...
    ├── heldout/                held-out records (csv.gz), unit log, analysis
    ├── rerun/                  re-run of the six archived strategies, unit log, reproduction check, input hashes
    ├── replay/                 recomputed missing fits (archive and extension packages)
    ├── tables/                 the generated LaTeX tables and numbers.json
    └── scaling/                runtime and memory on synthetic data
```

## How the controlled experiments were run

All packages ran with `run_units.py`, which executes Cells 1-4 of
`notebooks/nldf_benchmark_extension.ipynb` verbatim (loader, fold
preprocessor, classifiers, benchmark loop) and calls the notebook's
`run_benchmark_ext` on one task and one strategy at a time. Every seed depends
only on (task, repeat, fold, strategy) and (task, repeat, fold, classifier), so
a strategy run alone produces the same records as inside a package. The
environment had the archived library versions (Python 3.12, numpy 2.0.2,
pandas 2.2.2, scipy 1.16.3, scikit-learn 1.6.1, imbalanced-learn 0.14.2,
xgboost 3.3.0) on a two-vCPU Linux machine. Two execution changes do not alter
any record and were verified against the archive: Random Forest and XGBoost
ran with `n_jobs=1`, and N-LDF reused a fitted nearest-neighbour index for
identical reference sets (`nn_cache.py`). `bash run_queue.sh` repeats the whole
programme (about 14 hours on two cores); `python build_keel_cache.py` first
builds the inputs.

| Package | Strategies |
|---|---|
| `matched` | `N-LDF-r1.0`, `SMOTE-r0.75`, `ROS-r0.75`, `ROS-r1.0`, `Baseline-thr` |
| `matched_all` | `B-SMOTE-r0.75`, `ADASYN-r0.75`, `SMOTE-Tomek-r0.75`, `SLS-r0.75`, `SVM-SMOTE-r0.75`, `KMeans-SMOTE-r0.75` |
| `comparators` | `Baseline-cw`, `SLS`, `SVM-SMOTE`, `KMeans-SMOTE` |
| `ablation_statistic` | `ABL-lp0`, `ABL-lp1`, `ABL-localprop`, `ABL-uniform` |
| `ablation_components` | `ABL-nojitter`, `ABL-nodist`, `ABL-nosupport`, `ABL-noratio` |
| `sensitivity` | `SENS-lp0.5`, `SENS-lp0.9`, `SENS-kd5`, `SENS-kd11`, `SENS-acc-0.1`, `SENS-acc+0.1`, `SENS-rb2.0` |
| held-out | the six archived strategies and the five of `matched`, on 13 held-out tasks |
| re-run | the six archived strategies on the 41 benchmark tasks |

The Holm families of `extension_analysis.py` were written before the runs; the
families named `*_complete` (all mechanisms at one target) were added after the
first matched results as unions of prespecified contrasts and are marked as
such in the script.

## Provenance: which sampler produced the archived results

The executed notebook defines `NLDFv2` twice; the cell *"Cell 3 (FINAL
FIXED)"* produced every archived result (diagnostic columns, uncapped accepted
counts, eta maxima; `reproduce_paper_tables.py --check` verifies the first two).
`nldf.py` is that class with `build_sampler`, which fixes the sixteen benchmark
values; class defaults are not the benchmark values. The archived run
(2026-07-19) is the first run of that class; the development run one day
earlier used the same values with an earlier implementation and gave N-LDF a
higher F1. `notebooks/README.md` gives the run history.

## Data

The datasets come from the
[KEEL imbalanced-data repository](https://sci2s.ugr.es/keel/imbalanced.php) and
are not redistributed here. The archived run did not store file hashes. The
re-run, controlled and held-out experiments read the KEEL files distributed in
`imbalanced-databases==0.1.1` (`build_keel_cache.py` prints their SHA-256);
since the re-run reproduces the archive bit for bit, these files identify the
benchmark inputs.

## License

Code: MIT (`LICENSE`). Result files in `results/`: CC BY 4.0. The KEEL datasets
remain under the terms set by their maintainers.

## Citation

See `CITATION.cff`.
