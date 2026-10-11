# Held-out source evaluation: protocol

Written 2026-10-03 05:00 UTC, before any held-out task was run. The SHA-256 of
this file is recorded in `heldout/protocol.sha256` and quoted in the
manuscript; any later change to this file changes that hash.

## Why

All N-LDF parameter values were fixed while working with the 41 KEEL tasks of
the main benchmark, so that benchmark is not an external evaluation. This
protocol evaluates the frozen implementation on tasks derived from source
datasets that were not among the 15 benchmark sources.

## Task selection rule (mechanical, no choice left)

All binary tasks distributed in KEEL `.dat` format by the PyPI package
`imbalanced-databases==0.1.1` whose source dataset is not one of the 15
benchmark sources (abalone, ecoli, glass, haberman, iris, kddcup, kr-vs-k,
pima, poker, segment, shuttle, vehicle, vowel, winequality, yeast).
Files in other formats in that package (ada, cm1, german, hepatitis, hiva,
hypothyroid, kc1, pc1, satimage, spect_f, sylva) are excluded because the
frozen loader reads KEEL `.dat` files only and writing a new loader would add
an unfrozen preprocessing path. No size, dimension or imbalance filter.

Result: 13 tasks from 10 sources.

| task | source | SHA-256 of the .dat file |
|---|---|---|
| car-good | car | 2dcf089cba757df0f971e57a57d9ee13ea5d44c9921267b2881fdcb398c375fe |
| car-vgood | car | 685296b4f1e900c9676cf53ac9a3e6b8e1bf29eed5d90a6aa8fb825f78dec487 |
| cleveland-0_vs_4 | cleveland | 4e922510f5b09f6d9092117da922ee8999a3d488ff2333ac6c9b241a5eff58f9 |
| dermatology-6 | dermatology | 8560209e01b98ef9b819a2f031915edb3907144457d81c7b21fe43fe5bc25d05 |
| flare-F | flare | ec5ebf8f4000814a1717e6989ac6c1ebbf08939d28bfecae6704c390a4608396 |
| led7digit-0-2-4-5-6-7-8-9_vs_1 | led7digit | 12ee9811f1c741d8dec1fe89cdd3b101e467e87d4f8a68d53cec28997b590ff2 |
| lymphography-normal-fibrosis | lymphography | 3597c503e28622f78532e4e2329d51aac66c22168d5e570d777f74777fdd4066 |
| new_thyroid1 | new-thyroid | 720aedb31701469639bc330711a3d1171cacfa267921c7e9e849a5ce912dc68e |
| new_thyroid2 | new-thyroid | 9356c2215707f2705e92cecb674627f94f94976783ebe1e933566a675b67618d |
| page-blocks-1-3_vs_4 | page-blocks | c088d2185584faa4aa29cf52da1394e94354e5d1acdda79f6f900415b46f2144 |
| page-blocks0 | page-blocks | 7e495f1feda3deb25f585a2eff1aa0bb683fd10cdb787350e272da8d07fc0697 |
| wisconsin | wisconsin | 386dd29dd9ae7e966e1eb1f8398ebf74dd24c349c8db77f678c1aea32c34e9df |
| zoo-3 | zoo | a61c135eaff2229bfdbc650355c12bfea0f37dde15454e16532dfaec45d2012e |

None of these task names occurs in the executed benchmark notebook, the
development notebook of 18 July 2026, `nldf.py`, or the reproducible notebook.

## What is frozen

`nldf.py` (SHA-256 77c5f727…9872) and `nldf.build_sampler` (all sixteen
parameter values); the loader, fold preprocessor, classifiers and benchmark
loop of the extension notebook (Cells 1-4, executed by `run_units.py`); the
validation schedule (5-fold x 5 repeats, stratified, with the notebook's
adaptive fold rule for tasks with fewer than ten minority instances); every
seed (`stable_seed(task, repeat, fold, strategy)`). Nothing is tuned on these
tasks.

## Strategies

The six archived strategies (Baseline, SMOTE, B-SMOTE, ADASYN, SMOTE-Tomek,
N-LDF at r = 0.75) and the five matched-amount strategies (N-LDF-r1.0,
SMOTE-r0.75, ROS-r0.75, ROS-r1.0, Baseline-thr).

## Analysis (fixed now)

Primary outcome F1; AUPRC reported alongside. One score per task-strategy
block (mean over all fits of the four classifiers). Because tasks from the
same source are dependent, the unit of inference is the source: task scores
are averaged within source, giving 10 source-level values per strategy.

1. Archived family, as in the main paper: N-LDF minus each of SMOTE, B-SMOTE,
   ADASYN, SMOTE-Tomek, Baseline. Paired Wilcoxon-Pratt on the 10 source
   values, Holm over the five contrasts; source-cluster bootstrap 95%
   interval (10,000 resamples, seed 42) of the mean task-level difference.
2. Matched-amount families exactly as in `extension_analysis.py`
   (amount, mechanism at r = 0.75, mechanism at parity, threshold), each
   Holm-corrected within family, same procedures as in 1.
3. Friedman test over the six archived strategies on source-level values;
   mean ranks reported.

With 10 sources the smallest attainable two-sided Wilcoxon p-value is
0.002, so a Holm-corrected rejection over five contrasts is possible but only
for near-unanimous effects; non-rejections are not read as equivalence.
The held-out results are reported whatever they show.
