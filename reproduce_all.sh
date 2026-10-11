#!/usr/bin/env bash
# One command that regenerates every number and figure in the manuscript from
# the archived records, verifies them against the printed values, and tests
# the extension code on synthetic data. Run from the repository root:
# (needs: pip install -r requirements.txt; the notebook test also needs xgboost)
#
#   pip install -r requirements.txt
#   bash reproduce_all.sh                    # archived analyses + figures + self-tests + notebook test
#   bash reproduce_all.sh --quick            # skip the notebook test (saves about two minutes)
#   bash reproduce_all.sh --with-extension   # also analyse locally run extension packages (nldf_results_41_*)
#
# Steps 6-8 re-analyse the released records of the controlled experiments, the
# held-out evaluation and the re-run (results/extension, results/heldout,
# results/rerun) and check that the regenerated tables equal results/tables.
#
# ("bash reproduce_all.sh" works whether or not the file is executable; "chmod +x
# reproduce_all.sh" makes "./reproduce_all.sh" work too.)
#
# Nothing here retrains a model. The benchmark itself is re-run with
# notebooks/nldf_benchmark_reproducible.ipynb; the extension experiments with
# notebooks/nldf_benchmark_extension.ipynb.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON:-python3}"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -c1-16; else shasum -a 256 "$1" | cut -c1-16; fi; }
ARCHIVE="results/fold_level_results.csv"
[ -f "$ARCHIVE" ] || { echo "missing $ARCHIVE"; exit 1; }

QUICK=0; WITH_EXT=0
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    --with-extension) WITH_EXT=1 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

echo "== 0. integrity of the archived inputs"
echo "   nldf.py               $(sha nldf.py)...  (manuscript: 77c5f72721508e0c...)"
echo "   fold_level_results    $(sha "$ARCHIVE")...  (manuscript: f0c957d5633e8923...)"

echo "== 1. Tables 2, 8, 11, 21 and the F1 tests (reproduce_paper_tables.py --check)"
$PY reproduce_paper_tables.py --check

echo "== 2. Tables 4, 9, 10, 18-20, 22, Sections 6.2-6.4, Appendices A-D (supplementary_analysis.py --check)"
$PY supplementary_analysis.py --input "$ARCHIVE" --outdir results/supplementary --check

echo "== 3. Figures"
$PY make_fig1.py --out figs/fig1.pdf
$PY make_fig_results.py --input "$ARCHIVE" --outdir figs
$PY make_fig_paired_origin.py --input results/dataset_method_means.csv --out figs/fig_paired_diff_origin.pdf

echo "== 4. Extension code self-tests (synthetic data; not an experiment)"
$PY nldf_ablation.py
$PY nldf_comparators.py
$PY extension_analysis.py --selftest --outdir /tmp/nldf_extension_selftest

if [ "$QUICK" -eq 0 ]; then
  echo "== 5. Both re-runnable notebooks, headless, on synthetic data (about two minutes)"
  $PY tests/test_notebooks_synthetic.py
else
  echo "== 5. notebook test skipped (--quick)"
fi

echo "== 6. Sections 7-8: analysis and Tables 12-17 from the released records"
EXT=(results/extension/records/*.csv.gz)
$PY extension_analysis.py --archived "$ARCHIVE" --outdir results/extension "${EXT[@]}" > results/extension/run.log
TMPT="$(mktemp -d)"
$PY heldout_analysis.py results/heldout/fold_level_results.csv.gz --outdir results/heldout > results/heldout/run.log
echo "   held-out protocol SHA-256: $(sha heldout/PROTOCOL.md)...  (recorded: $(cut -c1-16 heldout/protocol.sha256)...)"
$PY extension_tables.py --ext results/extension --heldout results/heldout --out "$TMPT" > /dev/null
for f in "$TMPT"/tab_*.tex; do
  if cmp -s "$f" "results/tables/$(basename "$f")"; then echo "   OK   $(basename "$f") regenerated identically"
  else echo "   DIFF $(basename "$f") differs from results/tables"; exit 1; fi
done
rm -rf "$TMPT"

echo "== 7. Re-run of the archived benchmark against the archive (counts quoted in Sect. 5.6)"
$PY verify_reproduction.py results/rerun/fold_level_results.csv.gz --check-paper | tail -1

if [ "$WITH_EXT" -eq 1 ]; then
  echo "== 8. Locally run extension packages"
  PKGS=()
  for d in nldf_results_41_matched nldf_results_41_matched_all nldf_results_41_ablation_statistic nldf_results_41_ablation_components nldf_results_41_comparators nldf_results_41_sensitivity; do
    [ -f "$d/fold_level_results.csv" ] && PKGS+=("$d/fold_level_results.csv")
  done
  if [ ${#PKGS[@]} -gt 0 ]; then
    $PY extension_analysis.py --archived "$ARCHIVE" --outdir results/extension_local "${PKGS[@]}"
  else
    echo "   none found (run notebooks/nldf_benchmark_extension.ipynb first)"
  fi
fi

echo
echo "all steps completed"
