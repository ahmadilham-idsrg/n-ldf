#!/usr/bin/env bash
# Downloads the 41 KEEL imbalanced-data zips the benchmark uses into keel_cache/
# (the cache layout the notebooks read: keel_cache/<name>.zip), checks that each
# file is a zip, and prints its SHA-256. Nothing else is modified.
#
#   bash download_keel.sh            # from the repository root; needs curl and unzip
#
# The archived run computed these hashes at load time but did not store them,
# so they cannot be compared with the archive; they identify the files used by
# any run made from this cache (the re-runnable notebooks write them to
# dataset_audit.csv). KEEL redistributes the data under its own terms:
# https://sci2s.ugr.es/keel/imbalanced.php
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p keel_cache
DATASETS=(
  "glass1" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/glass1.zip"
  "pima" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/pima.zip"
  "iris0" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/iris0.zip"
  "yeast1" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast1.zip"
  "haberman" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/haberman.zip"
  "vehicle1" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/vehicle1.zip"
  "ecoli1" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/ecoli1.zip"
  "ecoli2" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/ecoli2.zip"
  "segment0" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/segment0.zip"
  "yeast3" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast3.zip"
  "yeast-2_vs_4" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast-2_vs_4.zip"
  "vowel0" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/vowel0.zip"
  "glass4" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/glass4.zip"
  "ecoli4" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/ecoli4.zip"
  "abalone9-18" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/abalone9-18.zip"
  "yeast-1-4-5-8_vs_7" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast-1-4-5-8_vs_7.zip"
  "glass5" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/glass5.zip"
  "yeast-2_vs_8" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast-2_vs_8.zip"
  "yeast4" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast4.zip"
  "yeast-1-2-8-9_vs_7" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast-1-2-8-9_vs_7.zip"
  "yeast5" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast5.zip"
  "ecoli-0-1-3-7_vs_2-6" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/ecoli-0-1-3-7_vs_2-6.zip"
  "abalone-21_vs_8" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/abalone-21_vs_8.zip"
  "yeast6" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast6.zip"
  "winequality-white-3_vs_7" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-white-3_vs_7.zip"
  "winequality-red-8_vs_6-7" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-8_vs_6-7.zip"
  "abalone-19_vs_10-11-12-13" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/abalone-19_vs_10-11-12-13.zip"
  "winequality-white-3-9_vs_5" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-white-3-9_vs_5.zip"
  "poker-8-9_vs_6" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8-9_vs_6.zip"
  "shuttle-2_vs_5" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/shuttle-2_vs_5.zip"
  "winequality-red-3_vs_5" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/winequality-red-3_vs_5.zip"
  "abalone-20_vs_8-9-10" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/abalone-20_vs_8-9-10.zip"
  "poker-8-9_vs_5" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8-9_vs_5.zip"
  "poker-8_vs_6" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/poker-8_vs_6.zip"
  "abalone19" "https://sci2s.ugr.es/keel/dataset/data/imbalanced/abalone19.zip"
  "kddcup-land_vs_portsweep" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kddcup-land_vs_portsweep.zip"
  "kddcup-buffer_overflow_vs_back" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kddcup-buffer_overflow_vs_back.zip"
  "kddcup-land_vs_satan" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kddcup-land_vs_satan.zip"
  "kddcup-rootkit-imap_vs_back" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kddcup-rootkit-imap_vs_back.zip"
  "kr-vs-k-zero_vs_eight" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kr-vs-k-zero_vs_eight.zip"
  "kr-vs-k-zero_vs_fifteen" "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p3/kr-vs-k-zero_vs_fifteen.zip"
)
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi; }
ok=0; bad=0
for ((i=0; i<${#DATASETS[@]}; i+=2)); do
  name="${DATASETS[i]}"; url="${DATASETS[i+1]}"; out="keel_cache/$name.zip"
  if [ ! -s "$out" ]; then
    curl -fsSL --retry 3 -o "$out" "$url" || { echo "FAILED  $name  ($url)"; bad=$((bad+1)); rm -f "$out"; continue; }
  fi
  if unzip -tq "$out" >/dev/null 2>&1; then
    echo "$(sha "$out")  $name"; ok=$((ok+1))
  else
    echo "NOT A ZIP  $name"; bad=$((bad+1)); rm -f "$out"
  fi
done
echo "downloaded/verified: $ok  failed: $bad  (cache: keel_cache/)"
[ "$bad" -eq 0 ]
