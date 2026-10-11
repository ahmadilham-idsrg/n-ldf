#!/usr/bin/env bash
# Builds ONE zip to hand to a reviewer (or an AI reviewer workspace) so that the
# manuscript and the code it cites are always the same version:
#
#   bash make-review-bundle.sh            # from the prism-uploads folder
#
# Produces  review-bundle-<date>.zip  containing
#   manuscript/   sn-article-mlj.tex, sn-bibliography.bib, Fig1-3.pdf (flat, as for submission),
#                 MLJ-contribution-information-sheet.md, build-mlj.sh
#   n-ldf/        the repository exactly as setup-repo.sh assembles it (no .git),
#                 i.e. nldf.py, nldf_ablation.py, nldf_comparators.py, extension_analysis.py,
#                 supplementary_analysis.py, reproduce_all.sh, tests/, notebooks/, results/, ...
# and prints the SHA-256 of the manuscript and of the three analysis scripts,
# so a review can be matched to the files it read.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# repository (setup-repo.sh refuses to run git init twice; it never touches SRC)
bash "$SRC/setup-repo.sh" "$WORK/n-ldf" >/dev/null
rm -rf "$WORK/n-ldf/.git"

# manuscript, flat
mkdir -p "$WORK/manuscript"
CO="$SRC/Claude outputs"
for f in sn-article-mlj.tex sn-bibliography.bib Fig1.pdf Fig2.pdf Fig3.pdf build-mlj.sh MLJ-contribution-information-sheet.md; do
  [ -f "$CO/$f" ] && cp "$CO/$f" "$WORK/manuscript/$f" || echo "  (missing: Claude outputs/$f)"
done

OUT="$SRC/review-bundle-$STAMP.zip"
( cd "$WORK" && zip -qr "$OUT" manuscript n-ldf )
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -c1-16; else shasum -a 256 "$1" | cut -c1-16; fi; }
echo "wrote $OUT"
echo "  manuscript/sn-article-mlj.tex        $(sha "$WORK/manuscript/sn-article-mlj.tex")...  ($(wc -l < "$WORK/manuscript/sn-article-mlj.tex") lines)"
for f in supplementary_analysis.py extension_analysis.py heldout_analysis.py extension_tables.py run_units.py nldf_ablation.py nldf_comparators.py heldout/PROTOCOL.md; do
  printf "  n-ldf/%-30s %s...\n" "$f" "$(sha "$WORK/n-ldf/$f")"
done
echo "  n-ldf files: $(find "$WORK/n-ldf" -type f | wc -l | tr -d ' ')"
grep -q "Alcala2011" "$WORK/manuscript/sn-bibliography.bib" && echo "  bib: Alcala2011 present" || echo "  bib: Alcala2011 MISSING"
