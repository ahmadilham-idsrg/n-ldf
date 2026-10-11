#!/usr/bin/env bash
# Runs every extension package, the held-out evaluation and the reference
# re-run, one after another, two worker processes each. Resumable: finished
# units are skipped. Log: logs/<package>.log
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
run() { local name=$1; shift; echo "=== $name start $(date -u +%FT%TZ)" >> logs/queue.log
        "$PY" run_units.py --workers 2 --timeout 7200 --nn-cache "$@" >> "logs/$name.log" 2>&1
        echo "=== $name end   $(date -u +%FT%TZ) exit $?" >> logs/queue.log; }
# wait for an already running package to finish
while pgrep -f "run_units.py --methods" >/dev/null; do sleep 30; done
run matched     --methods N-LDF-r1.0 SMOTE-r0.75 ROS-r0.75 ROS-r1.0 Baseline-thr --out runs/matched
run heldout     --methods Baseline SMOTE B-SMOTE ADASYN SMOTE-Tomek N-LDF N-LDF-r1.0 SMOTE-r0.75 ROS-r0.75 ROS-r1.0 Baseline-thr \
                --datasets car-good car-vgood cleveland-0_vs_4 dermatology-6 flare-F led7digit-0-2-4-5-6-7-8-9_vs_1 \
                           lymphography-normal-fibrosis new_thyroid1 new_thyroid2 page-blocks-1-3_vs_4 page-blocks0 wisconsin zoo-3 \
                --cache-dir heldout/keel_cache --out runs/heldout
run comparators --methods Baseline-cw SLS SVM-SMOTE KMeans-SMOTE --out runs/comparators
run matched_all --methods B-SMOTE-r0.75 ADASYN-r0.75 SMOTE-Tomek-r0.75 SLS-r0.75 SVM-SMOTE-r0.75 KMeans-SMOTE-r0.75 --out runs/matched_all
run ablation_components --methods ABL-nojitter ABL-nodist ABL-nosupport ABL-noratio --out runs/ablation_components
run ablation_statistic  --methods ABL-lp0 ABL-lp1 ABL-localprop ABL-uniform --out runs/ablation_statistic
run sensitivity --methods SENS-lp0.5 SENS-lp0.9 SENS-kd5 SENS-kd11 SENS-acc-0.1 SENS-acc+0.1 SENS-rb2.0 --out runs/sensitivity
run reference   --methods N-LDF Baseline SMOTE B-SMOTE ADASYN SMOTE-Tomek --out runs/reference
echo "=== queue done $(date -u +%FT%TZ)" >> logs/queue.log
