#!/bin/bash
# P1.2 orchestration v5 (2026-08-05, reviewer compact patch).
# FAIR gate comparison: the support gate is metric-independent, so the
# gated main line covers ALL kinds (C1/C2/C3A/C3S); the ungated runs
# are a separate STRESS track (how each configuration fails past first
# unresolved contact). All runs use the production post-processing
# bundle; artifacts are self-contained (full SupportReport + config +
# scales + stop_context per JSON). Audit: docs/p12_pair_findings.md.
cd "$(dirname "$0")/../.."
LOG=results/p12_long
mkdir -p $LOG
run1() {  # shape kind nsteps [extra-args] [log-suffix]
  OMP_NUM_THREADS=2 stdbuf -oL uv run python \
    scripts/experiments/p12_long_comparison.py --shape "$1" \
    --kind "$2" --n-steps "$3" $4 --out $LOG 2>&1 \
    | grep --line-buffered -v "REDIST\|REMOVE\|GATE" \
    > "$LOG/log_$1_$2$5.txt"
  echo "[done] $1 $2$5 at $(date +%H:%M)"
}
echo "phase A: pair gated C3A + C3S (production candidates)"
run1 pair C3A 1200 & A=$!
run1 pair C3S 1200 & B=$!
wait $A $B
echo "phase B: pair gated C1 + C2 (fair gate comparison -- same safety layer)"
run1 pair C1 1200 & A=$!
run1 pair C2 1200 & B=$!
wait $A $B
echo "phase C: pair ungated stress C1 (cap 100) + C2 (cap 50)"
run1 pair C1 100 --no-gates _nogates & A=$!
run1 pair C2 50 --no-gates _nogates & B=$!
wait $A $B
echo "phase D: pair ungated stress C3A + star gated C3A (align telemetry)"
run1 pair C3A 1200 --no-gates _nogates & A=$!
run1 star C3A 2000 & B=$!
wait $A $B
echo "phase E: flower C3A + ellipse C3A (gated)"
run1 flower C3A 2000 & A=$!
run1 ellipse C3A 2000 & B=$!
wait $A $B
echo "phase F: flower C1 + star C1 (gated)"
run1 flower C1 2000 & A=$!
run1 star C1 2000 & B=$!
wait $A $B
echo "phase G: flower C2 + star C2 (gated)"
run1 flower C2 2000 & A=$!
run1 star C2 2000 & B=$!
wait $A $B
echo "phase H: ellipse C1 + C2 (gated)"
run1 ellipse C1 2000 & A=$!
run1 ellipse C2 2000 & B=$!
wait $A $B
echo "ALL P1.2 v5 RUNS DONE at $(date +%H:%M)"
