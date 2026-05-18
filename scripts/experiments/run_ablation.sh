#!/bin/bash
# Ablation on the two-ellipses test case. Five cells: 1–4 cumulative,
# 5 = cell 4 with coherence taken back out (paper-style counterpart).
#
#   1. none           baseline (q ≡ 1, no redistribute, no dead-point removal)
#   2. +coh           turn on coherence factor q_i = |V_σ|/U_σ
#   3. +coh +redist   add tangential point redistribution
#   4. full           add dead-point removal (= showcase config)
#   5. full − coh     full minus coherence: q ≡ 1 again, everything else on
#
# Outputs per cell (auto-tagged by run_two_ellipses_batch.py):
#   scripts/outputs/experiments/ablation/two_ellipses_s5000_{tag}.mp4
#   scripts/outputs/experiments/ablation/two_ellipses_s5000_{tag}_data.json
#   scripts/outputs/experiments/ablation/two_ellipses_s5000_{tag}_final_frame.png
#   scripts/outputs/experiments/ablation/two_ellipses_s5000_{tag}.npz   (--save-history)
#
# Wall: cells 1–2 abort early (LU singular, <20s); cell 3 runs ~2.5 h
# (silent divergence, see README); cell 4 ~10–15 min; cell 5 ~55 min.
# Sequential because 1× GPU can't run cells in parallel without slowdown.

set -e
OUT=scripts/outputs/experiments/ablation
COMMON="--n-steps 5000 --n-per-ellipse 64 --mass-knn-k 9 --gap 0.1 \
        --optimizer-method trust-ncg --gtol 1e-8 \
        --device auto --skip-frames 10 --save-history \
        --output-dir $OUT"

echo "=== cell 1/5: none (q≡1, no redist, no dead) ==="
uv run python scripts/examples/run_two_ellipses_batch.py $COMMON \
    --use-unit-coherence --no-redistribute

echo "=== cell 2/5: +coh (q computed, no redist, no dead) ==="
uv run python scripts/examples/run_two_ellipses_batch.py $COMMON \
    --no-redistribute

echo "=== cell 3/5: +coh +redist (q, redistribute, no dead) ==="
uv run python scripts/examples/run_two_ellipses_batch.py $COMMON

echo "=== cell 4/5: full (q, redistribute, dead removal) ==="
uv run python scripts/examples/run_two_ellipses_batch.py $COMMON \
    --remove-dead --dead-threshold 0.15

echo "=== cell 5/5: full − coh (q≡1, redistribute, dead removal) ==="
uv run python scripts/examples/run_two_ellipses_batch.py $COMMON \
    --use-unit-coherence --remove-dead --dead-threshold 0.15

echo "=== ablation complete ==="
ls -la $OUT/
