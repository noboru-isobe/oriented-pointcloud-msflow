#!/usr/bin/env bash
# BIE-backend production reruns of the three paper computations
# (flower, two ellipses, concentric annulus), launched in parallel on
# the compute VM (e2-standard-32). Each run writes its own JSON/states
# under the "_bie" tag next to the grid production files; logs go to
# results/<dir>/run_<tag>.log. Usage (from the repo root on the VM):
#   bash scripts/experiments/bie_vm_runs.sh [threads_per_run]
set -u
T=${1:-8}
mkdir -p results/flower results/two_ellipses results/exact_merger
export PYTHONUNBUFFERED=1

OMP_NUM_THREADS=$T nohup uv run python scripts/experiments/flower_production.py \
  --steps 4000 --metric bie --tag _bie \
  > results/flower/run_bie.log 2>&1 &

OMP_NUM_THREADS=$T nohup uv run python scripts/experiments/two_ellipses_benchmark.py \
  --steps 4400 --metric bie --activation wbcc --auto-quotient --auto-splice \
  --first-moment-rows --tag _bie \
  > results/two_ellipses/run_bie.log 2>&1 &

OMP_NUM_THREADS=$T nohup uv run python scripts/experiments/exact_merger_benchmark.py \
  --metric bie --grid 1024 --steps 2400 --endgame --compression atomic \
  --quotient-mode certificate_shadow --bulk-rows \
  --mass-estimator loopwise_oriented_kde --q-mode self_renormalized \
  --angle-scope loopwise --angle-measure raw_loopwise \
  --redist-scope loopwise --redist-curvature loopwise --redist-q-policy r_loop \
  --redist-monotone --contact-rows-mode aligned_prequotient --tag _bie \
  > results/exact_merger/run_bie.log 2>&1 &

wait
echo "all BIE runs finished: $(date)"
