#!/usr/bin/env bash
# Two-ellipses production reruns after the order of the particles along
# a loop became a quantity derived from the state (never the storage
# order): the production command of bie_vm_runs.sh under the tag _bie2,
# and the same run with the particles of each loop stored in a random
# order (_bie2_shuffle). The two must give the same events and numbers.
# Logs go to results/two_ellipses/run_<tag>.log, checkpoints every 25
# steps as before. Usage (from the repo root on the compute VM):
#   bash scripts/experiments/order_free_vm_runs.sh [threads_per_run]
set -u
T=${1:-8}
mkdir -p results/two_ellipses
export PYTHONUNBUFFERED=1
for tag in _bie2 _bie2_shuffle; do
  if [ -e "results/two_ellipses/two_ellipses${tag}.json" ]; then
    echo "results/two_ellipses/two_ellipses${tag}.json exists -- not overwriting" >&2
    exit 1
  fi
done
COMMON="--steps 4400 --metric bie --activation wbcc --auto-quotient --auto-splice --first-moment-rows"

OMP_NUM_THREADS=$T nohup uv run python scripts/experiments/two_ellipses_benchmark.py \
  $COMMON --tag _bie2 \
  > results/two_ellipses/run_bie2.log 2>&1 &

OMP_NUM_THREADS=$T nohup uv run python scripts/experiments/two_ellipses_benchmark.py \
  $COMMON --shuffle-seed 1 --tag _bie2_shuffle \
  > results/two_ellipses/run_bie2_shuffle.log 2>&1 &

wait
echo "both runs finished: $(date)"
