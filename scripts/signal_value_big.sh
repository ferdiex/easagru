#!/usr/bin/env bash
# Cheap test before running evolution on the big task: take the champions we
# already have (small task) and evaluate them on the big task (1.5 m, 90 s)
# with the signal normal / off / random. Then compare the value of the
# signal on both tasks. No evolution; ~10 min on the Mac for 4 runs.
# Needs runs/<run>/champion.npy and evaluation_<run>.csv (evaluate_run.sh).
# Usage: scripts/signal_value_big.sh [run1,run2,...] [episodes]
# Default: gru_s1,gru_s2,gru_s3,easa_evo_s1 with 50 episodes
set -e
cd "$(dirname "$0")/.."
RUNS="${1:-gru_s1,gru_s2,gru_s3,easa_evo_s1}"
EPIS="${2:-50}"
for R in ${RUNS//,/ }; do
    if [ -f "runs/$R/champion.npy" ]; then
        python3 easagru_evaluate.py --run "$R" --test-task big --episodes "$EPIS"
    else
        echo "skip $R: no champion (run scripts/evaluate_run.sh $R first)"
    fi
done
echo
python3 easagru_signal_value.py --runs "$RUNS"
