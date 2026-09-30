#!/usr/bin/env bash
# Robustness of evolved champions to conditions never seen in evolution
# (bigger arena, motor noise, more sensor noise). Needs runs/<run>/champion.npy
# (saved by evaluate_run.sh). Results in runs/robustness/.
# Usage: scripts/robustness.sh [run1,run2,...] [extra args for easagru_robustness.py]
# Default runs: easa_s1,easa_safe_s1,easa_evo_s1,gru_s1
# Example: scripts/robustness.sh
#          scripts/robustness.sh gru_s1,easa_evo_s1 --episodes 100
set -e
cd "$(dirname "$0")/.."
RUNS="${1:-easa_s1,easa_safe_s1,easa_evo_s1,gru_s1}"
[ $# -gt 0 ] && shift
python3 easagru_robustness.py --runs "$RUNS" "$@"
