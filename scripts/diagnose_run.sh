#!/usr/bin/env bash
# Behavioural diagnostic of a run's champion (run evaluate_run.sh first, it
# saves runs/<name>/champion.npy): food keeping, which BG channels act on the
# food, whether the avoid reflex pushes robots off it, and how specific the
# signal is. Result in runs/<name>/diagnosis_<name>.txt.
# Usage: scripts/diagnose_run.sh <run_name> [extra args for easagru_diagnose.py]
# Example: scripts/diagnose_run.sh easa_safe_s1
#          scripts/diagnose_run.sh gru_s1 --episodes 50
set -e
cd "$(dirname "$0")/.."
RUN="${1:?run name, e.g. easa_safe_s1}"; shift
python3 easagru_diagnose.py --run "$RUN" "$@"
