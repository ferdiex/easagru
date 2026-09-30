#!/usr/bin/env bash
# Evaluate a finished (or running) evolution: pick the champion among the
# best genomes of the last generations, then test it with the signal
# normal / off / random. Results in runs/<name>/evaluation_<name>.csv and evaluation_<name>_summary.txt.
# Usage: scripts/evaluate_run.sh <run_name> [extra args for easagru_evaluate.py]
# Example: scripts/evaluate_run.sh easa_s1
#          scripts/evaluate_run.sh easa_s1 --episodes 50 --top 20
set -e
cd "$(dirname "$0")/.."
RUN="${1:?run name, e.g. easa_s1}"; shift
python3 easagru_evaluate.py --run "$RUN" "$@"
echo
echo "watch the champion:  python3 easagru_standalone.py $(python3 -c "import json;a=json.load(open('runs/$RUN/run_config.json'))['args'];print('--brain', a['condition'], '--task', a.get('task','small'))") --genome runs/$RUN/champion.npy"
