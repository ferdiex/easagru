#!/usr/bin/env bash
# Run several evolutions one after another (e.g. a night and a day), each
# resuming if it was interrupted.
# Usage: scripts/run_batch.sh <condition> <task> <seed1> [seed2 ...] [-- extra args]
#        task: small | big
# Example (in the background, Mac kept awake):
#   nohup caffeinate -i scripts/run_batch.sh easa_safe big 1 2 3 > runs/batch_easa_safe_big.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
COND="${1:?condition: easa_safe, easa or gru}"; shift
TASK="${1:?task: small or big}"; shift
SEEDS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SEEDS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
for S in "${SEEDS[@]}"; do
    if [ "$TASK" = "big" ]; then NAME="${COND}_big_s${S}"; else NAME="${COND}_s${S}"; fi
    echo "=== $(date '+%F %T') run $NAME ==="
    if [ -f "runs/$NAME/checkpoint.pkl" ]; then
        python3 easagru_evolve.py --name "$NAME" --resume "$@"
    else
        python3 easagru_evolve.py --name "$NAME" --condition "$COND" --seed "$S" --task "$TASK" "$@"
    fi
done
echo "=== $(date '+%F %T') batch done ==="
