#!/usr/bin/env bash
# Launch (or resume) one evolution run in the background, safe for overnight:
#  - on macOS it is wrapped in `caffeinate -i` so the Mac does not sleep,
#  - output goes to runs/<name>/stdout.log,
#  - if runs/<name>/checkpoint.pkl exists, the run is resumed,
#  - refuses to start if the same run is already running (no double launches).
# Usage: scripts/run_evolution.sh <condition> <run_seed> [task] [extra args for easagru_evolve.py]
#        condition: easa_safe | easa | gru
#        task:      small (default: 1.0 m, 60 s) | big (1.5 m, 90 s)
# Run names: small -> <condition>_s<seed>   big -> <condition>_big_s<seed>
# Examples: scripts/run_evolution.sh gru 1
#           scripts/run_evolution.sh easa_safe 1 big
set -e
cd "$(dirname "$0")/.."
COND="${1:?condition: easa_safe, easa or gru}"
SEED="${2:?run seed, e.g. 1}"
shift 2
TASK="small"
if [ "${1:-}" = "small" ] || [ "${1:-}" = "big" ]; then TASK="$1"; shift; fi
if [ "$TASK" = "big" ]; then NAME="${COND}_big_s${SEED}"; else NAME="${COND}_s${SEED}"; fi
if pgrep -f "easagru_evolve.py --name $NAME( |$)" >/dev/null 2>&1; then
    echo "run $NAME is already running; not starting it twice."
    echo "stop it first:  pkill -f 'easagru_evolve.py --name $NAME'; pkill -f multiprocessing.spawn"
    exit 1
fi
mkdir -p "runs/$NAME"
if [ -f "runs/$NAME/checkpoint.pkl" ]; then
    CMD=(python3 easagru_evolve.py --name "$NAME" --resume "$@")
else
    CMD=(python3 easagru_evolve.py --name "$NAME" --condition "$COND" --seed "$SEED" --task "$TASK" "$@")
fi
if command -v caffeinate >/dev/null 2>&1; then CMD=(caffeinate -i "${CMD[@]}"); fi
nohup "${CMD[@]}" >> "runs/$NAME/stdout.log" 2>&1 &
echo "started $NAME (pid $!)"
echo "follow it with:  tail -f runs/$NAME/stdout.log"
