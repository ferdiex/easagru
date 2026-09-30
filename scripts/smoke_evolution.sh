#!/usr/bin/env bash
# Step 5 smoke test: a tiny evolution (3 generations), then resume it to 5,
# then project the cost of a full run from the measured time per generation.
# Usage: scripts/smoke_evolution.sh [condition] [workers] [task]
#        condition: easa_safe (default), easa or gru; workers: default cores-2;
#        task: small (default) or big
set -e
cd "$(dirname "$0")/.."
COND="${1:-easa_safe}"
WORKERS="${2:-}"
TASK="${3:-small}"
NAME="smoke_${COND}_${TASK}_$(date +%Y%m%d_%H%M%S)"
WARGS=""; [ -n "$WORKERS" ] && WARGS="--workers $WORKERS"

echo "== smoke run $NAME: 3 generations, pop 12, 2 episodes =="
python3 easagru_evolve.py --name "$NAME" --condition "$COND" --pop 12 --gens 3 \
        --episodes 2 --elite 3 --parents 6 --seed 7 --task "$TASK" $WARGS
echo "== resume to 5 generations (checks the checkpoint) =="
python3 easagru_evolve.py --name "$NAME" --resume --gens 5 $WARGS

python3 - "$NAME" <<'PY'
import csv, json, sys
name = sys.argv[1]
cfg = json.load(open(f"runs/{name}/run_config.json"))
a = cfg["args"]
rows = list(csv.DictReader(open(f"runs/{name}/evolution_log.csv")))
secs = [float(r["seconds"]) for r in rows]
per_gen = sum(secs) / len(secs)
episodes_per_gen = a["pop"] * a["episodes"]
per_episode_core = per_gen * a["workers"] / episodes_per_gen
full_gen = per_episode_core * 48 * 6 / a["workers"]
print(f"\n== timing ({a['workers']} workers) ==")
print(f"measured: {per_gen:.1f} s per generation of {episodes_per_gen} episodes "
      f"-> {per_episode_core:.2f} s per episode per core (task {a.get('task', 'small')})")
print(f"projected full run (pop 48, 6 episodes, same workers): {full_gen/60:.1f} min per generation, "
      f"{150*full_gen/3600:.1f} h for 150 generations")
print(f"generations logged: {len(rows)} (expected 5); best fitness per gen: "
      + ", ".join(r['best'] for r in rows))
PY
