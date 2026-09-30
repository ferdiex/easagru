"""
easagru_evaluate.py - evaluate an evolved run on fresh episodes, with signal
ablations.

1. Champion selection: the best genome of each of the last --top generations
   is re-evaluated on --val-episodes validation episodes (seeds never used
   in evolution); the one with the highest mean fitness is the champion.
   (Per-generation "best" is noisy: it won on only a few episodes.)
2. Test: the champion is evaluated on --episodes test episodes (other fresh
   seeds) under each signal mode:
     normal - as in evolution
     off    - robots are deaf (nothing delivered)
     random - packets arrive when they would, but with a random bearing
   If communication matters, fitness drops with "off" (and with "random" if
   the direction is used).

Outputs in runs/<name>/:
    champion.npy          the champion genome (watch it: easagru_standalone.py --genome)
    evaluation_<run>.csv           one row per test episode and signal mode
    evaluation_<run>_summary.txt
    (with --test-task big: evaluation_<run>_big.csv / _big_summary.txt)
    File names carry the run name so copies from different runs never clash.

Usage:
    python3 easagru_evaluate.py --run easa_s1
    python3 easagru_evaluate.py --run easa_s1 --top 20 --episodes 50
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import easagru_evolve as E  # noqa: E402

def evaluation_path(out, run, suffix=""):
    """runs/<run>/evaluation_<run><suffix>.csv; falls back to the old name
    evaluation<suffix>.csv (runs evaluated before the rename) when reading."""
    new = os.path.join(out, f"evaluation_{run}{suffix}.csv")
    old = os.path.join(out, f"evaluation{suffix}.csv")
    return new if os.path.exists(new) or not os.path.exists(old) else old


VAL_SEED_BASE = 800_000     # evolution draws seeds < 2^31 at random; these fixed
TEST_SEED_BASE = 900_000    # bases keep validation and test sets separate and reproducible


def task(args):
    genome, condition, seed, mode = args
    E._ENV.signal.mode = mode
    fit, both, emit, takeover = E.evaluate((genome, condition, [seed]))
    return fit, both, emit, takeover


def parse():
    ap = argparse.ArgumentParser(description="Evaluate an easagru run: champion + signal ablations.")
    ap.add_argument("--run", required=True, help="run name (folder in runs/)")
    ap.add_argument("--top", type=int, default=10, help="candidate generations (the last N)")
    ap.add_argument("--val-episodes", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=30, help="test episodes per signal mode")
    ap.add_argument("--signals", default="normal,off,random")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--test-task", default=None, choices=list(E.TASKS),
                    help="evaluate on another task than the run's own (e.g. big); implies "
                         "--use-champion and writes evaluation_<task>.csv / _summary.txt")
    ap.add_argument("--use-champion", action="store_true",
                    help="skip champion selection and use the existing runs/<run>/champion.npy")
    return ap.parse_args()


def main():
    a = parse()
    out = os.path.join(HERE, "runs", a.run)
    cfg = json.load(open(os.path.join(out, "run_config.json")))["args"]
    condition = cfg["condition"]
    task_name = cfg.get("task", "small")          # runs made before --task existed are "small"
    suffix = ""
    if a.test_task and a.test_task != task_name:
        task_name, suffix, a.use_champion = a.test_task, f"_{a.test_task}", True
    arena, secs = E.TASKS[task_name]
    files = sorted(glob.glob(os.path.join(out, "best_genomes", "gen*.npy")))[-a.top:]
    if not files and not (a.use_champion or a.test_task):
        sys.exit(f"no genomes in {out}/best_genomes")
    modes = a.signals.split(",")
    if a.use_champion:
        print(f"[eval] {a.run}: condition={condition}, task={task_name}, existing champion, "
              f"{a.episodes} test episodes per mode, workers={a.workers}")
    else:
        print(f"[eval] {a.run}: condition={condition}, task={task_name}, {len(files)} candidates, "
              f"{a.val_episodes} validation + {a.episodes} test episodes per mode, workers={a.workers}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=E._init_worker, initargs=(secs, arena)) as pool:
        # 1. champion selection (or the existing champion)
        if a.use_champion:
            champion = np.load(os.path.join(out, "champion.npy"))
            champ_name, val_c = "champion.npy (existing)", float("nan")
            print(f"[champion] using existing runs/{a.run}/champion.npy on task {task_name}")
        else:
            genomes = [np.load(f) for f in files]
            val_seeds = [VAL_SEED_BASE + i for i in range(a.val_episodes)]
            jobs = [(g, condition, s, "normal") for g in genomes for s in val_seeds]
            res = pool.map(task, jobs)
            val = np.array([r[0] for r in res]).reshape(len(genomes), len(val_seeds)).mean(axis=1)
            c = int(np.argmax(val))
            champion = genomes[c]
            champ_name, val_c = os.path.basename(files[c]), val[c]
            np.save(os.path.join(out, "champion.npy"), champion)
            print(f"[champion] {champ_name}  validation fitness {val_c:.4f}  "
                  f"(candidates {val.min():.3f}..{val.max():.3f})")

        # 2. test under each signal mode (same seeds for every mode)
        test_seeds = [TEST_SEED_BASE + i for i in range(a.episodes)]
        rows, summary = [], []
        for mode in modes:
            res = pool.map(task, [(champion, condition, s, mode) for s in test_seeds])
            f = np.array([r[0] for r in res]); b = np.array([r[1] for r in res])
            em = np.array([r[2] for r in res]); tk = np.array([r[3] for r in res])
            for s, r in zip(test_seeds, res):
                rows.append([mode, s, *[f"{v:.5f}" for v in r]])
            line = (f"signal {mode:<7} fitness {f.mean():.4f} +- {f.std(ddof=1) / np.sqrt(len(f)):.4f} (sem) | "
                    f"both on food {100*b.mean():5.1f}% | emitting {100*em.mean():5.1f}% of time"
                    + (f" | GRU in control {100*np.nanmean(tk):5.1f}%" if condition.startswith("easa") else ""))
            summary.append(line)
            print(line)

    with open(os.path.join(out, f"evaluation_{a.run}{suffix}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["signal", "seed", "fitness", "both_on_food", "emit_frac", "gru_takeover"])
        w.writerows(rows)
    with open(os.path.join(out, f"evaluation_{a.run}{suffix}_summary.txt"), "w") as fh:
        fh.write(f"run {a.run}, condition {condition}, task {task_name}, champion {champ_name}, "
                 f"validation fitness {val_c:.4f}\n" + "\n".join(summary) + "\n")
    print(f"[saved] runs/{a.run}/evaluation_{a.run}{suffix}.csv, evaluation_{a.run}{suffix}_summary.txt")


if __name__ == "__main__":
    main()
