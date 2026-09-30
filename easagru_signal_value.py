"""
easagru_signal_value.py - is the evolved signal worth more when the task is
harder? Compares, for each run, the signal ablations on the run's own task
(evaluation_<run>.csv) and on the big task (evaluation_<run>_big.csv, written by
easagru_evaluate.py --test-task big). Same champion, no evolution.

For each run and task: fitness with the signal normal / off / random, the
paired gains normal - off and normal - random (in seconds on the food per
robot, over the episode length of that task), and Wilcoxon p-values.

Read "off" with care for signals that are almost always on (a beacon): the
network may depend on that input as a constant, so switching it off can
break it for reasons unrelated to information. "random" keeps the packets
and scrambles only the direction.

Usage:
    python3 easagru_signal_value.py --runs gru_s1,gru_s2,gru_s3,easa_evo_s1
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import easagru_evolve as E  # noqa: E402
from easagru_evaluate import evaluation_path  # noqa: E402


def load(path):
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["signal"], {})[int(r["seed"])] = (float(r["fitness"]), float(r["emit_frac"]))
    return by


def paired(by, a, b):
    seeds = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    if not seeds:
        return float("nan"), float("nan")
    x = np.array([by[a][s][0] for s in seeds]); y = np.array([by[b][s][0] for s in seeds])
    d = x - y
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    except Exception:
        p = float("nan")
    return d.mean(), p


def main():
    ap = argparse.ArgumentParser(description="Signal value on the run's own task vs the big task.")
    ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    lines = [f"{'run':<14} {'task':<6} {'normal':>7} {'off':>7} {'random':>7} {'emit':>6} | "
             f"{'normal-off':>18} {'normal-random':>20}"]
    for run in a.runs.split(","):
        out = os.path.join(HERE, "runs", run)
        own = json.load(open(os.path.join(out, "run_config.json")))["args"].get("task", "small")
        for task, suffix in ((own, ""), ("big", "_big")):
            if task == "big" and own == "big" and suffix:
                continue            # a big-task run: its own evaluation already is the big one
            path = evaluation_path(out, run, suffix)
            if not os.path.exists(path):
                continue
            by = load(path)
            secs = E.TASKS[task][1]
            m = {k: np.mean([v[0] for v in by[k].values()]) for k in by}
            emit = np.mean([v[1] for v in by["normal"].values()]) if "normal" in by else float("nan")
            g_off, p_off = paired(by, "normal", "off")
            g_rnd, p_rnd = paired(by, "normal", "random")
            lines.append(f"{run:<14} {task:<6} {m.get('normal', np.nan):7.3f} {m.get('off', np.nan):7.3f} "
                         f"{m.get('random', np.nan):7.3f} {100*emit:5.1f}% | "
                         f"{g_off*secs:+6.1f} s (p={p_off:.1g}) {g_rnd*secs:+8.1f} s (p={p_rnd:.1g})")
    text = "\n".join(lines)
    print(text)
    path = os.path.join(HERE, "runs", "signal_value_small_vs_big.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
