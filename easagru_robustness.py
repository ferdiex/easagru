"""
easagru_robustness.py - how do evolved champions cope with conditions they
never saw in evolution? No evolution here: only evaluation.

For every run (its runs/<run>/champion.npy, saved by easagru_evaluate.py)
and every test, the champion plays the same --episodes test episodes
(signal normal). Tests:

  nominal  the training task (arena and episode length of the run's task)
  big      larger arena, longer episode: 1.5 m, 90 s
  motor    wheel gains differ per episode (sd --wheel-bias) and every wheel
           command gets relative noise (sd --motor-noise) every step
  sensor   IR and ground sensor noise x --sensor-noise-scale (Webots levels x N)

Reported: mean fitness +- sem per run and test, and the change relative to
the run's own nominal score. The question: does the architecture with
innate EASA channels degrade less than the GRU alone outside its niche?

Usage:
    python3 easagru_robustness.py --runs easa_s1,easa_safe_s1,easa_evo_s1,gru_s1
    python3 easagru_robustness.py --runs gru_s1,easa_evo_s1 --tests nominal,big --episodes 100
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import easagru_evolve as E  # noqa: E402

TEST_SEED_BASE = 900_000
_ENVS = {}


def _env(arena, secs, motor_noise, wheel_bias, sensor_scale):
    key = (arena, secs, motor_noise, wheel_bias, sensor_scale)
    if key not in _ENVS:
        from easagru_env import Foraging3DEnv
        from easagru_world import load_config
        for env in _ENVS.values():
            env.close()
        _ENVS.clear()
        cfg = load_config()
        cfg["world"]["arena_size"] = arena
        cfg["task"]["episode_seconds"] = secs
        cfg["perturb"] = {"motor_noise": motor_noise, "wheel_bias": wheel_bias,
                          "sensor_noise_scale": sensor_scale}
        _ENVS[key] = Foraging3DEnv(cfg, gui=False, stop_on_success=False)
    return _ENVS[key]


def task(args):
    run, test, genome, condition, env_key, seed = args
    E._ENV = _env(*env_key)
    E._ENV.signal.mode = "normal"
    fit, both, emit, takeover = E.evaluate((genome, condition, [seed]))
    return run, test, seed, fit, both


def main():
    ap = argparse.ArgumentParser(description="Robustness of evolved champions to unseen conditions.")
    ap.add_argument("--runs", required=True, help="comma-separated run names")
    ap.add_argument("--tests", default="nominal,big,motor,sensor")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--motor-noise", type=float, default=0.10)
    ap.add_argument("--wheel-bias", type=float, default=0.05)
    ap.add_argument("--sensor-noise-scale", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "robustness"))
    a = ap.parse_args()

    runs, tests = a.runs.split(","), a.tests.split(",")
    jobs, info = [], {}
    for run in runs:
        out = os.path.join(HERE, "runs", run)
        cfg = json.load(open(os.path.join(out, "run_config.json")))["args"]
        gpath = os.path.join(out, "champion.npy")
        if not os.path.exists(gpath):
            sys.exit(f"{gpath} missing: run scripts/evaluate_run.sh {run} first")
        genome, condition = np.load(gpath), cfg["condition"]
        arena, secs = E.TASKS[cfg.get("task", "small")]
        info[run] = condition
        env_keys = {
            "nominal": (arena, secs, 0.0, 0.0, 1.0),
            "big": (1.5, 90.0, 0.0, 0.0, 1.0),
            "motor": (arena, secs, a.motor_noise, a.wheel_bias, 1.0),
            "sensor": (arena, secs, 0.0, 0.0, a.sensor_noise_scale),
        }
        for test in tests:
            for i in range(a.episodes):
                jobs.append((run, test, genome, condition, env_keys[test], TEST_SEED_BASE + i))
    jobs.sort(key=lambda j: j[4])   # group by world setup: fewer rebuilds per worker
    print(f"[robustness] runs={runs} tests={tests} episodes={a.episodes} workers={a.workers}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        res = pool.map(task, jobs, chunksize=max(1, a.episodes // 4))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "robustness.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "condition", "test", "seed", "fitness", "both_on_food"])
        for r in res:
            w.writerow([r[0], info[r[0]], r[1], r[2], f"{r[3]:.5f}", f"{r[4]:.3f}"])

    lines = [f"robustness: {a.episodes} test episodes per cell, signal normal; "
             f"motor: wheel bias sd {a.wheel_bias}, noise sd {a.motor_noise}; "
             f"sensor: noise x{a.sensor_noise_scale}; big: 1.5 m, 90 s",
             "",
             f"{'run':<16}" + "".join(f"{t:>22}" for t in tests)]
    for run in runs:
        nominal = np.mean([r[3] for r in res if r[0] == run and r[1] == "nominal"]) if "nominal" in tests else None
        cells = []
        for test in tests:
            f_ = np.array([r[3] for r in res if r[0] == run and r[1] == test])
            m, sem = f_.mean(), f_.std(ddof=1) / np.sqrt(len(f_))
            rel = f" ({100 * (m / nominal - 1):+4.0f}%)" if nominal and test != "nominal" else "        "
            cells.append(f"{m:.3f}+-{sem:.3f}{rel}")
        lines.append(f"{run:<16}" + "".join(f"{c:>22}" for c in cells))
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(a.out, "robustness_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\n[saved] {a.out}/robustness.csv and robustness_summary.txt")


if __name__ == "__main__":
    main()
