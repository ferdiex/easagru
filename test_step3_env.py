"""
test_step3_env.py - Step 3 validation of the two-robot foraging environment.

Checks:
  1. Reset is reproducible: same seed -> same food and start positions.
  2. Signal: bearing, 1/d^2 strength and the one-control-step delay.
  3. Food detection: a robot placed on the patch is "on food"; off it, not.
  4. Demo episodes (hand-written controller, NOT the brain), signal ON vs
     signal OFF on the same seeds: success rate and time for the second
     robot to reach the food. Expectation: the signal helps.
  5. Timing: wall-clock seconds per episode -> the number that fixes the
     evolution budget.

Usage:  python3 test_step3_env.py            (10 seeds per condition)
        python3 test_step3_env.py 30         (more seeds, more reliable)
"""
import sys
import time

import numpy as np
import pybullet as p

from easagru_env import Foraging3DEnv
from easagru_demo_controller import DemoController

RESULTS = []


def check(name, ok, detail):
    RESULTS.append(ok)
    print(f"{name:<32} {detail}  -> {'PASS' if ok else 'FAIL'}")


def run_episode(env, seed, use_signal):
    obs = env.reset(seed)
    ctrls = [DemoController(use_signal, np.random.default_rng(seed * 10 + k)) for k in range(2)]
    done = False
    while not done:
        actions = [c.act(o) for c, o in zip(ctrls, obs)]
        obs, done, info = env.step(actions)
    return info


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    env = Foraging3DEnv(gui=False)

    # 1. Reproducible reset
    env.reset(123); a = (env.food.copy(), [p.getBasePositionAndOrientation(r["id"])[0] for r in env.robots])
    env.reset(123); b = (env.food.copy(), [p.getBasePositionAndOrientation(r["id"])[0] for r in env.robots])
    env.reset(124); c = env.food.copy()
    ok = np.allclose(a[0], b[0]) and np.allclose(a[1], b[1], atol=1e-4) and not np.allclose(a[0], c)
    check("[1] Reset reproducible", ok, f"food seed123 {np.round(a[0], 3)}, seed124 {np.round(c, 3)}")

    # 2. Signal: robot A at origin facing +x, robot B at (0, 0.2): bearing +90 deg, strength 25
    env.reset(1)
    ra, rb = env.robots
    p.resetBasePositionAndOrientation(ra["id"], [0, 0, 0.0152], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(rb["id"], [0, 0.2, 0.0152], p.getQuaternionFromEuler([0, 0, 0]))
    o1, _, _ = env.step([(0, 0, False), (0, 0, True)])   # B emits now
    o2, _, _ = env.step([(0, 0, False), (0, 0, False)])  # A receives it now (1 step late)
    o3, _, _ = env.step([(0, 0, False), (0, 0, False)])  # nothing
    s = o2[0]["signal"]
    ok = (not o1[0]["signal"]["received"] and s["received"] and not o3[0]["signal"]["received"]
          and abs(np.degrees(s["bearing"]) - 90) < 2 and abs(s["strength"] - 25) / 25 < 0.05
          and not o2[1]["signal"]["received"])
    check("[2] Signal bearing/strength/delay", ok,
          f"bearing {np.degrees(s['bearing']):.1f} deg (exp 90), strength {s['strength']:.1f} (exp 25), "
          f"delay 1 step")

    # 3. Food detection
    env.reset(2)
    fx, fy = env.food
    p.resetBasePositionAndOrientation(ra["id"], [fx - 0.03, fy, 0.0152], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(rb["id"], [fx + 0.25, fy + 0.0, 0.0152], p.getQuaternionFromEuler([0, 0, 0]))
    o, _, _ = env.step([(0, 0, False), (0, 0, False)])
    ok = o[0]["on_food"] and not o[1]["on_food"]
    check("[3] Food detection", ok,
          f"on patch: ground {np.round(o[0]['ground'])} on_food={o[0]['on_food']}; "
          f"off patch: on_food={o[1]['on_food']}")

    # 4 + 5. Demo episodes, signal ON vs OFF, same seeds
    print(f"\n[4] Demo controller, {n_seeds} seeds per condition, "
          f"episode limit {env.cfg['task']['episode_seconds']} s:")
    summary = {}
    t_total, sim_total = 0.0, 0.0
    for use_signal in (True, False):
        firsts, seconds, both = [], [], 0
        for seed in range(1000, 1000 + n_seeds):
            t0 = time.perf_counter()
            info = run_episode(env, seed, use_signal)
            t_total += time.perf_counter() - t0
            sim_total += info["time_s"]
            f = [x for x in info["first_on_food_s"] if x is not None]
            if f:
                firsts.append(min(f))
            if len(f) == 2:
                both += 1
                seconds.append(max(f) - min(f))
        summary[use_signal] = (both, seconds)
        label = "signal ON " if use_signal else "signal OFF"
        gap = f"{np.mean(seconds):5.1f} s" if seconds else "  n/a"
        print(f"    {label}: both reached food {both}/{n_seeds}; "
              f"first arrival mean {np.mean(firsts) if firsts else float('nan'):5.1f} s; "
              f"second robot arrives {gap} after the first")
    on_both, on_gap = summary[True]
    off_both, off_gap = summary[False]
    helps = on_both > off_both or (on_both == off_both and on_gap and off_gap and np.mean(on_gap) < np.mean(off_gap))
    check("[4] Signal helps (demo controller)", helps, f"both-success ON {on_both} vs OFF {off_both}")

    n_ep = 2 * n_seeds
    print(f"\n[5] Timing: {n_ep} episodes, {t_total:.1f} s wall for {sim_total:.0f} s simulated -> "
          f"{t_total / n_ep:.2f} s per episode on average, x{sim_total / t_total:.1f} real time "
          f"(two robots, sensors, signal)")
    full = env.cfg["task"]["episode_seconds"] / (sim_total / t_total)
    print(f"    a full-length episode ({env.cfg['task']['episode_seconds']} s simulated) "
          f"costs ~{full:.1f} s wall on one core")

    env.close()
    print("\nSTEP 3:", "ALL PASS" if all(RESULTS) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
