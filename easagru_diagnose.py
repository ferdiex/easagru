"""
easagru_diagnose.py - what does an evolved team actually do? (headless)

Runs a champion (runs/<name>/champion.npy by default) on the same test
episodes as easagru_evaluate.py and measures, per robot and control step:

Food keeping
  - time on food, and time with BOTH robots on the food at once
  - departures: the robot was on the food and then left it (counted when it
    stays off for at least --leave-steps control steps)

Basal ganglia (EASA conditions only)
  - which channels hold control while the robot is ON the food
  - P(avoid selected | on food, partner near) vs (| on food, partner far):
    if the first is much larger, the avoid reflex fires at the partner
  - share of departures preceded by avoid (within --window steps): how often
    the reflex is what pushes the robot off the food

Signal specificity (all conditions)
  - P(emit | on food), P(emit | off food), P(emit | obstacle ahead)
    A signal that means "I am on the food" has P(emit | on food) high and
    P(emit | off food) low.

Usage:
    python3 easagru_diagnose.py --run easa_safe_s1
    python3 easagru_diagnose.py --run gru_s1 --episodes 50
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import easagru_evolve as E  # noqa: E402
from easagru_brain import make_controller, ALL_CHANNELS, ResGRU, VARIANTS, decode_bg_genes  # noqa: E402
from easagru_demo_controller import V  # noqa: E402

TEST_SEED_BASE = 900_000            # same test episodes as easagru_evaluate.py
PARTNER_NEAR = 0.15                 # m between centres: within IR range (0.074 bodies + 0.07 IR)
IR_OBSTACLE = 300.0


def run_episode(args):
    genome, condition, seed, leave_steps, window = args
    from easagru_world import get_pose
    env = E._ENV
    env.signal.mode = "normal"
    obs = env.reset(int(seed))
    ctrls = [make_controller(condition, genome, np.random.default_rng(int(seed) * 10 + k), k) for k in range(2)]
    for c in ctrls:
        c.reset()

    T = env.max_control_steps
    on = np.zeros((T, 2), bool)
    near = np.zeros(T, bool)
    emit = np.zeros((T, 2), bool)
    obst = np.zeros((T, 2), bool)
    sel = np.zeros((T, 2, len(ALL_CHANNELS)), bool)
    t, done = 0, False
    while not done:
        actions = []
        for k, (c, o) in enumerate(zip(ctrls, obs)):
            left, right, e, diag = c.act(o)
            actions.append((left * V, right * V, e))
            on[t, k] = o["on_food"]
            emit[t, k] = e
            obst[t, k] = max(o["ir"][[0, 1, 6, 7]]) > IR_OBSTACLE
            if "selected" in diag:
                sel[t, k] = diag["selected"]
        pa, pb = get_pose(env.robots[0]), get_pose(env.robots[1])
        near[t] = np.hypot(pa[0] - pb[0], pa[1] - pb[1]) < PARTNER_NEAR
        obs, done, info = env.step(actions)
        t += 1
    on, emit, obst, sel, near = on[:t], emit[:t], obst[:t], sel[:t], near[:t]

    avoid = ALL_CHANNELS.index("avoid")
    departures, by_avoid = 0, 0
    for k in range(2):
        for s in range(1, t - leave_steps):
            if on[s - 1, k] and not on[s:s + leave_steps, k].any():
                departures += 1
                if sel[max(0, s - window):s + 1, k, avoid].any():
                    by_avoid += 1
    return {"on": on, "near": near, "emit": emit, "obst": obst, "sel": sel,
            "departures": departures, "by_avoid": by_avoid}


def frac(num, den):
    return float(num) / float(den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Behavioural diagnostic of an evolved easagru team.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--genome", default=None, help="default: runs/<run>/champion.npy")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--leave-steps", type=int, default=8, help="off food this long = a departure (8 = 0.5 s)")
    ap.add_argument("--window", type=int, default=8, help="avoid within this many steps before leaving")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    out = os.path.join(HERE, "runs", a.run)
    cfg = json.load(open(os.path.join(out, "run_config.json")))["args"]
    condition, task = cfg["condition"], cfg.get("task", "small")
    arena, secs = E.TASKS[task]
    gpath = a.genome or os.path.join(out, "champion.npy")
    if not os.path.exists(gpath):
        sys.exit(f"{gpath} not found: run scripts/evaluate_run.sh {a.run} first (it saves the champion)")
    genome = np.load(gpath)
    print(f"[diagnose] {a.run}: condition={condition} task={task} genome={os.path.basename(gpath)} "
          f"episodes={a.episodes}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=E._init_worker, initargs=(secs, arena)) as pool:
        res = pool.map(run_episode, [(genome, condition, TEST_SEED_BASE + i, a.leave_steps, a.window)
                                     for i in range(a.episodes)])

    on = np.concatenate([r["on"] for r in res])
    near = np.concatenate([r["near"] for r in res])
    emit = np.concatenate([r["emit"] for r in res])
    obst = np.concatenate([r["obst"] for r in res])
    sel = np.concatenate([r["sel"] for r in res])
    deps = sum(r["departures"] for r in res)
    by_avoid = sum(r["by_avoid"] for r in res)
    n_steps = on.shape[0]
    lines = []
    lines.append(f"run {a.run} ({condition}, task {task}), {a.episodes} test episodes")
    lines.append("")
    lines.append("FOOD KEEPING")
    lines.append(f"  time on food (per robot)            {100*on.mean():5.1f}%")
    lines.append(f"  time with both on food at once      {100*on.all(axis=1).mean():5.1f}%")
    lines.append(f"  departures from food per episode    {deps / a.episodes:5.2f}")

    if sel.any():
        near2 = np.repeat(near[:, None], 2, axis=1)
        lines.append("")
        lines.append("BASAL GANGLIA (while ON the food)")
        for c, name in enumerate(ALL_CHANNELS):
            if sel[..., c].any() or name != "eat":
                lines.append(f"  {name:<8} selected {100*frac((sel[..., c] & on).sum(), on.sum()):5.1f}% of on-food time")
        av = ALL_CHANNELS.index("avoid")
        p_near = frac((sel[..., av] & on & near2).sum(), (on & near2).sum())
        p_far = frac((sel[..., av] & on & ~near2).sum(), (on & ~near2).sum())
        lines.append(f"  P(avoid | on food, partner near)    {100*p_near:5.1f}%")
        lines.append(f"  P(avoid | on food, partner far)     {100*p_far:5.1f}%")
        lines.append(f"  departures preceded by avoid        {100*frac(by_avoid, deps):5.1f}%  ({by_avoid} of {deps})")

    lines.append("")
    lines.append("SIGNAL SPECIFICITY")
    lines.append(f"  emitting overall                    {100*emit.mean():5.1f}%")
    lines.append(f"  P(emit | on food)                   {100*frac((emit & on).sum(), on.sum()):5.1f}%")
    lines.append(f"  P(emit | off food)                  {100*frac((emit & ~on).sum(), (~on).sum()):5.1f}%")
    lines.append(f"  P(emit | obstacle ahead)            {100*frac((emit & obst).sum(), obst.sum()):5.1f}%")
    lines.append(f"  ({n_steps} control steps x 2 robots)")

    if condition == "easa_evo":
        sw, cw, d1, d2 = decode_bg_genes(genome[ResGRU.N_PARAMS:])
        v = VARIANTS["safe"]
        lines.append("")
        lines.append("EVOLVED BG PRIORITIES (hand-set easa_safe value in brackets)")
        lines.append("  sensory weights    " + "".join(f"{s:>16}" for s in v["sensors"]))
        for i, ch in enumerate(v["channels"]):
            cells = "".join(f"{sw[i, j]:>8.2f} [{v['sw'][i][j]:+.1f}]" for j in range(len(v["sensors"])))
            lines.append(f"  {ch:<18} {cells}")
        lines.append("  cortex weights     " + "  ".join(f"{ch} {cw[i]:.2f} [{v['cw'][i]:.1f}]"
                                                        for i, ch in enumerate(v["channels"])))
        lines.append(f"  dopamine           D1 {d1:.2f} [0.2]   D2 {d2:.2f} [0.2]")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out, f"diagnosis_{a.run}.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\n[saved] runs/{a.run}/diagnosis_{a.run}.txt")


if __name__ == "__main__":
    main()
