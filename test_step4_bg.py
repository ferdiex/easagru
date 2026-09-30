"""
test_step4_bg.py - Step 4 validation of the EASA basal ganglia controller.

Checks:
  1. Fidelity: with EASA's own weights, easagru_bg.py reproduces the
     reference EASA core (easa_ref/bg_core.py) exactly, step by step.
  2. Selection: in canonical situations the expected channel wins
     (explore, avoid, eat, gru); avoid overrides a GRU that bids near a
     wall; on food a GRU that bids high can take over (so it can signal)
     unless the robot is starving.
  3. Anti-dithering: with two channels tied under noisy input, EASA
     switches far less often than a plain argmax over the same saliences
     (the thalamocortical loop gives persistence).
  4. Episodes, same seeds for all:
       demo controller, signal OFF (the fair reference: innate EASA has no
       communication any more)
       EASA, innate only (GRU never bids)   -> should forage like the demo
       EASA + random GRU                    -> runs; reports takeover rate
       EASA safe, innate only               -> never stops on food (expected:
                                               the task must be learned)
       EASA safe + random GRU
       GRU only, random                     -> baseline, expected to fail
  5. Timing: cost of one controller step, and of one full episode.

Usage:  python3 test_step4_bg.py          (8 seeds per condition)
        python3 test_step4_bg.py 20
"""
import sys
import time

import numpy as np

from easagru_bg import BGController as GeneralBG
from easagru_brain import (EASAController, GRUOnlyController, ResGRU, random_genome,
                           CORTEX_WEIGHTS, SENSORY_WEIGHTS, VARIANTS)
from easagru_demo_controller import DemoController, V
from easagru_env import Foraging3DEnv

sys.path.insert(0, "easa_ref")
import bg_core as ref  # noqa: E402  (reference copy of the EASA core)

RESULTS = []


def check(name, ok, detail):
    RESULTS.append(ok)
    print(f"{name:<34} {detail}  -> {'PASS' if ok else 'FAIL'}")


def innate_only_genome():
    g = np.zeros(ResGRU.N_PARAMS)
    net = ResGRU(g)
    net.b_out[3] = -10.0          # bid = sigmoid(-10) ~ 0: the GRU never asks for control
    return g


def run_episode(env, seed, make_controller):
    obs = env.reset(seed)
    ctrls = [make_controller(np.random.default_rng(seed * 10 + k), k) for k in range(2)]
    for c in ctrls:
        if hasattr(c, "reset"):
            c.reset()
    gru_steps, steps, done = 0, 0, False
    while not done:
        actions = []
        for c, o in zip(ctrls, obs):
            if isinstance(c, DemoController):
                actions.append(c.act(o))
            else:
                left, right, emit, diag = c.act(o)
                actions.append((left * V, right * V, emit))
                if "gru_selected" in diag:
                    gru_steps += int(diag["gru_selected"])
                    steps += 1
        obs, done, info = env.step(actions)
    info["takeover"] = gru_steps / steps if steps else float("nan")
    return info


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    # 1. Fidelity against the reference EASA core
    rng = np.random.default_rng(0)
    a = ref.BGController(0.2, 0.2)
    b = GeneralBG(ref.CORTEX_WEIGHTS, ref.SENSORY_WEIGHTS, 0.2, 0.2)
    worst = 0.0
    for _ in range(500):
        sensors = np.concatenate([rng.choice([-1.0, 1.0], 4), rng.uniform(0, 1, 2)])
        bh = rng.uniform(0, 1, 5)
        ga, gb = a.step(sensors, bh), b.step(sensors, bh)
        worst = max(worst, np.abs(ga - gb).max(), np.abs(a.channel_thalamus_out - b.thalamus).max())
    check("[1] Fidelity vs EASA core", worst < 1e-12, f"max difference over 500 steps: {worst:.1e}")

    # 2. Selection in canonical situations (steady state)
    cases = [
        ("nothing", [-1, -1, 0.0, 0.5, 0.5], "explore"),
        ("obstacle", [1, -1, 0.0, 0.5, 0.5], "avoid"),
        ("on food", [-1, 1, 0.0, 0.5, 0.5], "eat"),
        ("GRU bids", [-1, -1, 1.0, 0.5, 0.5], "gru"),
        ("GRU bids + obstacle", [1, -1, 1.0, 0.5, 0.5], "avoid"),
        ("GRU bids on food (can signal)", [-1, 1, 1.0, 0.5, 0.5], "gru"),
        ("GRU bids on food, starving", [-1, 1, 1.0, 0.5, 1.0], "eat"),
        ("on food + partner close", [1, 1, 0.0, 0.5, 0.5], "eat"),
    ]
    safe_cases = [
        ("safe: nothing, hungry", [-1, 0.0, 0.5, 0.8], "explore"),
        ("safe: obstacle", [1, 0.0, 0.5, 0.5], "avoid"),
        ("safe: GRU bids", [-1, 1.0, 0.5, 0.5], "gru"),
        ("safe: GRU bids + obstacle", [1, 1.0, 0.5, 0.5], "avoid"),
        ("safe: satiated, weak bid", [-1, 0.3, 0.5, 0.1], "gru"),
        ("safe: hungry, weak bid", [-1, 0.3, 0.5, 0.9], "explore"),
    ]
    all_ok, lines = True, []
    for variant, variant_cases in (("eat", cases), ("safe", safe_cases)):
        v = VARIANTS[variant]
        for name, s, expected in variant_cases:
            bg = GeneralBG(v["cw"], v["sw"], 0.2, 0.2)
            for _ in range(30):
                gpi = bg.step(np.array(s, dtype=float))
            winners = [v["channels"][i] for i in np.where(gpi < 0.4)[0]]
            ok = winners == [expected]
            all_ok &= ok
            lines.append(f"      {name:<30} -> {', '.join(winners) or 'none':<10} "
                         f"{'ok' if ok else 'EXPECTED ' + expected}")
    check("[2] Channel selection", all_ok, f"{len(cases) + len(safe_cases)} situations, two variants")
    print("\n".join(lines))

    # 3. Anti-dithering: explore vs gru tied, noisy bid, both variants
    #    eat:  explore = 0.6 - 0.3 bid + 0.6 h ; gru = 0.3 + 1.2 bid -> tie at bid 0.4 (h 0.5)
    #    safe: explore = 0.3 - 0.3 bid + 0.6 h ; gru = 0.3 + 1.2 bid -> tie at bid 0.2 (h 0.5)
    msgs, ok = [], True
    for variant, sensors_at in (("eat", lambda b: [-1, -1, b, 0.5, 0.5]), ("safe", lambda b: [-1, b, 0.5, 0.5])):
        v = VARIANTS[variant]
        tie = 0.4 if variant == "eat" else 0.2
        rng = np.random.default_rng(1)
        bg = GeneralBG(v["cw"], v["sw"], 0.2, 0.2)
        sw = np.array(v["sw"])
        easa_prev = argmax_prev = None
        easa_sw = argmax_sw = 0
        for _ in range(1000):
            s = np.array(sensors_at(tie + rng.normal(0, 0.05)))
            gpi = bg.step(s)
            w_easa, w_argmax = int(np.argmin(gpi)), int(np.argmax(sw @ s))
            easa_sw += int(easa_prev is not None and w_easa != easa_prev)
            argmax_sw += int(argmax_prev is not None and w_argmax != argmax_prev)
            easa_prev, argmax_prev = w_easa, w_argmax
        ok &= easa_sw < 0.2 * max(1, argmax_sw)
        msgs.append(f"{variant}: EASA {easa_sw} vs argmax {argmax_sw}")
    check("[3] Anti-dithering (tied, noisy)", ok, "switches in 1000 steps: " + "; ".join(msgs))

    # 4. Episodes
    env = Foraging3DEnv(gui=False)
    genome_rng = np.random.default_rng(42)
    rand_genomes = [random_genome(genome_rng) for _ in range(n_seeds)]
    conditions = [
        ("demo, signal OFF", lambda s, r, i: DemoController(False, r)),
        ("EASA, innate only", lambda s, r, i: EASAController(innate_only_genome(), r, i)),
        ("EASA + random GRU", lambda s, r, i: EASAController(rand_genomes[s % n_seeds], r, i)),
        ("EASA safe, innate", lambda s, r, i: EASAController(innate_only_genome(), r, i, variant="safe")),
        ("EASA safe + rnd GRU", lambda s, r, i: EASAController(rand_genomes[s % n_seeds], r, i, variant="safe")),
        ("GRU only, random", lambda s, r, i: GRUOnlyController(rand_genomes[s % n_seeds], r, i)),
    ]
    print(f"\n[4] Episodes, {n_seeds} seeds per condition:")
    stats = {}
    t_ep, n_ep = 0.0, 0
    for name, factory in conditions:
        both, first, gap, takeover = 0, [], [], []
        for k, seed in enumerate(range(2000, 2000 + n_seeds)):
            t0 = time.perf_counter()
            info = run_episode(env, seed, lambda r, i, k=k, f=factory: f(k, r, i))
            if not name.startswith("demo"):
                t_ep += time.perf_counter() - t0
                n_ep += 1
            f = [x for x in info["first_on_food_s"] if x is not None]
            if f:
                first.append(min(f))
            if len(f) == 2:
                both += 1
                gap.append(max(f) - min(f))
            if not np.isnan(info["takeover"]):
                takeover.append(info["takeover"])
        stats[name] = both
        tk = f"; GRU in control {100*np.mean(takeover):4.1f}% of steps" if takeover else ""
        print(f"    {name:<20} both reached food {both}/{n_seeds}; "
              f"first {np.mean(first) if first else float('nan'):5.1f} s; "
              f"gap {np.mean(gap) if gap else float('nan'):5.1f} s{tk}")
    ok = stats["EASA, innate only"] >= stats["demo, signal OFF"] - max(1, n_seeds // 4)
    check("[4] EASA innate forages like demo", ok,
          f"{stats['EASA, innate only']} vs {stats['demo, signal OFF']} (within {max(1, n_seeds // 4)})")

    # 5. Timing
    c = EASAController(random_genome(np.random.default_rng(3)))
    obs = env.reset(7)
    t0 = time.perf_counter()
    for _ in range(2000):
        c.act(obs[0])
    per_step = (time.perf_counter() - t0) / 2000
    print(f"\n[5] Timing: one EASA+GRU controller step {1e6*per_step:.0f} us; "
          f"average episode with EASA/GRU controllers {t_ep/n_ep:.2f} s wall")
    env.close()
    print("\nSTEP 4:", "ALL PASS" if all(RESULTS) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
