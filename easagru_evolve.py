"""
easagru_evolve.py - evolve the Res-GRU of a homogeneous two-robot team.

Fitness (Nolfi style: functional, implicit, one measure):
    fraction of the episode each robot spends on the food, averaged over the
    two robots and over the episodes. Range 0..1. Nothing else is rewarded:
    not approaching, not emitting. If signalling helps, evolution finds it.

Team: both robots use the same genome (homogeneous team, team-level
selection). They differ only in their initial motivations (MOTIVATION_INIT).

Conditions (same code, same fitness, same seeds; only the controller differs):
    easa_safe - the Res-GRU is one channel of the EASA basal ganglia, whose
                innate channels only keep the robot safe (explore, avoid)
    easa      - as above plus an innate "eat" channel (stop on food);
                run easa_s1 used this one
    easa_evo  - as easa_safe, but the BG priorities (sensory and cortex
                weights, dopamine D1/D2; 17 genes) evolve with the Res-GRU
    gru_compass - GRU alone plus a compass (bearing and proximity to the
                food, as in the earlier pipeline): does the signal still pay?
    gru       - the Res-GRU drives the robot directly

Genetic algorithm (the recipe used in the earlier GRU work, sizes are args):
    elitism (--elite best copied unchanged), parents drawn uniformly from the
    --parents best, uniform crossover, then with probability --mut-prob the
    child gets Gaussian noise of std --mut-sigma on every gene.

Evaluation noise control: every individual of a generation is evaluated on
the same --episodes seeds (common random numbers); seeds change every
generation. Elites are re-evaluated each generation like everyone else.

Outputs in runs/<name>/:
    run_config.json       all arguments + library versions + machine
    evolution_log.csv     one row per generation
    best_genomes/genXXXX.npy   best genome of every generation
    checkpoint.pkl        full state after every generation (for --resume)

Usage examples:
    python3 easagru_evolve.py --condition easa --name easa_r1 --seed 1
    python3 easagru_evolve.py --name easa_r1 --resume
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from easagru_brain import ResGRU, make_controller, CONDITIONS, genome_size, initial_genome  # noqa: E402
from easagru_demo_controller import V  # noqa: E402

# ------------------------------------------------------------------ workers
_ENV = None


TASKS = {
    # name: (arena side m, episode s). Food radius stays 0.10 m (fits two e-pucks).
    "small": (1.0, 60.0),   # runs easa_s1, easa_safe_s1, gru_s1
    "big": (1.5, 90.0),     # chosen with easagru_task_probe.py: the signal is worth more
}


def _init_worker(episode_seconds, arena_size=1.0):
    """One headless PyBullet world per worker process, reused for all episodes."""
    global _ENV
    from easagru_env import Foraging3DEnv
    from easagru_world import load_config
    cfg = load_config()
    cfg["task"]["episode_seconds"] = episode_seconds
    cfg["world"]["arena_size"] = arena_size
    _ENV = Foraging3DEnv(cfg, gui=False, stop_on_success=False)


def evaluate(args):
    """Fitness of one genome over the given episode seeds (runs in a worker)."""
    genome, condition, seeds = args
    env = _ENV
    fits, both, emit_frac, takeover = [], 0, [], []
    for seed in seeds:
        obs = env.reset(int(seed))
        ctrls = []
        for k in range(2):
            rng = np.random.default_rng(int(seed) * 10 + k)
            c = make_controller(condition, genome, rng, robot_index=k)
            c.reset()
            ctrls.append(c)
        gru_on, steps, done = 0, 0, False
        while not done:
            actions = []
            for c, o in zip(ctrls, obs):
                left, right, emit, diag = c.act(o)
                actions.append((left * V, right * V, emit))
                if "gru_selected" in diag:
                    gru_on += int(diag["gru_selected"])
                    steps += 1
            obs, done, info = env.step(actions)
        m = info["max_steps"]
        fits.append(sum(info["food_steps"]) / (2.0 * m))
        both += int(all(f is not None for f in info["first_on_food_s"]))
        emit_frac.append(sum(info["emit_steps"]) / (2.0 * m))
        takeover.append(gru_on / steps if steps else np.nan)
    return (float(np.mean(fits)), both / len(seeds), float(np.mean(emit_frac)),
            float(np.nanmean(takeover)) if condition.startswith("easa") else float("nan"))


# ------------------------------------------------------------------- GA ops
def next_generation(pop, fitness, rng, a):
    order = np.argsort(fitness)[::-1]
    ranked = [pop[i] for i in order]
    new = [ranked[i].copy() for i in range(a.elite)]
    pool = min(a.parents, len(ranked))
    while len(new) < a.pop:
        p1 = ranked[rng.integers(pool)]
        p2 = ranked[rng.integers(pool)]
        child = np.where(rng.random(p1.size) < 0.5, p1, p2)
        if rng.random() < a.mut_prob:
            child = child + rng.normal(0.0, a.mut_sigma, child.size)
        new.append(child)
    return new


# -------------------------------------------------------------------- main
def versions():
    import pybullet
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pybullet_api": pybullet.getAPIVersion(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count()}


def parse():
    ap = argparse.ArgumentParser(description="Evolve the easagru Res-GRU (Nolfi-style fitness).")
    ap.add_argument("--name", required=True, help="run name; output goes to runs/<name>/")
    ap.add_argument("--condition", choices=list(CONDITIONS), default="easa_safe",
                    help="easa = EASA with innate eat channel (run easa_s1); easa_safe = EASA with "
                         "safety-only innate channels; easa_evo = easa_safe with evolved BG priorities; "
                         "gru = GRU only; gru_compass = GRU only + compass to the food")
    ap.add_argument("--seed", type=int, default=1, help="run seed (population init, episode seeds)")
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--episodes", type=int, default=6, help="episodes per individual per generation")
    ap.add_argument("--task", choices=list(TASKS), default="small",
                    help="task preset: small = 1.0 m arena, 60 s; big = 1.5 m arena, 90 s")
    ap.add_argument("--elite", type=int, default=10)
    ap.add_argument("--parents", type=int, default=25, help="parents drawn from the N best")
    ap.add_argument("--mut-prob", type=float, default=0.2)
    ap.add_argument("--mut-sigma", type=float, default=0.12)
    ap.add_argument("--init-sigma", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel worker processes (default: cores - 2)")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name>/ from its checkpoint")
    return ap.parse_args()


def main():
    a = parse()
    out = os.path.join(HERE, "runs", a.name)
    ckpt = os.path.join(out, "checkpoint.pkl")
    log_path = os.path.join(out, "evolution_log.csv")

    if a.resume:
        with open(ckpt, "rb") as f:
            state = pickle.load(f)
        saved = state["args"]
        for k in ("condition", "seed", "pop", "episodes", "elite", "parents",
                  "mut_prob", "mut_sigma", "init_sigma"):
            setattr(a, k, saved[k])          # a resumed run keeps its original settings
        a.task = saved.get("task", "small")   # runs made before --task existed are "small"
        if "--gens" not in sys.argv:
            a.gens = saved["gens"]            # --gens on resume extends (or shortens) the run
        pop, rng, gen0 = state["pop"], state["rng"], state["gen"] + 1
        print(f"[resume] {a.name}: continuing at generation {gen0} of {a.gens}")
    else:
        if os.path.exists(ckpt):
            sys.exit(f"runs/{a.name} already exists; use --resume or another --name")
        os.makedirs(os.path.join(out, "best_genomes"), exist_ok=True)
        rng = np.random.default_rng(a.seed)
        pop = [initial_genome(a.condition, rng, a.init_sigma) for _ in range(a.pop)]
        gen0 = 0
        with open(os.path.join(out, "run_config.json"), "w") as f:
            json.dump({"args": vars(a), "n_params": genome_size(a.condition), "versions": versions(),
                       "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["gen", "best", "mean", "median", "worst", "best_both_rate",
                                    "best_emit_frac", "best_gru_takeover", "mean_both_rate",
                                    "seconds"])

    arena, secs = TASKS[a.task]
    print(f"[run] {a.name}: condition={a.condition} task={a.task} ({arena} m, {secs:.0f} s) "
          f"pop={a.pop} gens={a.gens} episodes={a.episodes} workers={a.workers} "
          f"params={genome_size(a.condition)}")

    ctx = mp.get_context("spawn")   # same behaviour on macOS and Linux
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(secs, arena)) as pool:
        for gen in range(gen0, a.gens):
            t0 = time.time()
            seeds = rng.integers(0, 2 ** 31 - 1, a.episodes)
            results = pool.map(evaluate, [(g, a.condition, seeds) for g in pop])
            fit = np.array([r[0] for r in results])
            both = np.array([r[1] for r in results])
            b = int(np.argmax(fit))
            dt = time.time() - t0

            np.save(os.path.join(out, "best_genomes", f"gen{gen:04d}.npy"), pop[b])
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([gen, f"{fit[b]:.5f}", f"{fit.mean():.5f}", f"{np.median(fit):.5f}",
                                        f"{fit.min():.5f}", f"{both[b]:.3f}", f"{results[b][2]:.4f}",
                                        f"{results[b][3]:.4f}", f"{both.mean():.3f}", f"{dt:.1f}"])
            print(f"gen {gen:4d} | best {fit[b]:.4f} mean {fit.mean():.4f} | best: both-on-food "
                  f"{100*both[b]:3.0f}% emit {100*results[b][2]:4.1f}%"
                  + (f" gru {100*results[b][3]:4.1f}%" if a.condition.startswith("easa") else "")
                  + f" | {dt:6.1f}s", flush=True)

            pop = next_generation(pop, fit, rng, a)
            tmp = ckpt + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"args": vars(a), "pop": pop, "rng": rng, "gen": gen}, f)
            os.replace(tmp, ckpt)   # atomic: a crash never leaves a broken checkpoint

    print(f"[done] {a.name}: results in runs/{a.name}/")


if __name__ == "__main__":
    main()
