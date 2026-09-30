"""
easagru_standalone.py - watch the two-robot foraging environment in the GUI.

Brains (--brain, default easa_safe):
    easa_safe       EASA with safety-only innate channels (explore, avoid)
    easa            EASA with an innate eat channel as well (run easa_s1)
      (without --genome both use a GRU that never bids: nobody signals)
    easa_evo        like easa_safe; with --genome its evolved BG priorities are used
    gru_compass-rand  random Res-GRU with a compass to the food (19 inputs)
    easa_safe-rand  / easa-rand   the same with a random (untrained) Res-GRU
    gru-rand        random Res-GRU driving the robot directly
    demo            hand-written step-3 test controller
  With --genome PATH (an evolved genome, e.g. runs/easa_safe_s1/champion.npy),
  use --brain easa_safe, easa, easa_evo, gru or gru_compass, matching the run's condition.
The channel(s) EASA selects are shown above each robot. With --manual you
drive robot A (the one the camera follows with F); robot B keeps its brain.

Keyboard (click inside the 3D view first so it has keyboard focus):
    SPACE          fast mode on/off (no real-time sleep), as in the Khepera
    P              pause / resume
    T              sensor rays on/off (IR: green free, red hit;
                   ground: blue floor, black food)
    L              periodic console log on/off (off at start unless --verbose)
    N              channel names above the robots on/off (they cost GUI speed;
                   G is taken by PyBullet for its own panels)
    F              camera follows robot A on/off
    R              new episode (new random food and start positions)
    Q              quit (closing the window also works)
  --manual only:
    UP / DOWN      forward / backward      LEFT / RIGHT   spin in place
    X              stop                    E              emit the signal while held

The front LED turns yellow while a robot emits. Events (episode end, new
episode, fast mode, pause) are always printed. The periodic log (every half
simulated second, per robot: on food, emitting, signal heard, mean ground
reading, strongest IR reading) is off by default; L or --verbose turns it on.

Log and live plot: every control step is written to easagru_log.csv
(disable with --nolog, other file with --log PATH). In a second terminal:
    python3 easagru_live_plot.py      (fear, hunger, ethogram, live)
    python3 easagru_plot.py           (same, saved as a PNG)

Options:  --brain NAME   --genome PATH   --signal normal|off|random
          --task small|big   (arena 1.0 m / 60 s or 1.5 m / 90 s; use the run's task)
          --manual   --nosignal (demo brain ignores the signal)
          --seed N   --verbose   --nolog   --log PATH
"""
import sys
import time

import numpy as np
import pybullet as p

from easagru_env import Foraging3DEnv
from easagru_demo_controller import DemoController, V
from easagru_brain import (ResGRU, random_genome, make_controller, CHANNEL_NAMES, CONDITIONS,
                           initial_genome, bg_prior_genes)
from easagru_sensors import IR_EXIT
from easagru_world import get_pose, set_camera
from easagru_log import StepLogger

OVERVIEW = dict(distance=1.2, yaw=0.0, pitch=-75.0)
CLOSEUP = dict(distance=0.35, yaw=0.0, pitch=-45.0)


class RayDrawer:
    """Draws one robot's sensor rays, reusing debug-line ids so the GUI stays fast.

    The macOS GUI sometimes leaves the old line on screen when a line is
    replaced (green "ghost" rays were seen above the food). So: if a replace
    returns a new id, the old id is removed explicitly, and every id ever
    created is tracked so clear() can remove them all.
    """

    def __init__(self):
        self.ids = [None] * 11
        self.all_ids = set()

    def _line(self, k, a, b, color, width):
        kw = dict(lineColorRGB=color, lineWidth=width)
        old = self.ids[k]
        if old is None:
            new = p.addUserDebugLine(a, b, **kw)
        else:
            new = p.addUserDebugLine(a, b, replaceItemUniqueId=old, **kw)
            if new != old:
                p.removeUserDebugItem(old)
                self.all_ids.discard(old)
        self.ids[k] = new
        self.all_ids.add(new)

    def draw(self, sensors, ground):
        ir_start, ir_end, ir_hit, ir_dist, gs_start, gs_end = sensors.last_rays
        for i in range(8):
            direction = (ir_end[i] - ir_start[i]) / np.linalg.norm(ir_end[i] - ir_start[i])
            end = ir_start[i] + direction * (ir_dist[i] - IR_EXIT[i])  # stops at the hit point
            self._line(i, ir_start[i], end, [1, 0, 0] if ir_hit[i] else [0, 0.8, 0], 2)
        for k in range(3):
            self._line(8 + k, gs_start[k], gs_end[k], [0, 0, 0] if ground[k] < 500 else [0, 0.3, 1], 3)

    def clear(self):
        for item in self.all_ids:
            p.removeUserDebugItem(item)
        self.ids = [None] * 11
        self.all_ids = set()


def clear_all_rays(rays):
    """Remove every ray; removeAllUserDebugItems also sweeps any ghost the GUI kept."""
    for r in rays:
        r.clear()
    p.removeAllUserDebugItems()


def innate_only_genome():
    g = np.zeros(ResGRU.N_PARAMS)
    ResGRU(g).b_out[3] = -10.0   # bid ~ 0: the GRU never asks for control
    return g


class Brain:
    """Uniform wrapper: act(obs) -> (action, label, diag).

    kind: demo | easa | easa_safe | gru, optionally with suffix "-rand"
    (random Res-GRU). Without a genome and without "-rand", EASA brains use
    an innate-only GRU (it never bids). With a genome, easa / easa_safe / gru
    load that evolved network."""

    def __init__(self, kind, use_signal, rng, robot_index, genome=None):
        self.kind = kind
        base = kind[:-5] if kind.endswith("-rand") else kind
        if base == "demo":
            self.c = DemoController(use_signal, rng)
            return
        if base not in CONDITIONS:
            raise SystemExit(f"unknown brain '{kind}'")
        if genome is None:
            if kind.endswith("-rand"):
                genome = initial_genome(base, rng)
            elif base in ("gru", "gru_compass"):
                raise SystemExit(f"--brain {base} needs --genome PATH (or use {base}-rand)")
            else:
                genome = innate_only_genome()
                if base == "easa_evo":
                    genome = np.concatenate([genome, bg_prior_genes()])
        self.c = make_controller(base, genome, rng, robot_index)
        self.c.reset()

    def act(self, obs):
        if self.kind == "demo":
            return self.c.act(obs), "demo", {}
        left, right, emit, diag = self.c.act(obs)
        if "selected" in diag:
            label = "+".join(CHANNEL_NAMES[i] for i in np.where(diag["selected"])[0]) or "none"
        else:
            label = "gru"
        return (left * V, right * V, emit), label, diag


class Labels:
    """Channel name floating above each robot (reuses the text item)."""

    def __init__(self):
        self.ids = [None, None]

    def draw(self, robots, labels):
        for k, (r, text) in enumerate(zip(robots, labels)):
            x, y = get_pose(r)[:2]
            kw = dict(textColorRGB=[0, 0, 0], textSize=1.2)
            if self.ids[k] is None:
                self.ids[k] = p.addUserDebugText(text, [x, y, 0.09], **kw)
            else:
                self.ids[k] = p.addUserDebugText(text, [x, y, 0.09], replaceItemUniqueId=self.ids[k], **kw)

    def forget(self):
        self.ids = [None, None]

    def clear(self):
        for item in self.ids:
            if item is not None:
                p.removeUserDebugItem(item)
        self.ids = [None, None]


def triggered(keys, key):
    return key in keys and keys[key] & p.KEY_WAS_TRIGGERED


def manual_action(keys, last):
    left, right = last
    if p.B3G_UP_ARROW in keys:
        left, right = V, V
    elif p.B3G_DOWN_ARROW in keys:
        left, right = -V, -V
    elif p.B3G_LEFT_ARROW in keys:
        left, right = -V, V
    elif p.B3G_RIGHT_ARROW in keys:
        left, right = V, -V
    elif ord("x") in keys:
        left = right = 0.0
    return left, right, ord("e") in keys


def main():
    manual = "--manual" in sys.argv
    use_signal = "--nosignal" not in sys.argv
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    brain = sys.argv[sys.argv.index("--brain") + 1] if "--brain" in sys.argv else "easa_safe"
    genome = np.load(sys.argv[sys.argv.index("--genome") + 1]) if "--genome" in sys.argv else None
    signal_mode = sys.argv[sys.argv.index("--signal") + 1] if "--signal" in sys.argv else "normal"
    log_path = sys.argv[sys.argv.index("--log") + 1] if "--log" in sys.argv else "easagru_log.csv"
    logger = None if "--nolog" in sys.argv else StepLogger(log_path)
    episode = 0

    from easagru_world import load_config
    cfg = load_config()
    cfg["signal"]["mode"] = signal_mode
    task = sys.argv[sys.argv.index("--task") + 1] if "--task" in sys.argv else "small"
    from easagru_evolve import TASKS
    cfg["world"]["arena_size"], cfg["task"]["episode_seconds"] = TASKS[task]
    env = Foraging3DEnv(cfg, gui=True)
    print(f"[brain {brain}{' + genome' if genome is not None else ''}, signal {signal_mode}, task {task}]")
    obs = env.reset(seed)
    for _ in range(3):
        p.stepSimulation()
    OVERVIEW["distance"] = 1.2 * cfg["world"]["arena_size"]   # whole arena in view
    set_camera(**OVERVIEW)
    print(__doc__)

    def new_controllers(s):
        return [Brain(brain, use_signal, np.random.default_rng(s * 10 + k), k, genome) for k in range(2)]

    ctrls = new_controllers(seed)
    rays = [RayDrawer(), RayDrawer()]
    labels = Labels()
    show_labels = True
    channel = ["", ""]
    show_rays, fast, paused, follow = True, False, False, False
    verbose = "--verbose" in sys.argv
    manual_cmd = (0.0, 0.0)
    report_every = max(1, int(round(0.5 / env.dt_control)))
    done = False

    try:
        while p.isConnected():
            keys = p.getKeyboardEvents()
            if triggered(keys, ord("q")):
                break
            if triggered(keys, ord(" ")):
                fast = not fast
                print(f"[fast mode {'ON' if fast else 'OFF'}]")
            if triggered(keys, ord("p")):
                paused = not paused
                print(f"[{'paused' if paused else 'running'}]")
            if triggered(keys, ord("t")):
                show_rays = not show_rays
                if not show_rays:
                    clear_all_rays(rays)
                    labels.forget()
                print(f"[rays {'ON' if show_rays else 'OFF'}]")
            if triggered(keys, ord("n")):
                show_labels = not show_labels
                if not show_labels:
                    labels.clear()
                print(f"[labels {'ON' if show_labels else 'OFF'}]")
            if triggered(keys, ord("l")):
                verbose = not verbose
                print(f"[log {'ON' if verbose else 'OFF'}]")
            if triggered(keys, ord("f")):
                follow = not follow
                if not follow:
                    set_camera(**OVERVIEW)
            if triggered(keys, ord("r")) or done:
                if done:
                    info = env.info()
                    print(f"[episode end] t={info['time_s']:.1f}s  first on food: {info['first_on_food_s']}")
                seed += 1
                episode += 1
                clear_all_rays(rays)  # rays of the old layout must not linger
                labels.forget()       # removeAllUserDebugItems also removed the labels
                obs = env.reset(seed)
                ctrls = new_controllers(seed)
                done = False
                print(f"[new episode, seed {seed}, food at {np.round(env.food, 3)}]")

            if paused:
                time.sleep(0.05)
                continue

            actions, diags = [], []
            for k, (c, o) in enumerate(zip(ctrls, obs)):
                a, channel[k], dg = c.act(o)
                actions.append(a)
                diags.append(dg)
            if manual:
                a = manual_action(keys, manual_cmd)
                manual_cmd = a[:2]
                actions[0] = a
            obs_before = obs
            obs, done, info = env.step(actions)
            if logger is not None:
                for k in range(2):
                    logger.log(episode, env.step_count, info["time_s"], k, get_pose(env.robots[k])[:3],
                               obs_before[k], actions[k][2], diags[k],
                               left=actions[k][0] / V, right=actions[k][1] / V)

            if show_rays:
                for r, s, o in zip(rays, env.sensors, obs):
                    r.draw(s, o["ground"])
            if show_labels:
                labels.draw(env.robots, channel)
            if follow:
                x, y = get_pose(env.robots[0])[:2]
                set_camera(target=(x, y, 0.0), **CLOSEUP)

            if verbose and env.step_count % report_every == 0:
                parts = []
                for name, o, e, ch in zip("AB", obs, env.emitting, channel):
                    sig = o["signal"]
                    heard = f"{np.degrees(sig['bearing']):+5.0f}deg" if sig["received"] else "   --  "
                    parts.append(f"{name}: [{ch:<9}] food={'Y' if o['on_food'] else 'n'} emit={'Y' if e else 'n'} "
                                 f"heard={heard} gs={o['ground'].mean():4.0f} irmax={o['ir'].max():5.0f}")
                print(f"t={info['time_s']:6.1f}s | " + " | ".join(parts))
            if not fast:
                time.sleep(env.dt_control)
    except p.error:
        pass  # window closed by the user: exit quietly
    finally:
        if logger is not None:
            logger.close()
            print(f"Log saved to {log_path}")
        env.close()
        print("Standalone closed.")


if __name__ == "__main__":
    main()
