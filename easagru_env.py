"""
easagru_env.py - two e-pucks, one food patch, one signal (Step 3).

Task: the food is a dark patch on the floor, detectable only with the
ground sensors (no compass, no distance-to-food input). A robot is "on
food" when the mean of its three ground readings is below a threshold.
The only long-range cue about the food is the partner's signal.

Control step: every `control_period_steps` physics steps (8 x 1/120 s =
66.7 ms) each robot gets an observation and returns an action:
    action = (left_wheel_rad_s, right_wheel_rad_s, emit_bool)

Headless runs (evolution, tests) use gui=False: p.DIRECT, lean robot model,
nothing rendered. gui=True loads the decorated model for watching.
"""
import numpy as np
import pybullet as p

from easagru_world import (load_config, setup_physics, build_arena, load_epuck, set_wheels,
                           get_pose, add_patch_visual)
from easagru_sensors import EpuckSensors, FloorMap
from easagru_signal import SignalChannel

LED_OFF = [0.90, 0.10, 0.10, 1.0]
LED_ON = [1.00, 0.95, 0.20, 1.0]


class Foraging3DEnv:
    def __init__(self, cfg=None, gui=False, stop_on_success=True):
        """stop_on_success: end the episode as soon as both robots have reached
        the food (tests, GUI). Evolution uses False: the fitness is the time
        spent on the food, so the episode always runs to the end."""
        self.cfg = cfg if cfg is not None else load_config()
        self.gui = gui
        self.stop_on_success = stop_on_success
        self.period = self.cfg["sensors"]["control_period_steps"]
        self.dt_control = self.period * self.cfg["world"]["timestep"]
        task = self.cfg["task"]
        self.max_control_steps = int(round(task["episode_seconds"] / self.dt_control))

        setup_physics(self.cfg, gui=gui)
        build_arena(self.cfg)
        self.floor = FloorMap(self.cfg["floor"]["reflectance"])
        self.robots = [load_epuck(self.cfg, (-0.2, 0.0), 0.0, lean=not gui),
                       load_epuck(self.cfg, (0.2, 0.0), 0.0, lean=not gui)]
        self.led_links = []
        if gui:
            for r in self.robots:
                names = {p.getJointInfo(r["id"], j)[12].decode(): j for j in range(p.getNumJoints(r["id"]))}
                self.led_links.append(names.get("front_led"))
        self.patch_body = None
        # Perturbations (robustness tests only; all off by default, and when off
        # nothing extra is drawn from any random generator, so results stay
        # bit-identical to runs made before this option existed).
        pt = self.cfg.get("perturb", {})
        self.motor_noise = float(pt.get("motor_noise", 0.0))     # per-step relative noise on each wheel
        self.wheel_bias = float(pt.get("wheel_bias", 0.0))       # per-episode wheel gain sd
        self.sensor_noise_scale = float(pt.get("sensor_noise_scale", 1.0))
        self.perturb_rng = None
        self.wheel_gain = np.ones((2, 2))
        self.signal = SignalChannel(2, self.cfg["signal"]["delay_control_steps"], self.cfg["signal"]["range"],
                                    mode=self.cfg["signal"].get("mode", "normal"))
        self.sensors = None
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------ reset
    def _sample_layout(self):
        t = self.cfg["task"]
        half = self.cfg["world"]["arena_size"] / 2.0
        lim_f = half - t["food_wall_margin"]
        lim_r = half - t["wall_margin"]
        for _ in range(10000):
            food = self.rng.uniform(-lim_f, lim_f, 2)
            a = self.rng.uniform(-lim_r, lim_r, 2)
            b = self.rng.uniform(-lim_r, lim_r, 2)
            if (np.linalg.norm(a - food) >= t["min_spawn_food_dist"]
                    and np.linalg.norm(b - food) >= t["min_spawn_food_dist"]
                    and np.linalg.norm(a - b) >= t["min_spawn_robot_dist"]):
                return food, [a, b], self.rng.uniform(-np.pi, np.pi, 2)
        raise RuntimeError("could not sample a valid layout")

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        t = self.cfg["task"]
        food, starts, yaws = self._sample_layout()
        self.food = food

        self.floor.patches.clear()
        self.floor.add_patch(food[0], food[1], t["food_radius"], t["food_reflectance"])
        if self.gui:
            if self.patch_body is not None:
                p.removeBody(self.patch_body)
            self.patch_body = add_patch_visual(food[0], food[1], t["food_radius"])

        for r, xy, yaw in zip(self.robots, starts, yaws):
            p.resetBasePositionAndOrientation(r["id"], [xy[0], xy[1], 0.0152],
                                              p.getQuaternionFromEuler([0, 0, yaw]))
            p.resetBaseVelocity(r["id"], [0, 0, 0], [0, 0, 0])
            for j in (r["left"], r["right"]):
                p.resetJointState(r["id"], j, 0.0, 0.0)
            set_wheels(r, 0.0, 0.0)
        for _ in range(self.period):  # settle
            p.stepSimulation()

        self.sensors = [EpuckSensors(self.floor, noise=self.cfg["sensors"]["noise"],
                                     rng=np.random.default_rng(self.rng.integers(1 << 31)))
                        for _ in self.robots]
        for s in self.sensors:
            s.noise_scale = self.sensor_noise_scale
        self.signal.rng = np.random.default_rng(self.rng.integers(1 << 31))  # only used by "random" mode
        if self.motor_noise > 0 or self.wheel_bias > 0:
            self.perturb_rng = np.random.default_rng(self.rng.integers(1 << 31))
            self.wheel_gain = 1.0 + self.perturb_rng.normal(0.0, self.wheel_bias, (2, 2))
        self.signal.reset()
        self.step_count = 0
        self.first_on_food = [None, None]   # control step of first arrival
        self.food_steps = [0, 0]            # control steps spent on the food
        self.emit_count = [0, 0]
        self.emitting = [False, False]
        self._set_leds()
        return self._observe([{"received": False, "bearing": 0.0, "strength": 0.0, "distance": 0.0}] * 2)

    # ------------------------------------------------------------------- step
    def step(self, actions):
        for k, (r, (vl, vr, _)) in enumerate(zip(self.robots, actions)):
            if self.perturb_rng is not None:
                g = self.wheel_gain[k] * (1.0 + self.perturb_rng.normal(0.0, self.motor_noise, 2))
                vl, vr = vl * g[0], vr * g[1]
            set_wheels(r, vl, vr)
        self.emitting = [bool(a[2]) for a in actions]
        for i, e in enumerate(self.emitting):
            self.emit_count[i] += int(e)
        for _ in range(self.period):
            p.stepSimulation()
        self.step_count += 1

        poses = [get_pose(r)[:3] for r in self.robots]
        msgs = self.signal.step(self.emitting, poses)
        obs = self._observe(msgs)
        for i, o in enumerate(obs):
            if o["on_food"]:
                self.food_steps[i] += 1
                if self.first_on_food[i] is None:
                    self.first_on_food[i] = self.step_count
        self._set_leds()

        both = all(f is not None for f in self.first_on_food)
        done = (both and self.stop_on_success) or self.step_count >= self.max_control_steps
        return obs, done, self.info()

    # ---------------------------------------------------------------- helpers
    def _observe(self, msgs):
        thr = self.cfg["task"]["on_food_threshold"]
        obs = []
        for r, s, m in zip(self.robots, self.sensors, msgs):
            reading = s.read(r)
            # Privileged "compass" (bearing and distance to the food centre). Only
            # the gru_compass condition reads it; every other brain ignores it.
            x, y, yaw = get_pose(r)[:3]
            dx, dy = self.food[0] - x, self.food[1] - y
            obs.append({
                "ir": reading["ir"],
                "ground": reading["ground"],
                "on_food": bool(reading["ground"].mean() < thr),
                "signal": m,
                "food_bearing": float((np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi),
                "food_distance": float(np.hypot(dx, dy)),
            })
        return obs

    def _set_leds(self):
        if not self.gui:
            return
        for r, link, e in zip(self.robots, self.led_links, self.emitting):
            if link is not None:
                p.changeVisualShape(r["id"], link, rgbaColor=LED_ON if e else LED_OFF)

    def info(self):
        return {
            "step": self.step_count,
            "time_s": self.step_count * self.dt_control,
            "first_on_food_s": [None if f is None else f * self.dt_control for f in self.first_on_food],
            "emit_steps": list(self.emit_count),
            "food_steps": list(self.food_steps),
            "max_steps": self.max_control_steps,
            "food": self.food.tolist(),
        }

    def close(self):
        if p.isConnected():
            p.disconnect()
