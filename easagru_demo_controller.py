"""
easagru_demo_controller.py - a hand-written controller used ONLY to test
the environment (Step 3). It is not part of the model and is never evolved.

Behaviour, in priority order:
  1. On food        -> stop and emit the signal.
  2. Obstacle ahead -> turn away from the side with the stronger IR reading.
  3. Signal heard   -> (if use_signal) steer toward the emitter's bearing.
  4. Otherwise      -> wander: drive forward, pick a new random turn rate
                       every ~2 s.

use_signal=False gives the "deaf" baseline, to check that in this
environment the signal actually shortens the time for the second robot.
"""
import numpy as np

V = 6.28            # max wheel speed, rad/s
IR_OBSTACLE = 300.0  # raw IR value treated as "obstacle close" (~2 cm)


class DemoController:
    def __init__(self, use_signal=True, rng=None):
        self.use_signal = use_signal
        self.rng = rng if rng is not None else np.random.default_rng()
        self.turn = 0.0
        self.timer = 0

    def act(self, obs):
        if obs["on_food"]:
            return 0.0, 0.0, True

        ir = obs["ir"]
        right_side = max(ir[0], ir[1])  # ps0, ps1 look front-right
        left_side = max(ir[6], ir[7])   # ps6, ps7 look front-left
        if max(right_side, left_side) > IR_OBSTACLE:
            if right_side > left_side:
                return -0.5 * V, 0.5 * V, False  # spin left
            return 0.5 * V, -0.5 * V, False      # spin right

        sig = obs["signal"]
        if self.use_signal and sig["received"]:
            k = np.clip(sig["bearing"] / (np.pi / 2), -1.0, 1.0)
            return V * (1.0 - k), V * (1.0 + k), False

        self.timer -= 1
        if self.timer <= 0:
            self.turn = self.rng.uniform(-0.5, 0.5)
            self.timer = int(self.rng.integers(16, 48))  # 1-3 s at 62.5 ms
        return V * (1.0 - self.turn), V * (1.0 + self.turn), False
