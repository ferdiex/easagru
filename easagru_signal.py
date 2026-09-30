"""
easagru_signal.py - the single communication signal, modelled on the
Webots e-puck Emitter/Receiver pair (E-puck.proto: EPUCK_EMITTER,
EPUCK_RECEIVER, default fields).

What the receiver gets, as in Webots:
  - whether a packet arrived (the other robot emitted),
  - the emitter direction, expressed in the receiver's own frame
    (Webots Receiver.getEmitterDirection) -> here as a bearing angle,
  - the signal strength = 1 / distance^2 (Webots Receiver.getSignalStrength).

Timing, as in Webots: a packet sent during one control step is received at
the next control step (delay_control_steps = 1).

Range: Webots Emitter default range is -1 (unlimited); config "range": null
keeps it unlimited, a number limits it (metres).

The signal carries no content: it is one bit (on / off). All the
information a receiver can use is that the partner is emitting, and from
which direction and how strongly the packet arrives.

Ablation modes (evaluation only; evolution always uses "normal"):
  normal   - as above
  off      - nothing is ever delivered (robots are deaf)
  random   - packets arrive exactly when they would, but the bearing is
             uniformly random: keeps "the partner is emitting", destroys
             "where it is"
"""
from collections import deque

import numpy as np


class SignalChannel:
    MODES = ("normal", "off", "random")

    def __init__(self, n_robots=2, delay_control_steps=1, max_range=None, mode="normal", rng=None):
        assert mode in self.MODES, f"signal mode must be one of {self.MODES}"
        self.n = n_robots
        self.delay = delay_control_steps
        self.range = max_range
        self.mode = mode
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        # queue of emit-flag vectors; the oldest is what is delivered now
        self.queue = deque([np.zeros(self.n, dtype=bool) for _ in range(self.delay)])

    def step(self, emitting, poses):
        """
        emitting: bool per robot, emitted during this control step.
        poses:    (x, y, yaw) per robot at reception time.
        Returns, per robot, a dict: received (bool), bearing (rad, robot
        frame, 0 = straight ahead, + = left), strength (1/d^2), distance (m).
        """
        self.queue.append(np.asarray(emitting, dtype=bool).copy())
        delivered = self.queue.popleft()
        if self.mode == "off":
            delivered = np.zeros(self.n, dtype=bool)

        out = []
        for i in range(self.n):
            msg = {"received": False, "bearing": 0.0, "strength": 0.0, "distance": 0.0}
            xi, yi, yawi = poses[i]
            for j in range(self.n):
                if j == i or not delivered[j]:
                    continue
                dx, dy = poses[j][0] - xi, poses[j][1] - yi
                dist = float(np.hypot(dx, dy))
                if self.range is not None and dist > self.range:
                    continue
                bearing = (np.arctan2(dy, dx) - yawi + np.pi) % (2 * np.pi) - np.pi
                if self.mode == "random":
                    bearing = float(self.rng.uniform(-np.pi, np.pi))
                msg = {"received": True, "bearing": float(bearing),
                       "strength": 1.0 / max(dist, 1e-3) ** 2, "distance": dist}
            out.append(msg)
        return out
