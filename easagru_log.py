"""
easagru_log.py - CSV log of what the brains do, one row per robot per
control step (long format). Read by easagru_plot.py and easagru_live_plot.py.

Only used by the GUI standalone (and later by evaluation scripts); the
evolution runs do not log per step.
"""
import csv

from easagru_brain import CHANNEL_NAMES

COLUMNS = (["episode", "step", "time_s", "robot", "x", "y", "yaw", "left_cmd", "right_cmd", "on_food", "emit",
            "signal_received", "signal_bearing", "fear", "hunger", "gru_bid"]
           + [f"gpi_{n}" for n in CHANNEL_NAMES]
           + [f"thalamus_{n}" for n in CHANNEL_NAMES])


class StepLogger:
    def __init__(self, path, flush_every=20):
        self.path = path
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(COLUMNS)
        self.flush_every = flush_every
        self.rows = 0

    def log(self, episode, step, time_s, robot, pose, obs, emit, diag, left=float("nan"), right=float("nan")):
        """left/right: wheel commands as a fraction of max speed (-1..1)."""
        n = len(CHANNEL_NAMES)
        gpi = diag.get("gpi")
        th = diag.get("thalamus")
        row = [episode, step, round(time_s, 4), robot,
               round(pose[0], 4), round(pose[1], 4), round(pose[2], 4),
               round(left, 3), round(right, 3),
               int(obs["on_food"]), int(emit),
               int(obs["signal"]["received"]), round(obs["signal"]["bearing"], 4),
               round(diag.get("fear", float("nan")), 4), round(diag.get("hunger", float("nan")), 4),
               round(diag.get("gru_bid", float("nan")), 4)]
        row += [round(v, 4) for v in gpi] if gpi is not None else [""] * n
        row += [round(v, 4) for v in th] if th is not None else [""] * n
        self.w.writerow(row)
        self.rows += 1
        if self.rows % self.flush_every == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()
