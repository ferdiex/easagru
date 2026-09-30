"""
test_step2_sensors.py - Step 2 validation of the e-puck sensors (headless, lean model).

Checks:
  1. IR distance response: every sensor, pointed at a wall at known
     distances, reads the Webots lookup value (noise off, 1% tolerance).
  2. IR out of range: robot in the arena centre reads the last table value.
  3. IR noise: 500 noisy readings at 3 cm match the Webots noise level.
  4. Robot sees robot: a second e-puck in front is detected by ps0/ps7.
  5. Ground: plain floor vs dark patch (reflectance 0.2), and a patch under
     one side only (gs0 vs gs2 differ).
  6. Timing: cost of one full sensor read, and physics speed with sensors
     read every control period.

Usage:  python3 test_step2_sensors.py
"""
import time

import numpy as np
import pybullet as p

from easagru_world import load_config, setup_physics, build_arena, load_epuck, set_wheels
from easagru_sensors import (EpuckSensors, FloorMap, IR_POS, IR_DIR, IR_LOOKUP, GS_LOOKUP,
                             lookup)

RESULTS = []


def check(name, ok, detail):
    RESULTS.append(ok)
    print(f"{name:<34} {detail}  -> {'PASS' if ok else 'FAIL'}")


def main():
    cfg = load_config()
    setup_physics(cfg, gui=False)
    build_arena(cfg)
    wall_x = cfg["world"]["arena_size"] / 2.0  # inner face of the east wall
    floor = FloorMap(cfg["floor"]["reflectance"])
    robot = load_epuck(cfg, (0, 0), 0, lean=True)
    for _ in range(120):
        p.stepSimulation()  # settle so the base height is the real resting height
    z_rest = p.getBasePositionAndOrientation(robot["id"])[0][2] - 0.015

    exact = EpuckSensors(floor, noise=False)
    noisy = EpuckSensors(floor, noise=True, rng=np.random.default_rng(0))

    # 1. IR response per sensor vs horizontal distance to the wall
    worst = 0.0
    for i in range(8):
        ang = np.arctan2(IR_DIR[i, 1], IR_DIR[i, 0])
        yaw = -ang  # rotate robot so sensor i points along +x
        c, s = np.cos(yaw), np.sin(yaw)
        sensor_x = c * IR_POS[i, 0] - s * IR_POS[i, 1]
        sensor_y = s * IR_POS[i, 0] + c * IR_POS[i, 1]
        horiz = np.hypot(IR_DIR[i, 0], IR_DIR[i, 1])
        for d_h in (0.008, 0.012, 0.02, 0.03, 0.045, 0.06):
            p.resetBasePositionAndOrientation(robot["id"], [wall_x - d_h - sensor_x, -sensor_y, z_rest + 0.015],
                                              p.getQuaternionFromEuler([0, 0, yaw]))
            p.performCollisionDetection()
            expected, _ = lookup(d_h / horiz, IR_LOOKUP)
            got = exact.read(robot)["ir"][i]
            worst = max(worst, abs(got - expected) / expected)
    check("[1] IR response, 8 sensors x 6 dist", worst < 0.01, f"worst error {100*worst:.2f}%")

    # 2. Out of range
    p.resetBasePositionAndOrientation(robot["id"], [0, 0, z_rest + 0.015], p.getQuaternionFromEuler([0, 0, 0]))
    p.performCollisionDetection()
    ir = exact.read(robot)["ir"]
    check("[2] IR out of range (centre)", np.allclose(ir, IR_LOOKUP[-1, 1]), f"values {np.round(ir, 1)}")

    # 3. Noise at 3 cm, sensor ps7 (front-left)
    i = 7
    ang = np.arctan2(IR_DIR[i, 1], IR_DIR[i, 0]); yaw = -ang
    c, s = np.cos(yaw), np.sin(yaw)
    sx = c * IR_POS[i, 0] - s * IR_POS[i, 1]; sy = s * IR_POS[i, 0] + c * IR_POS[i, 1]
    horiz = np.hypot(IR_DIR[i, 0], IR_DIR[i, 1])
    p.resetBasePositionAndOrientation(robot["id"], [wall_x - 0.03 * horiz - sx, -sy, z_rest + 0.015],
                                      p.getQuaternionFromEuler([0, 0, yaw]))
    p.performCollisionDetection()
    samples = np.array([noisy.read(robot)["ir"][i] for _ in range(500)])
    exp_v, exp_sd = lookup(0.03, IR_LOOKUP)
    ok = abs(samples.mean() - exp_v) / exp_v < 0.02 and abs(samples.std() - exp_sd) / exp_sd < 0.25
    check("[3] IR noise at 3 cm (ps7)", ok,
          f"mean {samples.mean():.1f} (exp {exp_v:.1f}), sd {samples.std():.2f} (exp {exp_sd:.2f})")

    # 4. Robot sees robot: second e-puck 3 cm (surface to surface) ahead
    other = load_epuck(cfg, (0.2, 0.0), np.pi, lean=True)
    p.resetBasePositionAndOrientation(robot["id"], [0.0, 0.0, z_rest + 0.015], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(other["id"], [0.074 + 0.03, 0.0, z_rest + 0.015],
                                      p.getQuaternionFromEuler([0, 0, np.pi]))
    p.performCollisionDetection()
    r = exact.read(robot)
    ok = r["ir"][0] > 100 and r["ir"][7] > 100 and r["ir"][3] < 70
    check("[4] Robot detects other robot", ok,
          f"ps0={r['ir'][0]:.0f} ps7={r['ir'][7]:.0f} (rear ps3={r['ir'][3]:.0f})")
    p.removeBody(other["id"])

    # 5. Ground sensors
    p.resetBasePositionAndOrientation(robot["id"], [0.0, 0.0, z_rest + 0.015], p.getQuaternionFromEuler([0, 0, 0]))
    p.performCollisionDetection()
    g_plain = exact.read(robot)["ground"]
    exp_plain, _ = lookup(0.0033, GS_LOOKUP)
    floor.add_patch(0.03, 0.0, 0.05, 0.2)           # patch under all three sensors
    g_patch = exact.read(robot)["ground"]
    floor.patches.clear()
    floor.add_patch(0.03, 0.035, 0.03, 0.2)         # patch covers gs0 (left) only
    g_side = exact.read(robot)["ground"]
    floor.patches.clear()
    ok = (np.allclose(g_plain, exp_plain, rtol=0.03) and np.allclose(g_patch, 0.2 * g_plain, rtol=0.01)
          and g_side[0] < 0.3 * g_side[2] and g_side[1] > 0.9 * g_plain[1])
    check("[5] Ground: floor / patch / one side", ok,
          f"floor {np.round(g_plain)} (exp ~{exp_plain:.0f}), patch {np.round(g_patch)}, "
          f"left-only {np.round(g_side)}")

    # 6. Timing
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        noisy.read(robot)
    per_read = (time.perf_counter() - t0) / n
    period = cfg["sensors"]["control_period_steps"]
    set_wheels(robot, 3.0, 2.5)
    steps = 240 * 5
    t0 = time.perf_counter()
    for k in range(steps):
        p.stepSimulation()
        if k % period == 0:
            noisy.read(robot)
    el = time.perf_counter() - t0
    print(f"[6] Timing                           one read = {1e6*per_read:.0f} us;  "
          f"physics + sensors every {period} steps: {steps/el:,.0f} steps/s (x{steps/240/el:.1f} real time)")

    p.disconnect()
    print("\nSTEP 2:", "ALL PASS" if all(RESULTS) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
