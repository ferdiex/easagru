"""
test_step1_kinematics.py -- Step 1 validation of the e-puck model (headless).

Compares PyBullet behaviour against the kinematics implied by the Webots
e-puck (wheel radius 20 mm, axle 52 mm, max wheel speed 6.28 rad/s):
  expected straight speed = 6.28 * 0.02            = 0.1256 m/s
  expected spin rate      = 2 * 0.1256 / 0.052     = 4.83 rad/s

Checks:
  1. Rest:     robot settles, stays level (tilt < 3 deg), does not drift.
  2. Straight: speed within 10% of expected, heading drift < 5 deg over 3 s.
  3. Spin:     yaw rate within 10% of expected, centre moves < 1 cm.
  4. Timing:   physics steps per second (for the Mac mini cost estimate).

Usage:  python3 test_step1_kinematics.py          (lean model, as in evolution)
        python3 test_step1_kinematics.py --full   (decorated GUI model)
"""
import sys
import time

import numpy as np
import pybullet as p

from easagru_world import load_config, setup_physics, build_arena, load_epuck, set_wheels, get_pose

TOL = 0.10
LEAN = "--full" not in sys.argv  # default: lean model, as used in evolution


def run_steps(n):
    for _ in range(n):
        p.stepSimulation()


def unwrap_yaw_track(handle, n_steps, sample_every=4):
    """Step n_steps, return accumulated (unwrapped) yaw change."""
    prev = get_pose(handle)[2]
    total = 0.0
    for i in range(n_steps):
        p.stepSimulation()
        if i % sample_every == 0:
            yaw = get_pose(handle)[2]
            d = (yaw - prev + np.pi) % (2 * np.pi) - np.pi
            total += d
            prev = yaw
    return total


def main():
    cfg = load_config()
    dt = cfg["world"]["timestep"]
    r = cfg["robot"]
    v_exp = r["max_wheel_angvel"] * r["wheel_radius"]
    w_exp = 2 * v_exp / r["axle_length"]
    steps_per_s = int(round(1.0 / dt))

    print(f"Model: {'lean (physics only)' if LEAN else 'full (decorated)'}")
    setup_physics(cfg, gui=False)
    build_arena(cfg)
    results = []

    # 1. Rest
    robot = load_epuck(cfg, (0.0, 0.0), 0.0, lean=LEAN)
    run_steps(steps_per_s)  # settle 1 s
    x0, y0, _, roll, pitch, z = get_pose(robot)
    run_steps(steps_per_s)
    x1, y1, _, roll, pitch, z = get_pose(robot)
    tilt = np.degrees(max(abs(roll), abs(pitch)))
    drift = np.hypot(x1 - x0, y1 - y0)
    ok = tilt < 3.0 and drift < 0.002
    results.append(ok)
    print(f"[1] Rest      tilt={tilt:.2f} deg  drift={drift*1000:.2f} mm  z={z*1000:.1f} mm  -> {'PASS' if ok else 'FAIL'}")

    # 2. Straight line (start at -0.35 so 3 s at ~0.126 m/s stays inside the arena)
    p.resetBasePositionAndOrientation(robot["id"], [-0.35, 0, 0.0005], p.getQuaternionFromEuler([0, 0, 0]))
    set_wheels(robot, 0, 0)
    run_steps(steps_per_s // 2)
    set_wheels(robot, r["max_wheel_angvel"], r["max_wheel_angvel"])
    run_steps(steps_per_s // 2)  # accelerate
    xa, ya, yaw_a, *_ = get_pose(robot)
    run_steps(3 * steps_per_s)
    xb, yb, yaw_b, roll, pitch, _ = get_pose(robot)
    v = np.hypot(xb - xa, yb - ya) / 3.0
    heading_drift = np.degrees(abs((yaw_b - yaw_a + np.pi) % (2 * np.pi) - np.pi))
    tilt = np.degrees(max(abs(roll), abs(pitch)))
    ok = abs(v - v_exp) / v_exp < TOL and heading_drift < 5.0
    results.append(ok)
    print(f"[2] Straight  v={v:.4f} m/s (expected {v_exp:.4f}, {100*(v-v_exp)/v_exp:+.1f}%)  "
          f"heading drift={heading_drift:.2f} deg  tilt={tilt:.2f} deg  -> {'PASS' if ok else 'FAIL'}")

    # 3. Spin in place
    set_wheels(robot, 0, 0)
    p.resetBasePositionAndOrientation(robot["id"], [0, 0, 0.0005], p.getQuaternionFromEuler([0, 0, 0]))
    run_steps(steps_per_s // 2)
    set_wheels(robot, -r["max_wheel_angvel"], r["max_wheel_angvel"])
    run_steps(steps_per_s // 2)
    xs, ys, *_ = get_pose(robot)
    total_yaw = unwrap_yaw_track(robot, 3 * steps_per_s)
    xe, ye, *_ = get_pose(robot)
    w = abs(total_yaw) / 3.0
    centre_move = np.hypot(xe - xs, ye - ys)
    ok = abs(w - w_exp) / w_exp < TOL and centre_move < 0.01
    results.append(ok)
    print(f"[3] Spin      w={w:.3f} rad/s (expected {w_exp:.3f}, {100*(w-w_exp)/w_exp:+.1f}%)  "
          f"centre moved={centre_move*1000:.1f} mm  -> {'PASS' if ok else 'FAIL'}")

    # 4. Timing (one robot, empty arena)
    set_wheels(robot, 3.0, 2.0)
    n = 5 * steps_per_s
    t0 = time.perf_counter()
    run_steps(n)
    el = time.perf_counter() - t0
    print(f"[4] Timing    {n/el:,.0f} physics steps/s  ({el/ (n*dt):.4f} s wall per s simulated, "
          f"real-time factor x{(n*dt)/el:.1f})")

    p.disconnect()
    print("\nSTEP 1:", "ALL PASS" if all(results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
