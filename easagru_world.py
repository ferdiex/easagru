"""
easagru_world.py -- shared PyBullet setup for the EASA-GRU project.

One place for: config loading, physics setup, arena construction, robot
loading and wheel commands. Both the GUI standalone and the headless
tests import from here, so they always run exactly the same physics.

Step 1 scope: empty square arena + e-puck. Sensors and food come later.
"""
import json
import os

import numpy as np
import pybullet as p
import pybullet_data

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "easagru_config.json")


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return json.load(f)


def setup_physics(cfg, gui=False):
    """Connect to PyBullet and set global physics parameters. Returns client id."""
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, cfg["world"]["gravity"])
    p.setTimeStep(cfg["world"]["timestep"])
    p.setPhysicsEngineParameter(numSolverIterations=cfg["world"]["solver_iterations"])
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    return client


def set_camera(distance=1.2, yaw=0.0, pitch=-75.0, target=(0.0, 0.0, 0.0)):
    """Place the GUI camera. Call AFTER the world and robots are loaded."""
    p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch,
                                 cameraTargetPosition=list(target))


WALL_COLOR = (0.9, 0.2, 0.27, 1.0)  # same red as the Khepera arena (bg90s_arena.py)


def _make_wall(center, half_extents, color=WALL_COLOR):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=center)
    p.changeDynamics(wall, -1, lateralFriction=0.5)
    return wall


def build_arena(cfg):
    """Floor plane plus four walls enclosing a square arena centred at the origin."""
    w = cfg["world"]
    # Standard PyBullet checkered plane (the look of the first version).
    # A separate visual floor slab was tried and caused z-fighting stripes.
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(plane, -1, lateralFriction=w["floor_friction"])
    half = w["arena_size"] / 2.0
    t = w["wall_thickness"] / 2.0
    h = w["wall_height"] / 2.0
    walls = [
        _make_wall([half + t, 0, h], [t, half + 2 * t, h]),
        _make_wall([-half - t, 0, h], [t, half + 2 * t, h]),
        _make_wall([0, half + t, h], [half, t, h]),
        _make_wall([0, -half - t, h], [half, t, h]),
    ]
    return {"plane": plane, "walls": walls}


def add_patch_visual(x, y, radius, rgba=(0.15, 0.15, 0.15, 1.0)):
    """Draw a floor patch (visual only, no collision: robots drive over it and
    rays pass through it; its effect on the ground sensors comes from FloorMap).

    It is drawn 5 mm tall on purpose: thinner patches (0.7-3 mm) showed
    z-fighting stripes in the OpenGL view. Physics and sensors never see it.
    """
    height = 0.005
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=list(rgba))
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[x, y, height / 2])


def _link_index_by_name(body_id):
    mapping = {}
    for j in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, j)
        mapping[info[12].decode("utf-8")] = j
    return mapping


def load_epuck(cfg, pos_xy=(0.0, 0.0), yaw=0.0, lean=False):
    """Load the e-puck, apply friction/damping, return a handle dict.

    lean=True loads the physics-only model (no decorative links): use it for
    every headless run (tests, evolution). Physics is identical; it is faster.
    """
    r = cfg["robot"]
    urdf = r["urdf_lean_path"] if lean else r["urdf_path"]
    if not os.path.isabs(urdf):
        urdf = os.path.join(HERE, urdf)
    orn = p.getQuaternionFromEuler([0, 0, yaw])
    body = p.loadURDF(urdf, [pos_xy[0], pos_xy[1], 0.0005], orn, useFixedBase=False)

    links = _link_index_by_name(body)
    fr = r["friction"]
    p.changeDynamics(body, -1, lateralFriction=fr["chassis"],
                     linearDamping=r["linear_damping"], angularDamping=r["angular_damping"])
    p.changeDynamics(body, links["left_wheel"], lateralFriction=fr["wheel"])
    p.changeDynamics(body, links["right_wheel"], lateralFriction=fr["wheel"])
    p.changeDynamics(body, links["caster_front"], lateralFriction=fr["caster"])
    p.changeDynamics(body, links["caster_rear"], lateralFriction=fr["caster"])

    if not lean:
        apply_colors(body, links)

    handle = {
        "id": body,
        "left": links["left_wheel"],
        "right": links["right_wheel"],
        "max_angvel": r["max_wheel_angvel"],
        "force": r["wheel_motor_force"],
    }
    set_wheels(handle, 0.0, 0.0)
    return handle


def apply_colors(body, links):
    """Re-apply every link colour explicitly (the macOS GUI may ignore URDF materials)."""
    import sys
    sys.path.insert(0, os.path.join(HERE, "robots"))
    from make_epuck_urdf import COLORS, color_for_link
    p.changeVisualShape(body, -1, rgbaColor=list(COLORS["base"]))
    for name, idx in links.items():
        color = color_for_link(name)
        if color is not None:
            p.changeVisualShape(body, idx, rgbaColor=list(color))


def set_wheels(handle, left, right):
    """Wheel commands in rad/s, clipped to the e-puck limit (6.28 rad/s)."""
    m = handle["max_angvel"]
    left = float(np.clip(left, -m, m))
    right = float(np.clip(right, -m, m))
    p.setJointMotorControl2(handle["id"], handle["left"], p.VELOCITY_CONTROL,
                            targetVelocity=left, force=handle["force"])
    p.setJointMotorControl2(handle["id"], handle["right"], p.VELOCITY_CONTROL,
                            targetVelocity=right, force=handle["force"])


def get_pose(handle):
    """Return (x, y, yaw, roll, pitch, z) of the robot base."""
    pos, orn = p.getBasePositionAndOrientation(handle["id"])
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    return pos[0], pos[1], yaw, roll, pitch, pos[2]
