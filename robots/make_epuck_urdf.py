"""
make_epuck_urdf.py - generates robots/epuck.urdf.

Why a generator: the PyBullet GUI on macOS (Metal) mixes up colours when a
link has several <visual> elements (green rendered black, grey rendered
yellow, wheels all black). So every decorative piece is its own link with a
single visual, attached by a fixed joint. Physics is untouched: only the
base (collision cylinder + Webots mass/inertia) and the two wheels collide.

All measurements come from webots_ref/E-puck.proto (e-puck v1); see the
header comment written into the URDF for details.

Writes robots/epuck.urdf (GUI, decorated) and robots/epuck_lean.urdf
(headless/evolution, physics only).

Usage:  python3 robots/make_epuck_urdf.py
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Colours (RGBA). Also re-applied at load time by easagru_world.apply_colors.
COLORS = {
    "base": (0.55, 0.55, 0.58, 1.0),          # lower chassis, mid grey
    # Upper ring. The 4th number is opacity: 0 = invisible, 1 = solid.
    # Raise it (e.g. 0.65) for a less translucent shell, then re-run this script.
    "shell": (0.30, 0.30, 0.33, 0.85),        # upper ring, dark translucent grey
    "pcb": (0.08, 0.45, 0.15, 1.0),           # top circuit board, green
    "speaker": (0.08, 0.08, 0.08, 1.0),
    "front_led": (0.90, 0.10, 0.10, 1.0),
    "ir": (0.05, 0.05, 0.05, 1.0),
    "left_wheel": (0.92, 0.92, 0.92, 1.0),    # white tyre
    "right_wheel": (0.92, 0.92, 0.92, 1.0),
    "ring": (0.85, 0.10, 0.10, 1.0),          # red ring on the wheel
    "hub": (0.95, 0.80, 0.30, 1.0),
    "caster": (0.70, 0.70, 0.70, 1.0),
}

# IR sensor positions (x, y) at z = 0.033, from E-puck.proto ps0..ps7.
IR_XY = [(0.03, -0.01), (0.022, -0.025), (0.0, -0.031), (-0.03, -0.015),
         (-0.03, 0.015), (0.0, 0.031), (0.022, 0.025), (0.03, 0.01)]

TINY = '''    <inertial>
      <mass value="0.0001"/>
      <inertia ixx="1e-9" iyy="1e-9" izz="1e-9" ixy="0" ixz="0" iyz="0"/>
    </inertial>'''


def rgba(key):
    return " ".join(f"{c:g}" for c in COLORS[key])


def deco_link(name, color_key, xyz, geometry, rpy="0 0 0", parent="base"):
    """A visual-only link attached to `parent` by a fixed joint."""
    return f'''
  <link name="{name}">
    <visual>
      <origin xyz="{xyz}" rpy="{rpy}"/>
      <geometry>{geometry}</geometry>
      <material name="m_{name}"><color rgba="{rgba(color_key)}"/></material>
    </visual>
{TINY}
  </link>
  <joint name="{name}_joint" type="fixed">
    <parent link="{parent}"/>
    <child link="{name}"/>
    <origin xyz="0 0 0"/>
  </joint>'''


def wheel(side, y, deco=True):
    sgn = 1 if y > 0 else -1
    parts = f'''
  <link name="{side}_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><cylinder length="0.005" radius="0.02"/></geometry>
      <material name="m_{side}_wheel"><color rgba="{rgba(side + '_wheel')}"/></material>
    </visual>
    <collision>
      <!-- radius 19 mm on purpose: Bullet adds ~1 mm collision margin to
           cylinders, so the effective rolling radius is the real 20 mm. -->
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><cylinder length="0.005" radius="0.019"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.005"/>
      <inertia ixx="5.1e-7" iyy="1.0e-6" izz="5.1e-7" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="{side}_wheel_joint" type="continuous">
    <parent link="base"/>
    <child link="{side}_wheel"/>
    <origin xyz="0 {y} 0.02" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>'''
    if not deco:
        return parts
    # Red ring and hub sit slightly outboard so they show on the outer face.
    parts += deco_link(f"{side}_ring", "ring", f"0 {sgn * 0.0006:.4f} 0",
                       '<cylinder length="0.0052" radius="0.0135"/>', rpy="1.5708 0 0",
                       parent=f"{side}_wheel")
    parts += deco_link(f"{side}_ring_inner", f"{side}_wheel", f"0 {sgn * 0.0012:.4f} 0",
                       '<cylinder length="0.0054" radius="0.009"/>', rpy="1.5708 0 0",
                       parent=f"{side}_wheel")
    parts += deco_link(f"{side}_hub", "hub", f"0 {sgn * 0.0018:.4f} 0",
                       '<cylinder length="0.0056" radius="0.0035"/>', rpy="1.5708 0 0",
                       parent=f"{side}_wheel")
    return parts


def caster(name, x):
    return f'''
  <link name="{name}">
    <visual>
      <origin xyz="{x} 0 0.0035"/>
      <geometry><sphere radius="0.003"/></geometry>
      <material name="m_{name}"><color rgba="{rgba('caster')}"/></material>
    </visual>
    <collision>
      <origin xyz="{x} 0 0.0035"/>
      <geometry><sphere radius="0.003"/></geometry>
    </collision>
    <inertial>
      <origin xyz="{x} 0 0.0035"/>
      <mass value="0.001"/>
      <inertia ixx="4e-9" iyy="4e-9" izz="4e-9" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="{name}_joint" type="fixed">
    <parent link="base"/>
    <child link="{name}"/>
    <origin xyz="0 0 0"/>
  </joint>'''


HEADER = '''<?xml version="1.0" ?>
<!--
  epuck.urdf - GENERATED by robots/make_epuck_urdf.py, do not edit by hand.

  e-puck (version 1) rebuilt for PyBullet from the official Webots model
  (webots_ref/E-puck.proto, cyberbotics/webots, master).

  From E-puck.proto:
  - Body collision: cylinder r = 37 mm, h = 45 mm, centred at z = 25 mm.
  - Body mass 0.15 kg, centre of mass (0, 0, 0.015),
    inertia diag(9.78585e-5, 8.64333e-5, 8.74869e-5).
  - Wheels: r = 20 mm, width 5 mm, mass 5 g, axle at z = 20 mm,
    y = +/-26 mm (axle 52 mm), hinge axis 0 1 0.
  - IR sensor poses ps0..ps7 (drawn as small black boxes).
  - Frame: x forward, y left, z up.

  Our additions / adjustments:
  - Two low-friction casters (front, rear, 0.5 mm above the floor) replace
    the sliding box Webots uses under the body.
  - Wheel collision radius 19 mm (visual 20 mm) cancels Bullet's ~1 mm
    cylinder margin; without it the robot rolled 5% too fast.
  - Visual pieces are separate links (one visual each) so colours render
    correctly in the macOS GUI. They have negligible mass (0.1 g each).
-->
<robot name="epuck">
'''

BASE = f'''
  <link name="base">
    <visual>
      <!-- lower chassis: the 0.05 x 0.04 m box of E-puck.proto; narrow, so
           the wheels are exposed on the sides as on the real robot -->
      <origin xyz="0 0 0.013"/>
      <geometry><box size="0.05 0.04 0.02"/></geometry>
      <material name="m_base"><color rgba="{rgba('base')}"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.025"/>
      <geometry><cylinder length="0.045" radius="0.037"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.015"/>
      <mass value="0.15"/>
      <inertia ixx="9.78585e-5" iyy="8.64333e-5" izz="8.74869e-5" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>'''


def build(lean=False):
    """Full model for the GUI, or lean=True: physics only (no decorative links),
    used for headless evolution runs where nothing is rendered."""
    if lean:
        return "".join([HEADER.replace("epuck.urdf - GENERATED", "epuck_lean.urdf - GENERATED")
                        .replace('<robot name="epuck">', '<robot name="epuck_lean">'),
                        BASE, wheel("left", 0.026, deco=False), wheel("right", -0.026, deco=False),
                        caster("caster_front", 0.025), caster("caster_rear", -0.025),
                        "\n</robot>\n"])
    parts = [HEADER, BASE]
    parts.append(deco_link("shell", "shell", "0 0 0.034",
                           '<cylinder length="0.024" radius="0.037"/>'))
    parts.append(deco_link("pcb", "pcb", "0 0 0.0470",
                           '<cylinder length="0.0016" radius="0.035"/>'))
    parts.append(deco_link("speaker", "speaker", "-0.004 0 0.0490",
                           '<cylinder length="0.003" radius="0.009"/>'))
    parts.append(deco_link("front_led", "front_led", "0.026 0 0.0485",
                           '<box size="0.006 0.006 0.002"/>'))
    for i, (x, y) in enumerate(IR_XY):
        a = math.atan2(y, x)
        parts.append(deco_link(f"ir{i}", "ir", f"{0.0368 * math.cos(a):.4f} {0.0368 * math.sin(a):.4f} 0.033",
                               '<box size="0.003 0.007 0.004"/>', rpy=f"0 0 {a:.4f}"))
    parts.append(wheel("left", 0.026))
    parts.append(wheel("right", -0.026))
    parts.append(caster("caster_front", 0.025))
    parts.append(caster("caster_rear", -0.025))
    parts.append("\n</robot>\n")
    return "".join(parts)


def color_for_link(name):
    """Colour key for a link name (used by easagru_world.apply_colors)."""
    if name.startswith("ir"):
        return COLORS["ir"]
    if name.endswith("_ring_inner"):
        return COLORS[name.split("_")[0] + "_wheel"]
    if name.endswith("_ring"):
        return COLORS["ring"]
    if name.endswith("_hub"):
        return COLORS["hub"]
    if name.startswith("caster"):
        return COLORS["caster"]
    return COLORS.get(name)


if __name__ == "__main__":
    for fname, lean in (("epuck.urdf", False), ("epuck_lean.urdf", True)):
        path = os.path.join(HERE, fname)
        with open(path, "w") as f:
            f.write(build(lean=lean))
        print("wrote", path)
