"""Stage 3 add-on: the real2sim lab scene (g1_xr_sim.py --scene lab, i.e. --addon lab_scene).

Our lab's desk task, rebuilt from measurements, photos and the real recordings: put the bubble-wrapped box
into the cardboard box glued to the desk.
  * scene_lab.xml: desk, cardboard box (static), wrapped box (free body), floor; scene_room.xml: the room
    around it (workbench + printers, cabinet, walls; assets/lab/make_room.py).
  * cam_head: mono head camera at the G1's D435 mount, pose and FOV calibrated against the real videos.
  * Uses the wrist_cameras add-on (gloved 5010 hands + D405 wrist cameras).
  * Photometric calibration (calib/photometric.json): material colours, light level, per-camera gain,
    plus lens blur and sensor noise like the real 640x480 streams.
  * LabTask: object start poses (randomised with --randomize), success check (rt/rewards_state).
  * Wrapped box: size set per run with --wrap-scale; its four sides are spring-loaded panels, so it gives
    about a centimetre when squeezed (bubble wrap over cardboard), unless --rigid-box.
  * --randomize, on every reset: box position / facing side / upright or upside down, desk position
    relative to the robot, cardboard box jitter, light level and tint, camera pose and field of view.
    A reset command may carry a JSON scene description (sample_scene) to ask for a specific scene.

Camera names match the real dataset keys (observation.images.cam_head / cam_left_wrist / cam_right_wrist).
"""

import json
import os
import tempfile

import cv2
import mujoco
import numpy as np
from wrist_cameras import mat_to_quat

REQUIRES = ["wrist_cameras"]
PROVIDES_SCENE = True

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_XML = os.path.join(HERE, "scene_lab.xml")
MESH_DIR = os.path.join(HERE, "assets", "lab", "mesh")

# Photometric calibration against the real recordings (tools/fit_materials.py): material colours, light
# level and per-camera RGB gain. Loaded when present.
PHOTOMETRIC_JSON = os.path.join(HERE, "calib", "photometric.json")
PHOTOMETRIC = json.load(open(PHOTOMETRIC_JSON)) if os.path.exists(PHOTOMETRIC_JSON) else {}
# Image softness and sensor noise of the real streams (640x480): Gaussian blur sigma [px] and noise std
# [0-255] per camera, chosen so the sim frames have the real frames' edge sharpness and noise level
# (measured on the recordings: head 136 / 0.8, left wrist 154 / 0.9, right wrist 127 / 1.0).
CAMERA_BLUR = {"cam_head": 0.75, "cam_left_wrist": 0.25, "cam_right_wrist": 0.6}
CAMERA_NOISE = {"cam_head": 1.0, "cam_left_wrist": 0.5, "cam_right_wrist": 1.2}

HEAD_GEOM_GROUP = 5  # robot head shell, pelvis and legs: hidden from the cameras, shown in the viewer
VIEWER_GEOM_GROUPS = (HEAD_GEOM_GROUP,)
DEBUG_VIEW = {"lookat": [0.35, 0.12, 0.85], "distance": 1.5, "azimuth": 135, "elevation": -25}

# Image streams served to xr_teleoperate / recorders: teleimager topic -> MuJoCo camera names.
LAB_CAMERAS = {
    "head_camera": ["cam_head"],
    "left_wrist_camera": ["cam_left_wrist"],
    "right_wrist_camera": ["cam_right_wrist"],
}

# --- head camera: the G1's head D435i (g1 URDF: d435_joint xyz 0.0576235 0.01753 0.42987, rpy 0 0.8307767 0).
# FOV from the datasheet: RGB 69 x 42 deg at 16:9; the 640x480 stream has fx ~ fy ~ 610 px -> 42.9 deg
# vertical (55.3 horizontal). Pose corrected against the 52 real recordings with each episode's wrapped
# box at its kinematic position and the FOV held fixed (tools/fit_head_full.py, FIXED_FOVY=42.9:
# wrap + desk + glove IoU): 3 cm back, 3 cm up, pitched 52.5 deg (URDF 47.6) - plausibly the torso
# posture while the robot hangs on its gantry.
HEAD_CAM_POS = [0.0576235 - 0.03, 0.01753, 0.42987 + 0.03]
HEAD_CAM_PITCH_DEG = 52.5
HEAD_CAM_FOVY = 42.9

def look_rotation(forward, up_hint, roll_deg=0.0):
    """Frame whose -z looks along `forward` (MuJoCo camera convention), y ~ up_hint, then rolled."""
    z = -forward / np.linalg.norm(forward)
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    return R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def head_camera_quat(pitch_deg):
    th = np.radians(pitch_deg)
    forward = np.array([np.cos(th), 0.0, -np.sin(th)])
    return mat_to_quat(look_rotation(forward, np.array([0.0, 0.0, 1.0])))


def _abs_assets(env, scene_dir):
    """Make mesh/texture paths absolute so they survive attaching into the robot spec."""
    mesh_dir = os.path.join(scene_dir, env.meshdir)
    tex_dir = os.path.join(scene_dir, env.texturedir)
    for m in env.meshes:
        m.file = os.path.join(mesh_dir, m.file)
    for t in env.textures:
        if t.file:
            t.file = os.path.join(tex_dir, t.file)
    env.meshdir = env.texturedir = ""


def add_args(p):
    p.add_argument("--head-fovy", type=float, default=None, help="lab: override the head camera vertical FOV")
    p.add_argument("--randomize", action="store_true",
                   help="lab: random object start poses + lighting/tint/camera jitter on every reset (sim2real)")
    p.add_argument("--seed", type=int, default=None, help="lab: random seed for --randomize")
    p.add_argument("--wrap-scale", type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"),
                   help="lab: size of the wrapped box relative to the calibrated 22.6 x 22.6 x 26.6 cm one "
                        "(the real box is 5-10 %% bigger; fixed for the whole run)")
    p.add_argument("--rigid-box", action="store_true",
                   help="lab: wrapped box as one rigid body, without the spring-loaded side panels")


def build(spec, args, cameras, env_prefix):
    """Attach the lab environment, the calibrated mono head camera and the photometric calibration."""
    env = mujoco.MjSpec.from_file(LAB_XML)
    _abs_assets(env, HERE)
    build_wrapped_box(env, getattr(args, "wrap_scale", None), soft=not getattr(args, "rigid_box", False))
    spec.attach(env, prefix=env_prefix, frame=spec.worldbody.add_frame())
    for attr in ("ambient", "diffuse", "specular"):  # the lab's lighting (attach keeps the robot file's <visual>)
        setattr(spec.visual.headlight, attr, getattr(env.visual.headlight, attr))
    spec.visual.rgba.haze = env.visual.rgba.haze
    spec.visual.quality.shadowsize = env.visual.quality.shadowsize
    apply_photometric(spec)
    for side in ("left", "right"):   # the gloves also touch the wrapped box's spring panels (collision bit 2)
        for geom in spec.body(f"{side}_wrist_yaw_link").geoms:
            if geom.conaffinity:
                geom.conaffinity = int(geom.conaffinity) | 2
    # Friction that holds: with MuJoCo's default soft friction a squeezed box slowly turns in the grip
    # (it ends up hanging below the hands like a pendulum), which a gloved hand on bubble wrap does not do.
    # Elliptic friction cones, a stiffer friction constraint and a few no-slip iterations stop that creep.
    grip = os.environ.get("G1_LAB_GRIP", "firm")
    if grip == "firm":
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.option.noslip_iterations = 4

    # The real head camera looks out through the visor; Unitree's head mesh is a solid shell that the
    # camera (at its calibrated pose) would see from inside. Put it in render group 5: cameras use the
    # default groups 0-2, the desktop viewer turns group 5 on (VIEWER_GEOM_GROUPS).
    for geom in spec.body("torso_link").geoms:
        if geom.meshname == "head_link":
            geom.group = HEAD_GEOM_GROUP
    # Same group for the pelvis and legs: behind the desk no camera sees them, and they are 43 % of the
    # triangles the three cameras would draw 30 times a second (the difference between 20 and 30 fps on an
    # integrated GPU). Their collision geoms are untouched.
    for body in spec.bodies:
        if body.name == "pelvis" or any(k in body.name for k in ("hip", "knee", "ankle")):
            for geom in body.geoms:
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.contype == 0 and geom.conaffinity == 0:
                    geom.group = HEAD_GEOM_GROUP

    # Head camera (mono, like the real recordings) replaces the default stereo pair.
    cam = spec.body("torso_link").add_camera()
    cam.name = "cam_head"
    cam.pos = HEAD_CAM_POS
    cam.quat = head_camera_quat(HEAD_CAM_PITCH_DEG)
    cam.fovy = getattr(args, "head_fovy", None) or HEAD_CAM_FOVY
    return {**cameras, **LAB_CAMERAS}


def make_task(model, prefix, args):
    return LabTask(model, prefix, getattr(args, "randomize", False), getattr(args, "seed", None))


# ----------------------------------------------------------------------------------------------
# Wrapped box model: size per run, spring-loaded side panels
# ----------------------------------------------------------------------------------------------
WRAP_COL_HALF = np.array([0.113, 0.113, 0.133])  # collision half sizes at scale 1 (scene_lab.xml)
WRAP_MASS = 0.35
PANEL_THICK = 0.006    # m, side panel thickness
PANEL_TRAVEL = 0.014   # m, how far a panel can be pushed in before it meets the rigid core
PANEL_STIFFNESS = 2000.0  # N/m: a 20 N squeeze moves a panel 1 cm
PANEL_DAMPING = 15.0      # N s/m, about critical for the panel mass
PANEL_MASS = 0.04
PANEL_NAMES = ("xp", "xn", "yp", "yn")


def build_wrapped_box(env, scale=None, soft=True):
    """Resize the wrapped box (visual mesh, collision, mass) and give it spring-loaded side panels.

    Bubble wrap over cardboard gives when squeezed. The rigid collision box is shrunk on its four sides
    and each side gets a thin plate on a slide joint with a spring: the hands (and the cardboard walls)
    touch the plates, which move in up to PANEL_TRAVEL before the core stops them. The plates are not
    drawn; the visual mesh stays on the core."""
    s = np.ones(3) if scale is None else np.asarray(scale, float)
    half = WRAP_COL_HALF * s
    body = env.body(WRAP)
    env.mesh(WRAP).scale = s
    col = env.geom("wrapped_box_col")
    col.pos = [0, 0, half[2]]
    mass = WRAP_MASS * s.prod()
    core = half.copy()
    if soft:
        core[:2] -= PANEL_THICK + PANEL_TRAVEL
        mass -= 4 * PANEL_MASS
    col.size = core
    body.mass = mass
    body.ipos = [0, 0, half[2]]
    body.inertia = mass / 3 * np.array([half[1] ** 2 + half[2] ** 2, half[0] ** 2 + half[2] ** 2,
                                        half[0] ** 2 + half[1] ** 2])
    if not soft:
        return half
    for name, axis, sign in zip(PANEL_NAMES, (0, 0, 1, 1), (1, -1, 1, -1)):
        n = np.zeros(3)
        n[axis] = sign
        panel = body.add_body(name=f"{WRAP}_panel_{name}", pos=n * (half[axis] - PANEL_THICK / 2) + [0, 0, half[2]])
        panel.add_joint(name=f"{WRAP}_panel_{name}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=-n,
                        range=[0, PANEL_TRAVEL], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                        stiffness=PANEL_STIFFNESS, damping=PANEL_DAMPING)
        size = np.array([half[0] - 0.008, half[1] - 0.008, half[2] - 0.006])  # clear of the corners and the desk
        size[axis] = PANEL_THICK / 2
        # contype 2 / conaffinity 0: the panels only touch the gloves (which get conaffinity bit 2 in build()).
        # Against the desk and the cardboard box the wrapped box is its core, 2 cm smaller on every side:
        # bubble wrap squashes and cardboard walls bow, so a box that overlaps the opening by a centimetre or
        # two still slides in; a rigid full-size box would hang on the rim.
        panel.add_geom(name=f"{WRAP}_panel_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, mass=PANEL_MASS,
                       contype=2, conaffinity=0, condim=4, friction=[1.5, 0.06, 0.001], group=3,
                       rgba=[0.8, 0.2, 0.2, 0.3])
    return half


# ----------------------------------------------------------------------------------------------
# Task: put the bubble-wrapped box into the cardboard box
# ----------------------------------------------------------------------------------------------
WRAP = "wrapped_box"
CARDBOARD = "cardboard_box"
DESK = "desk"
BOX_INNER_HALF = np.array([0.161, 0.241])  # inner half-extents of the cardboard box (x, y in its frame)
BOX_OUTER_HALF = np.array([0.165, 0.245])
WALL_H = 0.18
WRAP_LUMPS = 0.004  # the bubble wrap's lumps stick out this far beyond the collision box

# Ranges of --randomize, per reset. Positions are in the desk's own frame; the whole desk (with both
# boxes) then moves relative to the robot. sample_scene() draws from them.
SCENE = dict(
    desk_x=(-0.05, 0.02),    # m, desk relative to the robot: up to 5 cm nearer, 2 cm farther. Farther than that
    desk_y=(-0.05, 0.02),    # (or more to the left) the arms cannot get the box over the cardboard box's walls
    desk_yaw=2.0,            # deg
    card_xy=0.01,            # m, cardboard box on the desk (glued on the real setup)
    card_yaw=2.0,            # deg
    wrap_gap=(0.14, 0.28),   # m, free space between the wrapped box and the cardboard box: the gloved hand
                             # that goes in there is 8 cm thick
    wrap_yaw=10.0,           # deg, around "one of the four sides squarely faces the robot"
    light_bands=((0.55, 0.8), (0.8, 1.2), (1.2, 1.5)),  # dim, normal, bright: scales every light
    light_each=(0.8, 1.25),  # each ceiling light on top of the level
    light_tint=0.06,         # warm / cool: red and blue move in opposite directions
    fov=0.04,                # each camera's field of view, relative
)
CELLS = 3  # strata per axis for the wrapped box position (a "cell" of the 3 x 3 grid)
# How far from the robot the wrapped box may stand. The arms are short: at grasp height the palms get about
# 43 cm ahead of the waist axis, so a box whose centre is farther than WRAP_REACH cannot be gripped by its
# sides. WRAP_NEAR keeps it off the hands' rest pose just in front of the hips.
WRAP_REACH = 0.44   # m, box centre to the robot's vertical axis, at most
WRAP_NEAR = 0.215   # m, the box's nearest point along x, at least
DESK_FAR_EDGE = 0.225 - 0.02  # m from the desk centre along x: keep the box on the desk


def _in_cell(rng, lo, hi, k, n=CELLS):
    """Uniform in [lo, hi], or in its k-th of n equal parts."""
    if k is not None:
        lo, hi = lo + (hi - lo) * k / n, lo + (hi - lo) * (k + 1) / n
    return float(rng.uniform(lo, hi))


def sample_scene(rng, want=None, ncam=3, nlight=3):
    """One randomised scene as plain numbers. `want` may fix strata (used by the recorder to balance the
    dataset): flip 0/1 (upright / upside down), face 0-3 (which side faces the robot), cell [ix, iy]
    (third of the near-far range, third of the gap range), desk [ix, iy] (0/1: nearer or farther half of its x and y range),
    light 0-2 (dim / normal / bright). Everything else is drawn from SCENE."""
    want = want or {}
    S = SCENE
    cell = want.get("cell", (None, None))
    dx, dy = want.get("desk", (None, None))
    band = int(want["light"]) if "light" in want else int(rng.integers(len(S["light_bands"])))
    tint = float(rng.uniform(-S["light_tint"], S["light_tint"]))
    return dict(
        desk=[_in_cell(rng, *S["desk_x"], dx, 2), _in_cell(rng, *S["desk_y"], dy, 2),
              float(rng.uniform(-S["desk_yaw"], S["desk_yaw"]))],
        card=[float(rng.uniform(-S["card_xy"], S["card_xy"])), float(rng.uniform(-S["card_xy"], S["card_xy"])),
              float(rng.uniform(-S["card_yaw"], S["card_yaw"]))],
        wrap=dict(far=_in_cell(rng, 0.0, 1.0, cell[0]),  # 0 = as near as allowed, 1 = as far as the arms reach
                  gap=_in_cell(rng, *S["wrap_gap"], cell[1]),
                  yaw=float(rng.uniform(-S["wrap_yaw"], S["wrap_yaw"])),
                  face=int(want["face"]) if "face" in want else int(rng.integers(4)),
                  flip=int(want["flip"]) if "flip" in want else int(rng.integers(2))),
        light=dict(band=band, level=float(rng.uniform(*S["light_bands"][band])),
                   each=rng.uniform(*S["light_each"], nlight).tolist(), tint=[1 + tint, 1.0, 1 - tint]),
        fov=rng.uniform(1 - S["fov"], 1 + S["fov"], ncam).tolist(),
    )


def _yaw_extent(half_xy, yaw):
    """Half extent along y of a box with half sizes (hx, hy) turned by yaw."""
    return half_xy[0] * abs(np.sin(yaw)) + half_xy[1] * abs(np.cos(yaw))


def place_objects(scene, wrap_half, desk0, card0, wrap0):
    """World poses for a scene: returns (desk pos, desk yaw), (cardboard pos, yaw), (wrapped box pos, yaw).
    desk0 / card0 / wrap0 are the nominal world positions of scene_lab.xml; the wrapped box position is
    its bottom-face centre when upright, i.e. the point that touches the desk."""
    dx, dy, dyaw = scene["desk"]
    dyaw = np.radians(dyaw)
    c, s = np.cos(dyaw), np.sin(dyaw)

    def to_world(p):  # a point given at the nominal desk pose -> the moved desk
        rel = np.asarray(p, float) - desk0
        return desk0 + [dx, dy, 0.0] + [c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]]

    cx, cy, cyaw = scene["card"]
    cyaw = np.radians(cyaw)
    card = np.array([card0[0] + cx, card0[1] + cy, card0[2]])
    w = scene["wrap"]
    half = wrap_half[:2] + WRAP_LUMPS
    wall_y = card[1] - _yaw_extent(BOX_OUTER_HALF, cyaw)  # outer face of the wall next to the wrapped box
    turn = np.radians(90.0 * w["face"] + w["yaw"])
    yaw, gap = turn, w["gap"]   # yaw in the desk frame; refined below, it depends on where the box ends up
    xs = desk0[0] + np.linspace(-0.2, DESK_FAR_EDGE, 60)
    for _ in range(40):
        ext_x, ext_y = _yaw_extent(half[::-1], yaw), _yaw_extent(half, yaw)
        y = wall_y - gap - ext_y
        world = np.array([to_world([x, y, wrap0[2]]) for x in xs])
        ok = (world[:, 0] - ext_x >= WRAP_NEAR) & (np.hypot(world[:, 0], world[:, 1]) <= WRAP_REACH) & \
             (xs + ext_x <= desk0[0] + DESK_FAR_EDGE)
        if not ok.any():      # too far to the side for the arms: move it toward the cardboard box
            gap -= 0.01
            continue
        lo, hi = xs[ok][0], xs[ok][-1]
        wrap = to_world([lo + w["far"] * (hi - lo), y, wrap0[2]])
        # One of the box's sides squarely faces the robot (plus the yaw jitter), wherever the box stands.
        new = np.arctan2(wrap[1], wrap[0]) + turn - dyaw
        if abs(new - yaw) < 1e-4:
            break
        yaw = new
    return (to_world(desk0), dyaw), (to_world(card), dyaw + cyaw), (wrap, dyaw + yaw)


# Arm rest pose: the mean first-frame observation.state of the 52 real episodes (ARM_START_REAL_MEAN,
# std ~0.1 rad) with both hands pulled ~8 cm back (+2 cm up, 3 cm outward) by IK, same orientation:
# at the exact mean pose the gloved hands sit inside the wrapped box / on the cardboard wall (on the
# real robot the operator places the objects around the hands); pulled back, no layout is disturbed.
ARM_START_REAL_MEAN = [0.188, -0.055, -0.177, -0.164, 0.069, -0.289, 0.079,
                       0.155, 0.026, 0.031, -0.247, -0.160, -0.074, -0.143]
ARM_START = [0.611, 0.042, -0.177, -0.544, -0.025, -0.33, -0.028,
             0.56, -0.026, -0.031, -0.607, -0.029, -0.123, -0.011]


def _quat_yaw(yaw, flip=False):
    """Rotation about z by yaw; flip = turned upside down first (180 deg about x)."""
    c, s = np.cos(yaw / 2), np.sin(yaw / 2)
    return [0.0, c, s, 0.0] if flip else [c, 0.0, 0.0, s]


class LabTask:
    """Success check and (domain-)randomised resets for the lab scene."""

    arm_home = ARM_START  # held by the sim until a controller connects, and set on reset

    def __init__(self, model, prefix, randomize=False, seed=None):
        self.m, self.prefix, self.randomize = model, prefix, randomize
        self.rng = np.random.default_rng(seed)
        self.wrap = model.body(prefix + WRAP).id
        self.box = model.body(prefix + CARDBOARD).id
        self.desk = model.body(prefix + DESK).id
        self.box_pos0 = model.body_pos[self.box].copy()
        self.desk_pos0 = model.body_pos[self.desk].copy()
        qa, _ = self._free_qpos(self.wrap)
        self.wrap_pos0 = model.qpos0[qa:qa + 3].copy()
        # Size of the wrapped box as built (--wrap-scale): its inertial frame sits at the centre.
        self.wrap_half = np.array([0.0, 0.0, model.body_ipos[self.wrap][2]])
        core = model.geom(prefix + "wrapped_box_col")
        soft = any(model.body(i).name.startswith(prefix + WRAP + "_panel") for i in range(model.nbody))
        self.wrap_half[:2] = core.size[:2] + (PANEL_THICK + PANEL_TRAVEL if soft else 0.0)
        # Nominal values to randomise around.
        self.light_diffuse0 = model.light_diffuse.copy()
        self.light_dir0 = model.light_dir.copy()
        self.head_ambient0 = np.array(model.vis.headlight.ambient)
        self.head_diffuse0 = np.array(model.vis.headlight.diffuse)
        self.mat_rgba0 = model.mat_rgba.copy()
        self.cam_pos0 = model.cam_pos.copy()
        self.cam_quat0 = model.cam_quat.copy()
        self.cam_fovy0 = model.cam_fovy.copy()
        self.scene = None

    def _free_qpos(self, body):
        jnt = self.m.body_jntadr[body]
        return self.m.jnt_qposadr[jnt], self.m.jnt_dofadr[jnt]

    def reset_objects(self, data, randomize_objects=True, want=None):
        """Nominal layout, or a random scene (sample_scene). `want` (from a JSON reset command) fixes strata
        and, with "seed", makes the reset reproducible. The desk and the cardboard box are static bodies,
        moved through body_pos; the wrapped box is a free body."""
        m = self.m
        qa, va = self._free_qpos(self.wrap)
        nq = 7 + sum(m.body(i).name.startswith(self.prefix + WRAP + "_panel") for i in range(m.nbody))
        data.qpos[qa:qa + nq] = m.qpos0[qa:qa + nq]
        data.qvel[va:va + nq - 1] = 0.0
        if not (randomize_objects or want):
            m.body_pos[self.desk], m.body_quat[self.desk] = self.desk_pos0, [1, 0, 0, 0]
            m.body_pos[self.box], m.body_quat[self.box] = self.box_pos0, [1, 0, 0, 0]
            self.scene = None
            return
        rng = np.random.default_rng(want["seed"]) if want and "seed" in want else self.rng
        scene = sample_scene(rng, want, m.ncam, m.nlight)
        (desk, dyaw), (card, cyaw), (wrap, wyaw) = place_objects(scene, self.wrap_half, self.desk_pos0,
                                                                  self.box_pos0, self.wrap_pos0)
        m.body_pos[self.desk], m.body_quat[self.desk] = desk, _quat_yaw(dyaw)
        m.body_pos[self.box], m.body_quat[self.box] = card, _quat_yaw(cyaw)
        flip = bool(scene["wrap"]["flip"])
        data.qpos[qa:qa + 3] = wrap + [0, 0, 2 * self.wrap_half[2] if flip else 0.0]  # body origin = "bottom" face
        data.qpos[qa + 3:qa + 7] = _quat_yaw(wyaw, flip)
        self.scene = scene
        if self.randomize or want:
            self.randomize_appearance(rng, scene)

    def object_poses(self, data):
        """World poses of the task objects, for rt/sim_state: body origins (wrapped box: the face that was
        its bottom when built; cardboard box: its floor), plus the wrapped box's centre and half sizes and
        the scene that was drawn at the last reset."""
        out = {name: {"pos": np.round(data.xpos[b], 4).tolist(), "quat": np.round(data.xquat[b], 4).tolist()}
               for name, b in ((WRAP, self.wrap), (CARDBOARD, self.box), (DESK, self.desk))}
        out[WRAP]["center"] = np.round(data.xipos[self.wrap], 4).tolist()
        out[WRAP]["half"] = np.round(self.wrap_half, 4).tolist()
        if self.scene:
            out["scene"] = self.scene
        return out

    def randomize_appearance(self, rng, scene):
        """Sim2real domain randomisation: light level and tint, material tints, camera pose and field of view."""
        m = self.m
        light = scene["light"]
        tint = np.asarray(light["tint"])
        m.light_diffuse[:] = self.light_diffuse0 * light["level"] * np.asarray(light["each"])[:, None] * tint
        m.vis.headlight.ambient[:] = self.head_ambient0 * light["level"] * tint
        m.vis.headlight.diffuse[:] = self.head_diffuse0 * light["level"] * tint
        d = self.light_dir0 + rng.normal(0, 0.15, self.light_dir0.shape)
        m.light_dir[:] = d / np.linalg.norm(d, axis=1, keepdims=True)
        mat_tint = rng.uniform(0.9, 1.1, (m.nmat, 3))
        m.mat_rgba[:, :3] = np.clip(self.mat_rgba0[:, :3] * mat_tint, 0, 1)
        m.cam_fovy[:] = self.cam_fovy0 * np.asarray(scene["fov"])
        m.cam_pos[:] = self.cam_pos0 + rng.normal(0, 0.005, self.cam_pos0.shape)
        for i in range(m.ncam):
            axis = rng.normal(size=3)
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.radians(rng.normal(0, 1.0)))
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, self.cam_quat0[i], dq)
            m.cam_quat[i] = q

    def success(self, data):
        """Wrapped box centre inside the cardboard box walls, resting low (it may lean on a wall), still."""
        R = data.xmat[self.box].reshape(3, 3)
        rel = R.T @ (data.xipos[self.wrap] - data.xpos[self.box])  # box origin = its floor
        inside = np.all(np.abs(rel[:2]) < BOX_INNER_HALF)
        low = rel[2] < self.wrap_half[2] + 0.085  # resting flat the centre is at its half height; leaning on the rim is still in
        _, va = self._free_qpos(self.wrap)
        still = np.linalg.norm(data.qvel[va:va + 3]) < 0.05
        return bool(inside and low and still)


def apply_photometric(spec):
    """Material colours, texture gains and light level fitted to the real recordings (calib/photometric.json)."""
    if not PHOTOMETRIC:
        return
    for name, rgba in PHOTOMETRIC.get("materials", {}).items():
        try:
            spec.material(name).rgba = rgba
        except Exception:
            pass
    # Textures that are too dark even at rgba = 1 get brightened themselves: builtin ones through their
    # colours, image files through a scaled copy in a cache folder.
    for name, gain in PHOTOMETRIC.get("texture_gain_rgb", {}).items():
        try:
            tex = spec.texture(spec.material(name).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB])
        except Exception:
            continue
        if tex is None:
            continue
        gain = np.asarray(gain, float)
        if tex.file:
            img = cv2.imread(tex.file, cv2.IMREAD_COLOR)
            out = os.path.join(tempfile.gettempdir(), "g1_lab_tex_cache", name.replace("/", "_") + ".png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cv2.imwrite(out, np.clip(img * gain[::-1], 0, 255).astype(np.uint8))
            tex.file = out
        else:
            for attr in ("rgb1", "rgb2", "markrgb"):
                setattr(tex, attr, np.clip(np.asarray(getattr(tex, attr)) * gain, 0, 1))
    k = PHOTOMETRIC.get("light_scale", 1.0)
    for light in spec.lights:
        light.diffuse = np.asarray(light.diffuse) * k
    hl = spec.visual.headlight
    hl.ambient, hl.diffuse = np.asarray(hl.ambient) * k, np.asarray(hl.diffuse) * k


_NOISE_BANK = {}


def _noise(shape, rng):
    """Unit-variance sensor noise from a small pre-generated bank (drawing 0.9 M normals per frame and
    camera is too slow for 3 cameras at 30 Hz); a random bank entry, randomly flipped."""
    bank = _NOISE_BANK.get(shape)
    if bank is None:
        bank = _NOISE_BANK[shape] = np.random.default_rng(0).standard_normal((8, *shape), dtype=np.float32)
    n = bank[rng.integers(len(bank))]
    return n[::-1] if rng.integers(2) else n


def camera_effects(rgb, cam_name, rng=np.random.default_rng()):
    """Per-camera gain (exposure / white balance), lens softness and sensor noise, like the real streams."""
    if cam_name not in CAMERA_BLUR:
        return rgb
    out = rgb.astype(np.float32)
    gain = PHOTOMETRIC.get("camera_gain_rgb", {}).get(cam_name)
    if gain:
        out *= np.asarray(gain, np.float32)
    sigma = CAMERA_BLUR.get(cam_name, 0.0) * rgb.shape[1] / 640.0
    if sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    noise = CAMERA_NOISE.get(cam_name, 0.0)
    if noise:
        out += noise * _noise(out.shape, rng)
    return np.clip(out, 0, 255).astype(np.uint8)
