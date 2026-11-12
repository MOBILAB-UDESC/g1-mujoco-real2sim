"""Code-as-policy demo generator: put the bubble-wrapped box into the cardboard box (lab sim).

Closed loop on the sim's privileged object poses (rt/sim_state). Every motion is planned on a kinematic copy
of the simulator's model (sim_model.SimModel) before it is sent, in this order:

    turn the waist to face the box -> hands to both sides of it, well clear of it
    -> lower them beside it -> close in sideways and squeeze -> lift over the cardboard wall
    -> turn the waist toward the cardboard box -> reach over it -> release -> hands back, arms folded
    -> turn back to the rest pose.

What the plan guarantees (learned from the first real-robot test of a policy trained on the older script):
  * the hands never pass over the box: they stay below its top and outside its footprint (several cm to
    each side) until they close in sideways;
  * the upper arms stay clear of the torso (elbows out): at least TRUNK_CLEAR everywhere;
  * no arm or hand touches the cardboard box: at least CARD_CLEAR from it in every planned pose and move;
  * the torso faces the box before the grasp, so neither arm has to reach across the body;
  * no joint is commanded faster than V_MAX (the real robot's runner limits commands to 1.0 rad/s);
  * widths, heights and the hand yaw follow the box's real size and orientation (upright or upside down,
    any of its sides toward the robot).
The arms cannot reach far enough to set the box down flat inside the cardboard box (it is 49 x 33 cm and
starts 17 cm to the robot's left): the box is released over the near corner, as deep as the arms reach, and
tips in. That is also what the real teleoperated episodes do.

Recorded in the 15-joint "ArmsWaist" format (14 arm joints + kWaistYaw). An episode is kept only if the sim
reports success AND the measured motion respected the guarantees above (see Episode.verdict).

    scripts/run_sim.sh --scene lab --randomize --no-viewer --wrap-scale 1.08 1.06 1.07
    python -m g1lab.scripted.box_into_box --episodes 20 --root datasets/sim_scripted_v2 --repo-id local/sim_scripted_v2

--scenes FILE.json records a fixed list of scenes instead of random ones (04_recording/record_scripted_parts.sh
uses it to keep the dataset balanced): each entry is a scene request for the simulator (lab_scene.sample_scene).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from g1lab.robot_io import G1Arms, rot_z  # noqa: E402
from g1lab.scripted.sim_model import PELVIS_Z, SimModel  # noqa: E402

TASK = "put the wrapped box inside the cardboard box"
RATE = 30.0
PALM_FORWARD = 0.06       # palm centre ahead of the IK wrist frame (L_ee / R_ee) along its x axis
PALM_INSET = 0.02         # palm surface ~2 cm from the wrist frame toward the palm
HAND_HULL = 0.031         # the glove (fingers, thumb) reaches this much farther in than the palm surface: its
                          # collision hull spans 5.1 cm inward and 2.9 cm outward of the wrist axis (8 cm thick)
CORE = 0.02               # the wrapped box's rigid core is this much smaller than the box on every side
HAND_BELOW = 0.05         # ... and 5 cm below it (7 cm above), 13 cm ahead of the wrist frame
WALL_H = 0.18             # cardboard box height
BOX_INNER_HALF = np.array([0.161, 0.241])
WRAP_COL_HALF = np.array([0.113, 0.113, 0.133])  # wrapped box at --wrap-scale 1 1 1 (lab_scene.WRAP_COL_HALF)

JOINTS = [f"{side} {j}" for side in ("left", "right") for j in
          ("shoulder pitch", "shoulder roll", "shoulder yaw", "elbow", "wrist roll", "wrist pitch", "wrist yaw")] + ["waist"]
V_MAX = 0.85              # rad/s, fastest commanded joint (the robot-side runners clip at 1.0)
TRUNK_CLEAR = 0.025       # m, arms to torso / waist / pelvis, planned
TRUNK_CLEAR_MEASURED = 0.015  # m, the same on the measured joints (tracking error allowed)
HAND_CLEAR = 0.012        # m, hands to the wrapped box while approaching
CARD_CLEAR = 0.012        # m, arms and hands to the cardboard box, planned, during the whole episode
CARD_CLEAR_MEASURED = 0.004  # m, the same on the measured joints: nothing may touch the cardboard box
MAX_LEAD = 0.25           # rad, command ahead of the measured joint
SIDE_CLEAR = 0.06         # m, glove to box face while the hands come in beside the box (free side)
HOME_WRIST = np.array([0.05, 0.32, 0.215])  # start pose: wrist frames (x, +-y, z) in the pelvis frame: hands
                                           # behind the desk edge (no box placement can overlap them) and
                                           # with their underside above the cardboard box's rim
HOME_SEED = np.array([1.19, 0.48, 0.48, -0.81, -0.83, -0.22, -0.24,     # joint angles near the start pose,
                      1.19, -0.48, -0.48, -0.81, 0.83, -0.22, 0.24])    # where its IK starts from
POSTURE = np.full(14, np.nan)
POSTURE[1], POSTURE[8] = 0.5, -0.5         # shoulder roll: elbows away from the body
POSTURE_GAIN = 0.3
LEFT_ELBOW = (0.5, 0.3, 0.15, 0.75, 1.0, 0.0)  # left shoulder roll postures to try, in this order


def pace(robot, t):
    """Wait for the next control tick: real time against the simulator, one physics tick offline."""
    if getattr(robot, "offline", False):
        robot.tick()
        return t
    t += 1.0 / RATE
    time.sleep(max(0.0, t - time.perf_counter()))
    return t


class PlanError(RuntimeError):
    """The planner found no motion that keeps the guarantees for this scene."""


def sample_params(rng):
    """Per-episode motion variation."""
    return dict(
        press=rng.uniform(0.035, 0.05),      # palms commanded this far inside each box face: the arm's own
                                             # springiness plus the box's give leave a firm 1-2 cm squeeze
        grasp_dz=rng.uniform(-0.01, 0.02),   # palm height relative to the box centre: low enough that the
                                             # carry stays near shoulder height, where the arms reach farthest
        slide_x=rng.uniform(-0.065, -0.045),  # palm centre along the box at the grasp (0 = box centre): the
                                             # rear half of its sides. The box then sticks out ahead of the
                                             # hands, which is what lets it get over the cardboard box - the
                                             # arms are short. The planner grips farther back for far boxes.
        side=rng.uniform(SIDE_CLEAR - 0.01, SIDE_CLEAR + 0.015),  # the left hand gets what the gap allows
        lift_extra=rng.uniform(0.02, 0.04),  # box bottom above the cardboard rim while carrying
        jitter=0.005,                        # m, noise on every hand target
        speed=rng.uniform(0.8, 1.0),         # fraction of V_MAX
    )


def wrist_targets(center, yaw, x, left, right, hand_yaw=None):
    """Wrist frames (pelvis frame) for palms at lateral distances `left` / `right` from `center` (palm
    surfaces, measured along the y axis of a frame turned by `yaw`), `x` ahead of it; hands turned by
    hand_yaw (default yaw): fingers forward, palms facing each other."""
    G = rot_z(yaw)[:3, :3]
    H = rot_z(yaw if hand_yaw is None else hand_yaw)
    out = []
    for side, dist in ((1, left), (-1, right)):
        T = H.copy()
        T[:3, 3] = np.asarray(center, float) + G @ [x, side * (dist + PALM_INSET), 0.0] + H[:3, :3] @ [-PALM_FORWARD, 0, 0]
        out.append(T)
    return out


def box_geometry(sim_state):
    """The wrapped box as the grasp needs it (pelvis frame): centre, the yaw of the side pair to grip
    (the sides closest to "left and right of the robot as it faces the box"), half sizes across / along / up."""
    w = sim_state["wrapped_box"]
    centre = np.array(w["center"]) - [0, 0, PELVIS_Z]
    qw, qx, qy, qz = w["quat"]
    psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))  # direction of the box's own x axis
    facing = np.arctan2(centre[1], centre[0])                               # where the torso will look
    k = int(np.round((facing - psi) / (np.pi / 2)))
    yaw = psi + k * np.pi / 2
    hx, hy, hz = w["half"]
    along, across = (hx, hy) if k % 2 == 0 else (hy, hx)
    return dict(centre=centre, yaw=float(yaw), facing=float(facing), across=across, along=along, up=hz,
                bottom=centre[2] - hz)


class Planner:
    """Turns hand targets into joint waypoints on the kinematic model, checking every one."""

    def __init__(self, model, rng, jitter=0.0):
        self.M, self.rng, self.jitter = model, rng, jitter
        self.seed = None

    def solve(self, name, targets, waist, check=("trunk",), seed=None):
        """IK for one waypoint; raises PlanError if it is out of reach or breaks a clearance."""
        L, R = [T.copy() for T in targets]
        if self.jitter:
            for T in (L, R):
                T[:3, 3] += self.rng.normal(0, self.jitter, 3)
        # The same hand pose can be reached with the elbow more or less far out. Elbows out keeps the arms off
        # the torso; next to the cardboard box the left forearm may have to come in straighter, or higher,
        # to stay off its walls. The first elbow posture that passes every check is used.
        failure = None
        for left_roll in LEFT_ELBOW:
            posture = POSTURE.copy()
            posture[1] = left_roll
            q, err = self.M.ik(L, R, waist=waist, seed=self.seed if seed is None else seed, posture=posture,
                               posture_gain=POSTURE_GAIN)
            if err > 0.003:
                failure = failure or PlanError(f"{name}: out of reach by {err * 100:.1f} cm")
                continue
            q15 = np.append(q, waist)
            try:
                self.check(name, q15, check)
            except PlanError as error:
                failure = error
                continue
            self.seed = q
            return q15
        raise failure

    def check(self, name, q15, what):
        c = self.M.clearance(q15)
        if "trunk" in what and c["arm_trunk"] < TRUNK_CLEAR:
            raise PlanError(f"{name}: arm {c['arm_trunk'] * 100:.1f} cm from the torso")
        if "wrap" in what and min(c["hand_wrap"]) < HAND_CLEAR:
            raise PlanError(f"{name}: hand {min(c['hand_wrap']) * 100:.1f} cm from the wrapped box")
        if "card" in what and c["arm_card"] < CARD_CLEAR:
            raise PlanError(f"{name}: arm or hand {c['arm_card'] * 100:.1f} cm from the cardboard box")
        if "desk" in what and c["arm_desk"] < 0.01:
            raise PlanError(f"{name}: arm {c['arm_desk'] * 100:.1f} cm from the desk")
        return c

    def check_path(self, name, q0, q1, what, samples=6):
        """The straight joint-space move between two waypoints, at a few points along it."""
        for s in np.linspace(0, 1, samples + 2)[1:-1]:
            self.check(f"{name} (on the way)", q0 + (q1 - q0) * s, what)


def plan_home(model, rng=None):
    """Start / rest pose: hands low and wide just in front of the hips, elbows out."""
    x, y, z = HOME_WRIST + (rng.normal(0, 0.012, 3) if rng is not None else 0.0)
    L, R = np.eye(4), np.eye(4)
    L[:3, 3], R[:3, 3] = [x, y, z], [x, -y, z]
    q, err = model.ik(L, R, waist=0.0, seed=HOME_SEED, posture=POSTURE, posture_gain=POSTURE_GAIN)
    if err > 0.003:
        raise PlanError("home pose out of reach")
    return np.append(q, 0.0)


def plan_drop(P, home, floor, yaw_card, across, along, slide, palm_above_bottom, lift_extra, press):
    """Where to release, and with the box turned how. The opening is only 6-7 cm wider than the box, so the
    box goes in flat only if it is nearly square to the cardboard box and far enough from both near walls
    - which is at the very end of the arms' reach. Among the reachable releases this takes the one where the
    box overlaps the walls least (0 = it fits), then the squarest one.
    Returns (box centre xy, waist yaw, palm z, joint waypoints "over" and "release", report)."""
    C = rot_z(yaw_card)[:3, :3]
    rim = floor[2] + WALL_H
    z = rim + lift_extra + palm_above_bottom
    cands = []
    for turn in (0.2, 0.3, 0.4, 0.5, 0.65):      # box (= hand) yaw relative to the cardboard box
        # Against the cardboard box the wrapped box counts as CORE cm smaller on every side (its wrap gives).
        need_u = (along - CORE) * np.cos(turn) + (across - CORE) * np.sin(turn) + 0.008   # centre to the near wall
        need_v = (along - CORE) * np.sin(turn) + (across - CORE) * np.cos(turn) + 0.008   # ... and to the right wall
        for u in np.arange(0.11, 0.20, 0.01):
            for v in np.arange(0.11, 0.22, 0.01):
                if u >= need_u and v >= need_v:      # only releases where the box fits between the walls:
                    cands.append((0, turn, -(u + v), u, v))   # a box that lands on the rim stays stuck there
    seed = home[:14]   # a neutral posture: the arms come from another waist angle
    out_of_reach = set()
    for overlap, turn, _, u, v in sorted(cands):
        if any(turn == t and u >= u0 and v >= v0 for t, u0, v0 in out_of_reach):
            continue       # a point nearer to the robot was already out of reach with the hands turned like this
        xy = floor[:2] + C[:2, :2] @ [-BOX_INNER_HALF[0] + u, -BOX_INNER_HALF[1] + v]
        yaw = yaw_card + turn
        for waist in (yaw + 0.1, yaw + 0.25, yaw - 0.05):
            waist = float(np.clip(waist, 0.0, 1.0))
            try:
                over = P.solve("over the cardboard box", wrist_targets([*xy, z], yaw, slide, across - press, across - press),
                               waist, seed=seed, check=("trunk", "card"))
                release = P.solve("release", wrist_targets([*xy, z], yaw, slide, across + 0.06, across + 0.06), waist,
                                  seed=over[:14], check=("trunk", "card"))
                return xy, waist, z, over, release, dict(inside=(round(float(u), 2), round(float(v), 2)), yaw=yaw,
                                                         turn_deg=round(float(np.degrees(turn))), overlap_cm=int(overlap))
            except PlanError:
                continue
        out_of_reach.add((turn, u, v))
    raise PlanError("the cardboard box is too far to set the box down inside it")


def plan_episode(model, sim_state, home, p, rng):
    """All joint waypoints of one episode: a list of (name, q15, recorded phase)."""
    g = box_geometry(sim_state)
    floor = np.array(sim_state["cardboard_box"]["pos"]) - [0, 0, PELVIS_Z]
    qw, _, _, qz = sim_state["cardboard_box"]["quat"]
    yaw_card = 2 * np.arctan2(qz, qw)
    model.set_scene(sim_state)
    c, across, along = g["centre"], g["across"], g["along"]
    z_grasp = c[2] + p["grasp_dz"]
    # The hands travel at a height where their underside clears the cardboard rim (the left hand passes over
    # the cardboard box's corner on its way into the gap between the boxes). That is still below the wrapped
    # box's top: the hands are beside the box all the time, never above it.
    z_transit = max(z_grasp, floor[2] + WALL_H + 0.03 + HAND_BELOW)

    last, drops = None, {}
    # The torso turns halfway between "facing the box" and "parallel to the box": both arms then work nearly
    # straight ahead. Alternatives are tried when a check fails; so are a narrower left side (the cardboard
    # box is next to it) and a grip farther back on the box (when the box is at the end of the arms' reach).
    tries = [(turn, left_clear, slide)
             for slide in (p["slide_x"], -0.07)
             for turn in (0.5, 1.0, 0.0)
             for left_clear in (p["side"], 0.045, 0.035, 0.025)]
    for turn, left_clear, slide in tries:
        if True:
            waist = float(np.clip(g["facing"] + turn * (g["yaw"] - g["facing"]), -1.0, 1.0))
            P = Planner(model, rng, p["jitter"])
            P.seed = home[:14]
            try:
                centre = [c[0], c[1], z_grasp]
                above = [c[0], c[1], z_transit]
                wide = (across + left_clear + HAND_HULL, across + p["side"] + HAND_HULL)
                # Turn the torso first, arms still: the hands swing round at their rest height, above the rim.
                face = np.append(home[:14], waist)
                P.check_path("face the box", home, face, ("trunk", "wrap", "card"))
                P.check("face the box", face, ("trunk", "wrap", "card"))
                steps = [("face the box", face, "approach")]
                # Out to both sides of the box in two legs: a straight move in joint space would let the
                # hands dip on the way.
                goal = wrist_targets(above, g["yaw"], slide, *wide)
                start = [rot_z(waist) @ T for T in model.fk(home)]
                mid = []
                for T0, T1 in zip(start, goal):
                    T = T1.copy()
                    T[:3, 3] = (T0[:3, 3] + T1[:3, 3]) / 2
                    T[2, 3] = max(T0[2, 3], T1[2, 3]) + 0.02
                    mid.append(T)
                q = P.solve("halfway to the box", mid, waist, check=("trunk", "wrap", "card"))
                P.check_path("halfway to the box", face, q, ("trunk", "wrap", "card"))
                steps.append(("halfway to the box", q, "approach"))
                q = P.solve("beside the box", goal, waist, check=("trunk", "wrap", "card"))
                P.check_path("to beside the box", steps[-1][1], q, ("trunk", "wrap", "card"))
                steps.append(("beside the box", q, "approach"))
                q = P.solve("lower beside the box", wrist_targets(centre, g["yaw"], slide, *wide), waist,
                            check=("trunk", "wrap", "card", "desk"))
                P.check_path("lower beside the box", steps[-1][1], q, ("trunk", "wrap", "card"))
                steps.append(("lower beside the box", q, "approach"))
                grip = (across - p["press"], across - p["press"])
                steps.append(("squeeze", P.solve("squeeze", wrist_targets(centre, g["yaw"], slide, *grip), waist,
                                                 check=("trunk", "card")), "grasp"))
                # From here the box travels with the hands: plan without it.
                model.hide_wrap()
                palm_above_bottom = z_grasp - g["bottom"]
                if slide not in drops:   # the release point only depends on where along the box it is held
                    try:
                        drops[slide] = plan_drop(P, home, floor, yaw_card, across, along, slide, palm_above_bottom,
                                                 p["lift_extra"], p["press"])
                    except PlanError as err:
                        drops[slide] = err
                if isinstance(drops[slide], PlanError):
                    raise drops[slide]
                xy, waist_drop, z_carry, over, release, drop = drops[slide]
                P.seed = steps[-1][1][:14]
                # Lift and pull the box toward the body: the arms reach less far at carry height. As little
                # as reach allows, and never so much that the box comes within 17 cm of the waist axis.
                lift = None
                room = float(np.hypot(c[0], c[1])) - along - 0.17
                for pull in (0.05, 0.08, 0.11, 0.14, 0.17):
                    if pull > max(room, 0.05):
                        break
                    try:
                        lift = P.solve("lift", wrist_targets([c[0], c[1], z_carry], g["yaw"], slide - pull, *grip),
                                       waist, check=("trunk", "card"), seed=steps[-1][1][:14])
                        break
                    except PlanError as err:
                        last = err
                if lift is None:
                    raise last
                P.check_path("lift", steps[-1][1], lift, ("trunk", "card"))
                steps.append(("lift", lift, "carry"))
                turned = np.append(lift[:14], waist_drop)
                P.check_path("turn to the cardboard box", lift, turned, ("trunk", "card"), samples=10)
                P.check("turn to the cardboard box", turned, ("trunk", "card"))
                steps.append(("turn to the cardboard box", turned, "carry"))
                P.check_path("over the cardboard box", turned, over, ("trunk", "card"), samples=10)
                steps.append(("over the cardboard box", over, "carry"))
                P.check_path("release", over, release, ("trunk", "card"))
                steps.append(("release", release, "release"))
                # Back: open hands straight back off the box, then the arms fold to their rest posture while
                # the torso still faces the cardboard box, and only then the torso turns back. Turning first
                # would sweep the left hand through the box that now stands in the cardboard box.
                last_q = release
                for back in (0.09, 0.05):
                    try:
                        P.seed = release[:14]
                        last_q = P.solve("hands back", wrist_targets([*xy, z_carry + 0.02], drop["yaw"], slide - back,
                                                                     across + 0.06, across + 0.06), waist_drop,
                                         check=("trunk", "card"))
                        P.check_path("hands back", release, last_q, ("trunk", "card"))
                        steps.append(("hands back", last_q, "return"))
                        break
                    except PlanError:
                        pass
                folded = np.append(home[:14], waist_drop)
                P.check_path("fold the arms", last_q, folded, ("trunk", "card"), samples=10)
                steps.append(("fold the arms", folded, "return"))
                P.check_path("turn back", folded, home, ("trunk", "card"))
                steps.append(("rest", home.copy(), "return"))
                info = dict(waist_grasp=round(waist, 3), waist_drop=round(waist_drop, 3), left_clear=round(float(left_clear), 3),
                            slide=round(float(slide), 3), drop_inside=drop["inside"], drop_turn_deg=drop["turn_deg"],
                            drop_overlap_cm=drop["overlap_cm"])
                return steps, info
            except PlanError as err:
                last = err
            finally:
                model.set_scene(sim_state)
    raise last


class Episode:
    """Sends the planned waypoints, records every control tick and watches the guarantees on what is measured."""

    def __init__(self, robot, writer, model, p):
        self.robot, self.writer, self.M, self.p = robot, writer, model, p
        self.frames = 0
        self.min_trunk, self.min_wrap, self.min_card = np.inf, np.inf, np.inf
        self.max_speed, self.max_lead = 0.0, 0.0
        self.phase, self.step, self.lead_at, self.load = "approach", "start", "", 0.0
        self._last_action = None

    def record(self):
        obs = self.robot.observe()
        action = self.robot.last_command()
        state = np.asarray(obs["state"], float)
        c = self.M.clearance(state)
        self.min_trunk = min(self.min_trunk, c["arm_trunk"])
        self.min_card = min(self.min_card, c["arm_card"])
        if self.phase == "approach":
            self.min_wrap = min(self.min_wrap, *c["hand_wrap"])
        lead = np.abs(action - state)
        if lead.max() > self.max_lead:
            self.max_lead, self.lead_at = float(lead.max()), f"{JOINTS[int(lead.argmax())]} while {self.step}"
        if self._last_action is not None:
            self.max_speed = max(self.max_speed, float(np.max(np.abs(action - self._last_action))) * RATE)
        self._last_action = np.array(action, float)
        if self.writer is not None and len(obs["images"]) == 3:
            self.writer.add_frame(obs["state"], action, obs["images"])
            self.frames += 1

    def move(self, q1, phase, min_time=0.5):
        """Cosine move to q1 (15), slow enough that no joint command exceeds speed * V_MAX; the gravity
        feed-forward follows the motion, so the arms carry the box without sagging."""
        self.phase = phase
        q0 = self.robot.last_command().astype(float)
        q1 = np.asarray(q1, float)
        duration = max(min_time, np.pi / 2 * float(np.max(np.abs(q1 - q0))) / (V_MAX * self.p["speed"]))
        # While the box is in the hands its weight is part of the feed-forward, half in each hand: without it
        # the outstretched arms sag a few cm and the box does not clear the cardboard wall.
        load = self.M.wrap_weight / 2 if phase == "carry" else 0.0
        tau0, tau1 = self.M.gravity(q0, self.load), self.M.gravity(q1, load)
        self.load = load
        n = max(1, int(np.ceil(duration * RATE)))
        t = time.perf_counter()
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)
            self.robot.command(q0 + (q1 - q0) * s, tau0 + (tau1 - tau0) * s)
            self.record()
            t = pace(self.robot, t)

    def hold(self, seconds):
        t = time.perf_counter()
        for _ in range(int(seconds * RATE)):
            self.record()
            t = pace(self.robot, t)

    def verdict(self, sim_success):
        """Why the episode is kept or dropped."""
        if not sim_success:
            return "box not in the cardboard box"
        if self.min_wrap < 0.003:
            return f"a hand touched the box before the squeeze ({self.min_wrap * 100:.1f} cm)"
        if self.min_card < CARD_CLEAR_MEASURED:
            return f"an arm or hand touched the cardboard box ({self.min_card * 100:.1f} cm)"
        if self.min_trunk < TRUNK_CLEAR_MEASURED:
            return f"arm {self.min_trunk * 100:.1f} cm from the torso"
        if self.max_speed > 1.0:
            return f"command speed {self.max_speed:.2f} rad/s"
        if self.max_lead > MAX_LEAD:
            return f"command {self.max_lead:.2f} rad ahead of the joint"
        return "success"


def wait_scene(robot, want=None, timeout=4.0):
    """The sim's state after a reset: with a requested scene, wait until the sim reports that scene."""
    t_end = time.time() + timeout
    st = None
    while time.time() < t_end:
        st = robot.sim_state()
        if st and "wrapped_box" in st and "cardboard_box" in st and "center" in st["wrapped_box"]:
            scene = st.get("scene") or {}
            if not want or all(scene.get("wrap", {}).get(k) == want[k] for k in ("flip", "face") if k in want):
                return st
        time.sleep(0.05)
    raise RuntimeError("no (matching) rt/sim_state from the sim - is it running with --scene lab on this domain?")


def go_to(robot, model, q15, speed=0.6):
    """Unrecorded, slow move (to the start pose)."""
    q0 = robot.last_command().astype(float)
    q1 = np.asarray(q15, float)
    n = max(1, int(np.ceil(max(0.8, np.pi / 2 * float(np.max(np.abs(q1 - q0))) / speed) * RATE)))
    tau0, tau1 = model.gravity(q0), model.gravity(q1)
    t = time.perf_counter()
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        robot.command(q0 + (q1 - q0) * s, tau0 + (tau1 - tau0) * s)
        t = pace(robot, t)


def run_episode(robot, writer, model, sim_state, home, rng, task=TASK):
    p = sample_params(rng)
    steps, info = plan_episode(model, sim_state, home, p, rng)
    ep = Episode(robot, writer, model, p)
    if writer:
        writer.start_episode(task)
    ep.hold(0.4)
    for name, q, phase in steps:
        ep.step = name
        ep.move(q, phase)
        if name in ("squeeze", "release"):
            ep.hold(0.3)
    ep.hold(0.6)
    info.update(min_trunk_cm=round(ep.min_trunk * 100, 1), min_wrap_cm=round(ep.min_wrap * 100, 1),
                min_card_cm=round(ep.min_card * 100, 1),
                max_speed=round(ep.max_speed, 2), max_lead=round(ep.max_lead, 2), lead_at=ep.lead_at)
    return ep, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=10, help="successful episodes to collect")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--root", default=None, help="LeRobot dataset root (omit to run without recording)")
    p.add_argument("--repo-id", default="local/sim_scripted_v2")
    p.add_argument("--task", default=TASK)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--domain", type=int, default=None)
    p.add_argument("--image-port", type=int, default=60000)
    p.add_argument("--scenes", default=None,
                   help="JSON file with a list of scene requests (lab_scene.sample_scene): one kept episode per "
                        "entry, each retried up to --tries times, then redrawn in the same stratum")
    p.add_argument("--tries", type=int, default=4, help="attempts on the same scene before it is redrawn")
    p.add_argument("--log", default=None, help="append one JSON line per attempt (scene, result, measurements)")
    p.add_argument("--tag", default="", help="copied into every --log line (the recorder passes the part name)")
    p.add_argument("--offline", action="store_true",
                   help="no simulator process: run on a private copy of it, faster than real time, without "
                        "images (for trying the policy and for feasibility sweeps; nothing can be recorded)")
    p.add_argument("--wrap-scale", type=float, nargs=3, default=[1.08, 1.06, 1.07], help="--offline: box size")
    a = p.parse_args()

    if a.offline:
        from g1lab.scripted.offline import OfflineRobot
        robot = OfflineRobot(a.wrap_scale, seed=a.seed)
        a.root = None
    else:
        robot = G1Arms("sim", domain=a.domain, image_port=a.image_port, waist=True)
    robot.reset_sim()
    st = wait_scene(robot)
    model = SimModel(np.array(st["wrapped_box"]["half"]) / WRAP_COL_HALF)
    writer = None
    if a.root:
        from g1lab.lerobot_writer import LeRobotWriter
        writer = LeRobotWriter(a.root, a.repo_id, a.task, waist=True)
    rng = np.random.default_rng(a.seed)
    scenes = json.load(open(a.scenes)) if a.scenes else None
    wanted = len(scenes) if scenes else a.episodes
    max_attempts = a.max_attempts or (16 if scenes else 3) * wanted
    log = open(a.log, "a") if a.log else None
    ok = attempts = tries = 0
    try:
        while ok < wanted and attempts < max_attempts:
            attempts += 1
            want = None
            if scenes:
                want = dict(scenes[ok])
                want["seed"] = int(want.get("seed", 0)) + 100000 * (tries // a.tries)  # same stratum, new scene
            home = plan_home(model, rng)
            go_to(robot, model, home)
            robot.reset_sim(json.dumps(want) if want else 1)
            if not a.offline:
                time.sleep(1.0)
            st = wait_scene(robot, want)
            verdict, info, frames = "", {}, 0
            try:
                ep, info = run_episode(robot, writer, model, st, home, rng, a.task)
                for _ in range(24):   # let the box come to rest
                    pace(robot, time.perf_counter())
                verdict, frames = ep.verdict(robot.success()), ep.frames
            except PlanError as err:
                verdict = f"no plan: {err}"
            success = verdict == "success"
            if writer and (frames or writer.num_frames):
                (writer.save_episode if success else writer.discard_episode)()
            ok += success
            if success:
                tries = 0
            elif verdict.startswith("no plan"):   # the same scene would have no plan again: draw another one
                tries = (tries // a.tries + 1) * a.tries
            else:
                tries += 1
            print(f"[scripted] attempt {attempts}: {verdict.upper() if success else verdict} ({frames} frames, {info}) "
                  f"-> {ok}/{wanted}", flush=True)
            if log:
                log.write(json.dumps(dict(tag=a.tag, attempt=attempts, kept=ok if success else None, verdict=verdict, want=want,
                                          scene=st.get("scene"), info=info)) + "\n")
                log.flush()
    finally:
        if writer:
            writer.close()
    print(f"[scripted] {ok} successful of {attempts} attempts ({100 * ok / max(1, attempts):.0f}%)", flush=True)
    return ok >= wanted


if __name__ == "__main__":
    done = main()
    sys.stdout.flush()
    os._exit(0 if done else 1)
