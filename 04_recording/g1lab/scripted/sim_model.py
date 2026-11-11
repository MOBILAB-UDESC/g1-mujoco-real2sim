"""A kinematic copy of the lab simulator's model, for planning scripted motions.

Built by the simulator's own code (g1_xr_sim.build_model), so robot, gloves, wrist cameras, desk and both boxes
are exactly what the simulator has. Nothing is simulated here: joint angles and object poses are written in and
geometry is read out.

    model = SimModel(wrap_scale=(1.08, 1.06, 1.07))
    model.set_scene(robot.sim_state())            # desk, cardboard box and wrapped box where the sim has them
    q, err = model.ik(left_T, right_T, waist=0.2, seed=q14)   # wrist targets like G1Arms.ik (pelvis frame)
    model.clearance(np.append(q, 0.2))            # {"arm_trunk": m, "hand_wrap": m, ...}

Why not only xr_teleoperate's IK: it has no say on where the elbows go (the arm has one joint more than a wrist
pose needs), and on the real robot the upper arms ended up against the torso. This IK reaches the same wrist
frames (L_ee / R_ee: 5 cm along x from the wrist yaw joint) and uses the spare freedom to keep a posture.
"""

import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (os.path.join(REPO, "01_teleop", "g1sim"), os.path.join(REPO, "02_wrist_cameras"),
          os.path.join(REPO, "03_real2sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

PELVIS_Z = 0.793   # world -> pelvis frame: z_pelvis = z_world - PELVIS_Z (the pelvis is fixed at x = y = 0)
EE_OFFSET = np.array([0.05, 0.0, 0.0])  # L_ee / R_ee in the wrist yaw link, as in xr_teleoperate's IK
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
ARM_BODIES = ["shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link"]
TRUNK_BODIES = ["torso_link", "waist_yaw_link", "waist_roll_link", "pelvis"]
ENV = "env/"


def _pose(pos, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pos
    return T


class SimModel:
    def __init__(self, wrap_scale=(1.0, 1.0, 1.0), rigid_box=False):
        import g1_xr_sim
        argv = ["--scene", "lab", "--no-camera", "--no-viewer", "--wrap-scale", *[str(float(s)) for s in wrap_scale]]
        if rigid_box:
            argv.append("--rigid-box")
        self.m, _ = g1_xr_sim.build_model(g1_xr_sim.parse_args(argv))
        self.d = mujoco.MjData(self.m)
        m = self.m
        names = [f"{side}_{j}_joint" for side in ("left", "right") for j in ARM_JOINTS]
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        self.limits = np.array([m.jnt_range[m.joint(n).id] for n in names])
        self.waist_adr = m.jnt_qposadr[m.joint("waist_yaw_joint").id]
        self.wrists = [m.body(f"{side}_wrist_yaw_link").id for side in ("left", "right")]

        def geoms(bodies, colliding=True):
            ids = {m.body(b).id for b in bodies}
            return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids
                    and (not colliding or m.geom_contype[g] or m.geom_conaffinity[g])]

        self.arm_geoms = {side: geoms([f"{side}_{b}" for b in ARM_BODIES]) for side in ("left", "right")}
        self.hand_geoms = {side: geoms([f"{side}_wrist_yaw_link"]) for side in ("left", "right")}
        self.trunk_geoms = geoms(TRUNK_BODIES)
        self.wrap_body = m.body(ENV + "wrapped_box").id
        self.card_body = m.body(ENV + "cardboard_box").id
        self.desk_body = m.body(ENV + "desk").id
        wrap_bodies = [m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith(ENV + "wrapped_box")]
        self.wrap_geoms = geoms(wrap_bodies)
        self.card_geoms = geoms([ENV + "cardboard_box"])
        self.desk_geoms = geoms([ENV + "desk"])
        jnt = m.body_jntadr[self.wrap_body]
        self.wrap_qadr = m.jnt_qposadr[jnt]
        mujoco.mj_forward(m, self.d)

    # ------------------------------------------------------------------ state in
    def set_scene(self, sim_state):
        """Desk, cardboard box and wrapped box at the poses the simulator reports (rt/sim_state)."""
        m, d = self.m, self.d
        for key, body in (("desk", self.desk_body), ("cardboard_box", self.card_body)):
            m.body_pos[body], m.body_quat[body] = sim_state[key]["pos"], sim_state[key]["quat"]
        a = self.wrap_qadr
        d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = sim_state["wrapped_box"]["pos"], sim_state["wrapped_box"]["quat"]
        mujoco.mj_kinematics(m, d)

    def hide_wrap(self):
        """Take the wrapped box out of the way (when checking motions that carry it)."""
        self.d.qpos[self.wrap_qadr:self.wrap_qadr + 3] = [5.0, 5.0, 5.0]

    def set_q(self, q):
        """Arm joints (14) and, if given, the waist yaw (15th value)."""
        q = np.asarray(q, float)
        self.d.qpos[self.qadr] = q[:14]
        if q.size > 14:
            self.d.qpos[self.waist_adr] = q[14]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    # ---------------------------------------------------------------- kinematics
    def _ee(self, i):
        b = self.wrists[i]
        R = self.d.xmat[b].reshape(3, 3)
        return self.d.xpos[b] + R @ EE_OFFSET, R

    def fk(self, q):
        """Wrist frames (L_ee, R_ee) in the pelvis frame, as 4x4 arrays."""
        self.set_q(q)
        out = []
        for i in (0, 1):
            p, R = self._ee(i)
            out.append(_pose(p - [0, 0, PELVIS_Z], R))
        return out

    def ik(self, left_T, right_T, waist=0.0, seed=None, posture=None, posture_gain=0.0, iters=80):
        """Arm joints that put both wrist frames on the targets (pelvis frame) with the waist at `waist`.
        The spare freedom of each arm is pulled toward `posture` (14 joint values; NaN = don't care).
        Returns (q14, largest position error in m)."""
        m, d = self.m, self.d
        q = np.array(d.qpos[self.qadr] if seed is None else seed, float)[:14]
        d.qpos[self.waist_adr] = waist
        targets = [np.asarray(left_T, float), np.asarray(right_T, float)]
        if posture is None:
            posture, posture_gain = np.full(14, np.nan), 0.0
        pull = np.where(np.isnan(posture), 0.0, 1.0)
        want = np.nan_to_num(posture)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err, history = np.inf, []
        for it in range(iters):
            self.set_q(q)
            err, rot_err, step = 0.0, 0.0, 0.0
            for i, T in enumerate(targets):
                p, R = self._ee(i)
                e_pos = T[:3, 3] + [0, 0, PELVIS_Z] - p
                Re = T[:3, :3] @ R.T
                e_rot = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                mujoco.mj_jac(m, d, jacp, jacr, p, self.wrists[i])
                dofs = self.dadr[7 * i:7 * i + 7]
                J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
                e = np.concatenate([e_pos, e_rot])
                Jp = J.T @ np.linalg.inv(J @ J.T + 4e-4 * np.eye(6))   # damped: steady near a stretched arm
                sl = slice(7 * i, 7 * i + 7)
                dq = Jp @ e + (np.eye(7) - Jp @ J) @ (posture_gain * pull[sl] * (want[sl] - q[sl]))
                dq = np.clip(0.8 * dq, -0.15, 0.15)
                q[sl] += dq
                err, rot_err = max(err, float(np.linalg.norm(e_pos))), max(rot_err, float(np.linalg.norm(e_rot)))
                step = max(step, float(np.max(np.abs(dq))))
            q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
            history.append(err)
            if it >= 3 and err < 3e-4 and rot_err < 2e-3 and step < 2e-3:
                break  # there, and the posture has settled
            if it >= 30 and err > 0.01 and err > 0.97 * history[-10]:
                break  # no longer getting closer: out of reach
        self.set_q(q)
        return q, err

    def gravity(self, q, load=0.0):
        """Joint torques that hold the arms against gravity at q (the feed-forward xr_teleoperate also sends),
        plus a weight of `load` newtons in each hand (half of what the hands carry)."""
        self.set_q(q)
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        tau = self.d.qfrc_bias[self.dadr].copy()
        if load:
            jacp = np.zeros((3, self.m.nv))
            for i in (0, 1):
                p, R = self._ee(i)
                mujoco.mj_jac(self.m, self.d, jacp, None, p + R @ [0.06, 0.0, 0.0], self.wrists[i])  # palm centre
                tau[7 * i:7 * i + 7] += jacp[2, self.dadr[7 * i:7 * i + 7]] * load
        return tau

    @property
    def wrap_weight(self):
        return float(self.m.body_subtreemass[self.wrap_body] * 9.81)

    # ----------------------------------------------------------------- distances
    def _min_dist(self, a, b, far=0.3):
        best = far
        for g1 in a:
            for g2 in b:
                best = min(best, mujoco.mj_geomDistance(self.m, self.d, g1, g2, best, None))
        return float(best)

    def clearance(self, q):
        """Smallest distances [m] at joint angles q (negative = overlapping):
        arm_trunk  upper arms, forearms and hands to torso / waist / pelvis
        hand_wrap  each hand (glove, wrist camera) to the wrapped box: (left, right)
        hand_card  hands to the cardboard box
        arm_desk   arms and hands to the desk top"""
        self.set_q(q)
        arms = self.arm_geoms["left"] + self.arm_geoms["right"]
        hands = self.hand_geoms["left"] + self.hand_geoms["right"]
        return dict(
            arm_trunk=self._min_dist(arms, self.trunk_geoms),
            hand_wrap=(self._min_dist(self.hand_geoms["left"], self.wrap_geoms),
                       self._min_dist(self.hand_geoms["right"], self.wrap_geoms)),
            hand_card=self._min_dist(hands, self.card_geoms),
            arm_desk=self._min_dist(arms, self.desk_geoms),
        )
