"""The lab simulator without DDS, cameras or real time, behind the same interface as G1Arms.

    robot = OfflineRobot(wrap_scale=(1.08, 1.06, 1.07), seed=3)

It owns a private g1_xr_sim.G1XrSim (same model, same PD law, same scene randomisation and success check) and
advances its physics by one control tick (1/30 s) per tick() call, as fast as the PC computes it. For trying
scripted policies and for feasibility sweeps: an episode takes a few seconds instead of half a minute, and
nothing is rendered. Recording still needs the real simulator (the images).
"""

import json
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (os.path.join(REPO, "01_teleop", "g1sim"), os.path.join(REPO, "02_wrist_cameras"),
          os.path.join(REPO, "03_real2sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

RATE = 30.0
WAIST = 12


class OfflineRobot:
    sim = True
    offline = True
    waist = True

    def __init__(self, wrap_scale=(1.0, 1.0, 1.0), seed=None, randomize=True, rigid_box=False):
        import g1_xr_sim as G
        argv = ["--scene", "lab", "--no-viewer", "--no-camera", "--wrap-scale", *[str(float(s)) for s in wrap_scale]]
        if randomize:
            argv += ["--randomize"] + (["--seed", str(seed)] if seed is not None else [])
        if rigid_box:
            argv.append("--rigid-box")
        self.G = G
        self.s = G.G1XrSim(G.parse_args(argv))
        self._q = np.array(self.s.arm_home, float)
        self._tau = np.zeros(14)
        self._waist = 0.0
        self._time = 0.0
        self.tick()

    # ------------------------------------------------------------------ time
    def tick(self):
        """One control period of physics with the current command held."""
        s, G = self.s, self.G
        s.cmd[0][:] = 0.0
        s.cmd[0][WAIST] = self._waist
        s.cmd[0][G.ARMS] = self._q
        s.cmd[1][:] = 0.0
        s.cmd[2], s.cmd[3] = G.HOLD_KP, G.HOLD_KD
        s.cmd[4][:] = 0.0
        s.cmd[4][G.ARMS] = self._tau
        self._time += 1.0 / RATE
        while s.data.time < self._time:
            s.last_cmd_time = time.monotonic()
            s.loco_step()
            s.apply_pd()
            mujoco.mj_step(s.model, s.data)

    # ----------------------------------------------------------------- G1Arms
    def state(self):
        q = self.s.data.sensordata[:self.G.NUM_BODY_MOTORS]
        return np.append(q[self.G.ARMS], q[WAIST]).astype(np.float32)

    def last_command(self):
        return np.append(self._q, self._waist).astype(np.float32)

    def observe(self):
        return {"state": self.state(), "images": {}}

    def command(self, q, tau=None):
        q = np.asarray(q, float).reshape(-1)
        if q.size == 15:
            self._waist = float(np.clip(q[14], -1.0, 1.0))
        self._q = q[:14].copy()
        self._tau = np.zeros(14) if tau is None else np.asarray(tau, float).reshape(14)

    def sim_state(self):
        s = self.s
        mujoco.mj_kinematics(s.model, s.data)
        state = dict(s.task.object_poses(s.data))
        state["success"] = s.task.success(s.data)
        return state

    def success(self):
        return bool(self.sim_state()["success"])

    def reset_sim(self, category=1):
        want = None
        if isinstance(category, str) and category.lstrip().startswith("{"):
            want = json.loads(category)
        self.s.reset(robot=False, want=want)
        mujoco.mj_forward(self.s.model, self.s.data)
        for _ in range(20):
            self.tick()
