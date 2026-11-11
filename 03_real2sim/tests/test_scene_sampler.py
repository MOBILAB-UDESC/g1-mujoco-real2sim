"""lab_scene: randomised scenes stay inside what the robot can do, strata are honoured, the box model builds.

    source scripts/env.sh g1lab && python -m pytest 03_real2sim/tests
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "02_wrist_cameras"))
import lab_scene as L  # noqa: E402

HALF = np.array([0.122, 0.120, 0.142])
DESK0, CARD0, WRAP0 = np.array([0.445, 0.165, 0.0]), np.array([0.44, 0.415, 0.7405]), np.array([0.40, -0.05, 0.7405])


def placed(scene):
    return L.place_objects(scene, HALF, DESK0, CARD0, WRAP0)


def test_scenes_stay_within_reach_and_clear_of_the_cardboard_box():
    rng = np.random.default_rng(0)
    for _ in range(400):
        scene = L.sample_scene(rng)
        (desk, dyaw), (card, cyaw), (wrap, wyaw) = placed(scene)
        assert np.hypot(wrap[0], wrap[1]) <= L.WRAP_REACH + 1e-6
        ext_x = L._yaw_extent((HALF[:2] + L.WRAP_LUMPS)[::-1], wyaw - dyaw)
        assert wrap[0] - ext_x >= L.WRAP_NEAR - 0.012          # desk yaw moves it by a few mm
        # free space between the two boxes, measured across the desk
        wall = card[1] - L._yaw_extent(L.BOX_OUTER_HALF, cyaw - dyaw)
        gap = wall - wrap[1] - L._yaw_extent(HALF[:2] + L.WRAP_LUMPS, wyaw - dyaw)
        assert gap >= L.SCENE["wrap_gap"][0] - 0.03            # the reach limit may pull the box in a little
        assert L.SCENE["desk_x"][0] <= scene["desk"][0] <= L.SCENE["desk_x"][1]
        assert L.SCENE["desk_y"][0] <= scene["desk"][1] <= L.SCENE["desk_y"][1]
        assert all(1 - L.SCENE["fov"] <= f <= 1 + L.SCENE["fov"] for f in scene["fov"])


def test_one_side_of_the_box_faces_the_robot():
    rng = np.random.default_rng(1)
    for _ in range(200):
        scene = L.sample_scene(rng)
        _, _, (wrap, wyaw) = placed(scene)
        off = np.degrees(wyaw - np.arctan2(wrap[1], wrap[0]) - np.radians(90 * scene["wrap"]["face"]))
        assert abs(off) <= L.SCENE["wrap_yaw"] + 0.01


def test_requested_strata_are_kept_and_a_seed_repeats_the_scene():
    want = dict(flip=1, face=2, cell=[0, 2], desk=[1, 0], light=0)
    a = L.sample_scene(np.random.default_rng(5), want)
    b = L.sample_scene(np.random.default_rng(5), want)
    assert a == b
    assert a["wrap"]["flip"] == 1 and a["wrap"]["face"] == 2 and a["light"]["band"] == 0
    assert a["wrap"]["far"] <= 1 / 3
    lo, hi = L.SCENE["wrap_gap"]
    assert a["wrap"]["gap"] >= lo + 2 * (hi - lo) / 3
    assert a["desk"][0] >= np.mean(L.SCENE["desk_x"]) and a["desk"][1] <= np.mean(L.SCENE["desk_y"])
    assert L.SCENE["light_bands"][0][0] <= a["light"]["level"] <= L.SCENE["light_bands"][0][1]


def test_wrapped_box_model_scales_and_has_four_spring_panels():
    for scale, soft in (((1, 1, 1), False), ((1.1, 1.05, 1.08), True)):
        env = mujoco.MjSpec.from_file(L.LAB_XML)
        L._abs_assets(env, L.HERE)
        half = L.build_wrapped_box(env, scale, soft)
        m = env.compile()
        assert np.allclose(half, L.WRAP_COL_HALF * scale)
        panels = [m.body(i).name for i in range(m.nbody) if "panel" in m.body(i).name]
        assert len(panels) == (4 if soft else 0)
        assert np.isclose(m.body_subtreemass[m.body("wrapped_box").id], L.WRAP_MASS * np.prod(scale))
        d = mujoco.MjData(m)
        for _ in range(500):
            mujoco.mj_step(m, d)
        assert abs(d.xpos[m.body("wrapped_box").id][2] - 0.7405) < 0.003   # rests on the desk
