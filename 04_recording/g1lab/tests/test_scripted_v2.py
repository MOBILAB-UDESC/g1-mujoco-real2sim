"""Scripted policy v2: balanced schedule, box geometry, and a full plan on the kinematic model.

    source scripts/env.sh g1lab && python -m pytest 04_recording/g1lab/tests/test_scripted_v2.py
The plan test builds the simulator's model (needs third_party/ from install/install.sh); skipped without it.
"""
import collections
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from g1lab.scripted import schedule as S  # noqa: E402


def test_schedule_is_balanced_and_split_in_parts():
    sched = S.make_schedule(episodes=200, part=5, seed=1)
    scenes = [w for p in sched["parts"] for w in p["scenes"]]
    assert len(sched["parts"]) == 40 and len(scenes) == 200
    assert len({w["seed"] for w in scenes}) == 200
    for name, values in S.FACTORS.items():
        counts = collections.Counter(S.strata(w)[name] for w in scenes)
        assert set(counts) == set(values)
        assert max(counts.values()) - min(counts.values()) <= 1
    for p in sched["parts"]:
        assert all(S.SCALE[0] * (1 - S.AXIS) - 1e-3 <= s <= S.SCALE[1] * (1 + S.AXIS) + 1e-3 for s in p["scale"])
    assert S.make_schedule(200, 5, 1) == sched


def test_coverage_counts_attempts_and_kept_episodes():
    sched = S.make_schedule(episodes=4, part=2, seed=3)
    want = sched["parts"][0]["scenes"][0]
    rows = S.coverage(sched, [dict(want=want, verdict="success"), dict(want=want, verdict="box not in the cardboard box")])
    row = next(r for r in rows if r[0] == "flip" and r[1] == S.LABELS["flip"][want["flip"]])
    assert row[3:5] == (1, 2)


@pytest.fixture(scope="module")
def planning():
    try:
        from g1lab.scripted import box_into_box as B
        from g1lab.scripted.plan_check import fake_sim_state
        from g1lab.scripted.sim_model import SimModel
        import lab_scene as L
        model = SimModel((1.08, 1.06, 1.07))
    except Exception as err:  # no third_party/, no mujoco, ...
        pytest.skip(f"simulator model not available: {err}")
    return B, L, model, L.LabTask(model.m, "env/"), fake_sim_state


def test_box_geometry_picks_the_sides_left_and_right_of_the_robot(planning):
    B, L, model, task, fake = planning
    for face in range(4):
        scene = L.sample_scene(np.random.default_rng(face), dict(face=face, flip=face % 2))
        g = B.box_geometry(fake(task, scene))
        assert abs(g["yaw"] - g["facing"]) <= np.radians(L.SCENE["wrap_yaw"]) + 1e-6
        hx, hy = task.wrap_half[:2]
        assert (g["across"], g["along"]) == ((hy, hx) if face % 2 == 0 else (hx, hy))


def test_a_plan_keeps_every_guarantee(planning):
    B, L, model, task, fake = planning
    rng = np.random.default_rng(0)
    home = B.plan_home(model)
    plans = 0
    for seed in range(12):
        scene = L.sample_scene(np.random.default_rng(100 + seed), dict(desk=[1, 1], cell=[1, 1]))
        st = fake(task, scene)
        try:
            steps, info = B.plan_episode(model, st, home, B.sample_params(rng), rng)
        except B.PlanError:
            continue
        plans += 1
        model.set_scene(st)
        previous = home
        for name, q, phase in steps:
            if phase != "approach":
                model.hide_wrap()
            c = model.clearance(q)
            assert c["arm_trunk"] >= B.TRUNK_CLEAR - 1e-6, name
            assert c["arm_card"] >= B.CARD_CLEAR - 1e-6, name
            if phase == "approach":
                assert min(c["hand_wrap"]) >= B.HAND_CLEAR - 1e-6, name   # beside the box, never on it
            assert np.all(q[:14] >= model.limits[:, 0] - 1e-6) and np.all(q[:14] <= model.limits[:, 1] + 1e-6)
            previous = q
        assert np.allclose(steps[-1][1], home)
        assert info["drop_overlap_cm"] == 0
    assert plans >= 6   # most scenes in the middle of the range have a plan
