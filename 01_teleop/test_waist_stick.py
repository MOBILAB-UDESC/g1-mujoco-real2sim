"""WaistStick (01_teleop/teleop_hand_and_arm.py): right stick -> waist target + body turn rate.

    python -m pytest 01_teleop/test_waist_stick.py
The class is cut out of the script, so the test needs neither xr_teleoperate nor a robot.
"""
import ast
import os

import numpy as np


def load_waist_stick():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teleop_hand_and_arm.py")
    source = open(path).read()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "WaistStick")
    scope = {"np": np}
    exec(ast.get_source_segment(source, node), scope)
    return scope["WaistStick"]


WaistStick = load_waist_stick()
DT = 1 / 30


def test_stick_twists_the_waist_and_the_body_stands_still():
    w = WaistStick(q0=0.2)
    q, turn = w.step(0.6, walking=False, dt=DT)
    assert np.isclose(q, 0.2 + 0.6 * DT) and turn == 0.0
    assert w.step(0.0, walking=False, dt=DT) == (q, 0.0)       # stick released: the waist stays


def test_past_the_limit_the_body_turns_instead():
    w = WaistStick(q0=0.99, limit=1.0, max_turn=0.6)
    q, turn = w.step(0.6, walking=False, dt=DT)
    assert np.isclose(q, 1.0) and 0.0 < turn <= 0.6
    q, turn = w.step(0.6, walking=False, dt=DT)
    assert np.isclose(q, 1.0) and np.isclose(turn, 0.6)        # all of the stick goes into turning


def test_while_walking_the_body_turns_until_the_waist_is_straight():
    w = WaistStick(q0=0.8, follow_rate=1.5, max_turn=0.6)
    turned = 0.0
    for _ in range(30 * 8):
        q, turn = w.step(0.0, walking=True, dt=DT)
        assert abs(turn) <= 0.6
        turned += turn * DT
    assert abs(q) < 0.01 and np.isclose(turned + q, 0.8, atol=1e-6)   # the torso keeps facing the same way


def test_it_never_leaves_the_limit():
    w = WaistStick(q0=0.0, limit=0.5)
    for rate, walking in [(3.0, False), (3.0, True), (-3.0, False), (-3.0, True)] * 60:
        q, _ = w.step(rate, walking, DT)
        assert -0.5 - 1e-9 <= q <= 0.5 + 1e-9
