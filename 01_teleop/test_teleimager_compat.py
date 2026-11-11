"""teleimager_compat: the 1.x client reads 1.x and 2.x camera configs.  python -m pytest 01_teleop/test_teleimager_compat.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1sim"))
from teleimager_compat import cameras_of  # noqa: E402

CAMERAS = {"head_camera": {"enable_zmq": True}, "left_wrist_camera": {"enable_zmq": False}}


def test_a_1x_config_is_returned_as_is():
    assert cameras_of(CAMERAS) is CAMERAS


def test_a_2x_config_gives_its_camera_part():
    assert cameras_of({"webrtc": {"bitrate": {}}, "camera": CAMERAS}) is CAMERAS


def test_no_answer_stays_no_answer():
    assert cameras_of(None) is None
