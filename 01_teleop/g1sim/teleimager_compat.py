"""Let the teleimager 1.x client of the pinned xr_teleoperate read the camera config of a teleimager 2.x server.

teleimager 2.x nests the cameras under "camera" (next to e.g. "webrtc"); 1.x has head_camera /
left_wrist_camera / right_wrist_camera at the top level, and the 1.x client fails on a 2.x config with
KeyError: 'head_camera'. install() makes the client's config request return the per-camera part in both cases.
Call it before an ImageClient is created (robot_io, record_live, the teleop launcher and script do).
"""


def cameras_of(cam_config):
    """The per-camera part of a teleimager config (1.x: the config itself; 2.x: its "camera" entry)."""
    if isinstance(cam_config, dict) and isinstance(cam_config.get("camera"), dict):
        return cam_config["camera"]
    return cam_config


def install():
    import teleimager.image_client as image_client

    original = image_client.ZMQ_Requester.request
    if getattr(original, "_unwraps_cameras", False):
        return

    def request(self, *args, **kwargs):
        return cameras_of(original(self, *args, **kwargs))

    request._unwraps_cameras = True
    image_client.ZMQ_Requester.request = request
