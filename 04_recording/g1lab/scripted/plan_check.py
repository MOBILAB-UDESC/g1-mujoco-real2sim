"""Plan scripted episodes for random scenes without the simulator: which scenes have a plan, and why not.

    python -m g1lab.scripted.plan_check --scenes 60 --wrap-scale 1.08 1.06 1.07

Draws scenes exactly like the simulator's --randomize (lab_scene.sample_scene / place_objects) and runs the
planner of box_into_box.py on each. Fast (no physics, no rendering): use it to see how much of the scene
range the planner covers before recording.
"""

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from g1lab.scripted import box_into_box as B  # noqa: E402
from g1lab.scripted.sim_model import SimModel  # noqa: E402


def fake_sim_state(task, scene):
    """What rt/sim_state would report right after a reset to `scene` (objects at rest where they were put).
    task: a lab_scene.LabTask made from the untouched model (it remembers the nominal layout)."""
    import lab_scene as L
    (desk, dyaw), (card, cyaw), (wrap, wyaw) = L.place_objects(scene, task.wrap_half, task.desk_pos0, task.box_pos0,
                                                               task.wrap_pos0)
    flip = bool(scene["wrap"]["flip"])
    up = task.wrap_half[2]
    pos = wrap + [0, 0, 2 * up if flip else 0.0]
    return {"desk": {"pos": desk.tolist(), "quat": L._quat_yaw(dyaw)},
            "cardboard_box": {"pos": card.tolist(), "quat": L._quat_yaw(cyaw)},
            "wrapped_box": {"pos": pos.tolist(), "quat": L._quat_yaw(wyaw, flip),
                            "center": (wrap + [0, 0, up]).tolist(), "half": task.wrap_half.tolist()},
            "scene": scene}


def main():
    import lab_scene as L
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", type=int, default=40)
    p.add_argument("--wrap-scale", type=float, nargs=3, default=[1.08, 1.06, 1.07])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    model = SimModel(a.wrap_scale)
    task = L.LabTask(model.m, "env/")
    rng = np.random.default_rng(a.seed)
    home = B.plan_home(model)
    reasons = collections.Counter()
    ok = 0
    for i in range(a.scenes):
        scene = L.sample_scene(rng)
        st = fake_sim_state(task, scene)
        try:
            steps, info = B.plan_episode(model, st, home, B.sample_params(rng), rng)
            ok += 1
            msg = f"ok   {info}"
        except B.PlanError as err:
            reasons[str(err).split(":")[0]] += 1
            msg = f"FAIL {err}"
        if a.verbose:
            w = scene["wrap"]
            print(f"{i:3d} far {w['far']:.2f} gap {w['gap']:.3f} yaw {w['yaw']:+5.1f} face {w['face']} flip {w['flip']} "
                  f"desk {scene['desk'][0]:+.3f} {scene['desk'][1]:+.3f} | {msg}", flush=True)
    print(f"plans found for {ok} of {a.scenes} scenes; failures by step: {dict(reasons)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
