"""Watch recorded scripted episodes again in the simulator: rebuilds each episode's scene from scenes.jsonl
and sends the recorded joint targets at 30 Hz.

Start the simulator first, with a window and the box size of those episodes (all episodes of one part share
it; --print-sim prints the command):

    python -m g1lab.scripted.replay_dataset --root datasets/sim_scripted_v2 --episodes 2 3 4 --print-sim
    python -m g1lab.scripted.replay_dataset --root datasets/sim_scripted_v2 --episodes 2 3 4 --loop

It is a replay, not the recording: the dataset has the joint targets but not the feed-forward torques, which
are computed again here (the box's weight is added while the box is off the desk), so the motion can differ
by a few millimetres and an episode can, rarely, end differently.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from g1lab.robot_io import G1Arms
from g1lab.scripted.box_into_box import RATE, WRAP_COL_HALF, go_to, pace, wait_scene
from g1lab.scripted.sim_model import SimModel


def load_actions(root, episodes):
    import pandas as pd
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "action"])
                   for f in sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet"))))
    return {e: np.stack(df[df.episode_index == e]["action"].to_numpy()).astype(float) for e in episodes}


def sim_command(scale):
    return ("python 01_teleop/g1sim/g1_xr_sim.py --scene lab --randomize --camera-shadows none --no-camera "
            "--wrap-scale " + " ".join(f"{s:g}" for s in scale))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="dataset with scenes.jsonl (datasets/sim_scripted_v2)")
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--loop", action="store_true", help="repeat the episodes until Ctrl+C")
    p.add_argument("--print-sim", action="store_true", help="only print the simulator command for these episodes")
    p.add_argument("--domain", type=int, default=None)
    a = p.parse_args()

    rows = {r["episode_index"]: r for r in map(json.loads, open(os.path.join(a.root, "scenes.jsonl")))}
    scales = {tuple(rows[e]["box_scale"]) for e in a.episodes}
    if len(scales) > 1:
        raise SystemExit("these episodes have different box sizes (one simulator run has one size): "
                         + ", ".join(f"{e}: {rows[e]['part']}" for e in a.episodes))
    scale = scales.pop()
    if a.print_sim:
        print(sim_command(scale))
        return

    actions = load_actions(a.root, a.episodes)
    robot = G1Arms("sim", domain=a.domain, cameras=False, waist=True)
    robot.reset_sim()
    st = wait_scene(robot)
    if not np.allclose(np.array(st["wrapped_box"]["half"]) / WRAP_COL_HALF, scale, atol=2e-3):
        raise SystemExit("the simulator runs with another box size. Start it with:\n  " + sim_command(scale))
    model = SimModel(scale)

    while True:
        for e in a.episodes:
            act, row = actions[e], rows[e]
            go_to(robot, model, act[0])
            robot.reset_sim(json.dumps(row["strata"]))
            time.sleep(1.0)
            st = wait_scene(robot, row["strata"])
            same = np.allclose(st["scene"]["desk"], row["scene"]["desk"], atol=1e-6) and \
                abs(st["scene"]["wrap"]["far"] - row["scene"]["wrap"]["far"]) < 1e-6
            print(f"[replay] episode {e} ({row['part']}, {len(act) / RATE:.0f} s)"
                  + ("" if same else ": the sim drew another scene than the recorded one, skipped"), flush=True)
            if not same:
                continue
            z0, load = st["wrapped_box"]["center"][2], 0.0
            t = time.perf_counter()
            for q in act:
                lifted = robot.sim_state()["wrapped_box"]["center"][2] > z0 + 0.008
                load += 0.2 * ((model.wrap_weight / 2 if lifted else 0.0) - load)
                robot.command(q, model.gravity(q, load))
                t = pace(robot, t)
            for _ in range(30):   # let the box come to rest
                t = pace(robot, t)
            print(f"[replay] episode {e}: " + ("box in the cardboard box" if robot.success() else "box NOT in the cardboard box"),
                  flush=True)
        if not a.loop:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    sys.stdout.flush()
    os._exit(0)
