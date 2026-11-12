"""A balanced plan for a scripted recording, and the coverage it actually reached.

    python -m g1lab.scripted.schedule --episodes 200 --part 5 --out datasets/parts_v2/schedule.json
    python -m g1lab.scripted.schedule --coverage datasets/parts_v2/attempts.jsonl --schedule datasets/parts_v2/schedule.json

Why a plan: episodes are only kept when they succeed. If every scene were drawn at random, the kept episodes
would lean toward whatever is easy (a small box, a box right in front of the robot). Here every kept episode
is assigned its strata beforehand, each stratum equally often; the recorder (box_into_box --scenes) keeps
trying scenes of that stratum until one succeeds.

  per episode   upright / upside down, which side faces the robot, near-far third, gap third,
                desk nearer / farther half in x and y, light dim / normal / bright
  per part      the box size (the simulator fixes it at start): scale 1.03-1.10, each axis +-2 %
"""

import argparse
import collections
import json

import numpy as np

FACTORS = {"flip": [0, 1], "face": [0, 1, 2, 3], "cell_x": [0, 1, 2], "cell_y": [0, 1, 2],
           "desk_x": [0, 1], "desk_y": [0, 1], "light": [0, 1, 2]}
LABELS = {"flip": ["upright", "upside down"], "face": ["side 0", "side 1", "side 2", "side 3"],
          "cell_x": ["near", "middle", "far"], "cell_y": ["small gap", "medium gap", "large gap"],
          "desk_x": ["desk nearer", "desk farther"], "desk_y": ["desk toward the centre", "desk away from it"],
          "light": ["dim", "normal", "bright"]}
SCALE = (1.03, 1.10)   # the real box is 5-10 % bigger than the calibrated model (with AXIS: 1.01 to 1.12)
AXIS = 0.02


def make_schedule(episodes=200, part=5, seed=1):
    rng = np.random.default_rng(seed)
    columns = {}
    for name, values in FACTORS.items():
        col = (values * (episodes // len(values) + 1))[:episodes]
        rng.shuffle(col)
        columns[name] = col
    scenes = [dict(seed=1000 + i, flip=columns["flip"][i], face=columns["face"][i],
                   cell=[columns["cell_x"][i], columns["cell_y"][i]],
                   desk=[columns["desk_x"][i], columns["desk_y"][i]], light=columns["light"][i])
              for i in range(episodes)]
    n = -(-episodes // part)
    base = np.linspace(*SCALE, n)
    rng.shuffle(base)
    parts = []
    for k in range(n):
        scale = base[k] * (1 + rng.uniform(-AXIS, AXIS, 3))
        parts.append(dict(scale=np.round(scale, 4).tolist(), sim_seed=int(seed * 1000 + k),
                          scenes=scenes[k * part:(k + 1) * part]))
    return dict(seed=seed, episodes=episodes, parts=parts)


def strata(want):
    return dict(flip=want["flip"], face=want["face"], cell_x=want["cell"][0], cell_y=want["cell"][1],
                desk_x=want["desk"][0], desk_y=want["desk"][1], light=want["light"])


def coverage(schedule, attempts):
    """Rows (factor, level, planned, recorded, attempts, success rate)."""
    planned, kept, tried = collections.Counter(), collections.Counter(), collections.Counter()
    for p in schedule["parts"]:
        for want in p["scenes"]:
            for item in strata(want).items():
                planned[item] += 1
    for a in attempts:
        if not a.get("want"):
            continue
        for item in strata(a["want"]).items():
            tried[item] += 1
            kept[item] += a["verdict"] == "success"
    rows = []
    for name, values in FACTORS.items():
        for v in values:
            k = (name, v)
            rows.append((name, LABELS[name][v], planned[k], kept[k], tried[k], kept[k] / max(1, tried[k])))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--part", type=int, default=5, help="episodes per part (one simulator run, one box size)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", help="write the schedule here")
    p.add_argument("--coverage", metavar="ATTEMPTS.jsonl", help="print planned / recorded / attempts per stratum")
    p.add_argument("--schedule", help="with --coverage: the schedule file")
    a = p.parse_args()
    if a.coverage:
        attempts = [json.loads(line) for line in open(a.coverage)]
        ok = sum(x["verdict"] == "success" for x in attempts)
        print(f"{ok} episodes kept from {len(attempts)} attempts ({100 * ok / max(1, len(attempts)):.0f} %)")
        print(f"{'':28s} planned  recorded  attempts  success")
        for name, label, planned, kept, tried, rate in coverage(json.load(open(a.schedule)), attempts):
            print(f"{label:28s} {planned:7d}  {kept:8d}  {tried:8d}  {100 * rate:6.0f} %")
        why = collections.Counter(x["verdict"].split(":")[0] if x["verdict"].startswith("no plan") else x["verdict"].split(" (")[0]
                                  for x in attempts if x["verdict"] != "success")
        print("attempts not kept:", dict(why))
        return
    schedule = make_schedule(a.episodes, a.part, a.seed)
    json.dump(schedule, open(a.out, "w"), indent=1)
    print(f"{a.out}: {a.episodes} episodes in {len(schedule['parts'])} parts")


if __name__ == "__main__":
    main()
