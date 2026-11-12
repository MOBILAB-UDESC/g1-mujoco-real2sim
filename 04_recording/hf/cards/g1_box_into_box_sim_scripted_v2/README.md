---
# LeRobot dataset card (Hub metadata spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md)
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- unitree
- g1
- simulation
- mujoco
- synthetic
---

# G1 box into box (MuJoCo sim, scripted, 15 joints), version 2

**200 episodes, 161,176 frames (90 minutes), all successful.** Episodes are 23 to 32 s long (mean 26.9 s). Every video frame was decoded once with the decoder LeRobot trains with (torchcodec) before upload.

The episodes are **simulated**. A MuJoCo Unitree G1 (29 DoF, gloved rubber hands, D405 wrist cameras) runs in a real2sim replica of the lab desk and puts a bubble-wrapped box into a cardboard box. Simulator, scripted policy and recorder: [MOBILAB-UDESC/g1-mujoco-real2sim](https://github.com/MOBILAB-UDESC/g1-mujoco-real2sim).

| | |
|---|---|
| Task | `put the wrapped box inside the cardboard box` |
| State / action | 15 joints: 14 arm joints + waist yaw (`Unitree_G1_ArmsWaist_3Cam`), action = commanded joint targets |
| Cameras | `cam_head`, `cam_left_wrist`, `cam_right_wrist`, 640 x 480, 30 fps |
| Format | LeRobot v3.0 (lerobot 0.4.1) |
| Per-episode scene | [`scenes.jsonl`](scenes.jsonl): box size, strata, desk / box / light / camera values and the plan's numbers, one line per episode |

## What changed from version 1 (`g1_box_into_box_sim_scripted`)

Version 1 was tried on the real robot and failed; version 2 was made from what that test showed.

| | Version 1 | Version 2 |
|---|---|---|
| Grasp | hands lowered over the box from above | hands come in beside the box, below its top, then close in sideways |
| Arms and torso | upper arm up to 1.7 cm inside the torso shell | at least 2.5 cm clear in the plan; measured 2.9 cm or more in every episode |
| Arms and the cardboard box | not checked | at least 1.2 cm clear in the plan; measured 0.7 cm or more in every episode (never touching) |
| Joint command speed | up to 2.1 rad/s | at most 0.85 rad/s, command at most 0.24 rad ahead of the joint (the robot-side runners clip at 1.0 rad/s and 0.25 rad) |
| Box size | 22.6 x 22.6 x 26.6 cm | 2 to 12 % bigger (mean 7.6 %), a different size every 5 episodes |
| Box placement | 8 x 5.5 cm, yaw +-12 deg, upright | about 11 x 26 cm, one of its four sides toward the robot +-10 deg, upright or upside down |
| Desk relative to robot | fixed | 5 cm nearer to 2 cm farther, 5 cm toward the centre to 2 cm away; the two shifts add up to between -5.5 and +1.5 cm |
| Light | each light x0.75-1.25 | dim / normal / bright levels, per-light variation, warm-cool tint |
| Camera field of view | fixed | +-4 % per camera |

Episodes are kept only when the box ends in the cardboard box and the measured motion kept the limits above. To avoid a dataset of "easy" scenes only, every episode was assigned its scene strata beforehand (orientation, facing side, position cell, desk offset, light), each equally often; a scene that failed was replaced by another one of the same strata.

| Stratum | Episodes |
|---|---|
| Upright / upside down | 100 / 100 |
| Side toward the robot (0, 1, 2, 3) | 50 / 50 / 50 / 50 |
| Box near / middle / far | 67 / 67 / 66 |
| Gap to the cardboard box small / medium / large | 67 / 67 / 66 |
| Desk nearer / farther | 100 / 100 |
| Desk toward the centre / away from it | 100 / 100 |
| Light dim / normal / bright | 67 / 67 / 66 |

It took 615 attempts. 399 never started because the planner found no motion that keeps the clearances, 6 ended with the box outside the cardboard box and 1 broke the lead limit; 9 successful episodes were recorded again with their part (three parts that ran out of attempts, one with a damaged video frame). Success per stratum was between 26 % and 51 %.

## Limits

- Simulation only. Object poses come from the simulator; the policy that generated the motions is scripted.
- The robot's arms are short for this task: the box cannot stand farther than 44 cm from the robot, and the desk cannot be farther away than in the ranges above.
- Two corners of the desk range are missing. With the desk both nearer and more toward the centre (shifts adding up to less than about -6 cm) no plan kept the left arm off both the torso and the cardboard box: 0 of 76 attempts. With the desk both farther and more to the left the cardboard box is out of reach. Some of the first 70 episodes were recorded before this limit was set; the lowest is at -5.8 cm.
- Within a stratum the kept scenes still lean toward the easier ones (about one attempt in three is kept).
- About 10 % of the camera frames repeat the previous frame (the simulated camera stream is not locked to the 30 Hz recording).
- Against the cardboard walls the wrapped box counts as 2 cm smaller on every side (bubble wrap squashes); its sides are spring-loaded and give about 8 mm in the grip.
