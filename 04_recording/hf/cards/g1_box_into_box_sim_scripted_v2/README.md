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

**Recording in progress.** Finished parts are uploaded under `parts/` every 10 episodes (each `parts/part_NN/` is a small LeRobot v3.0 dataset with 5 episodes). When all 200 episodes are recorded, the merged dataset replaces them at the root of this repository and this card gets the final numbers.

The episodes are **simulated**. A MuJoCo Unitree G1 (29 DoF, gloved rubber hands, D405 wrist cameras) runs in a real2sim replica of the lab desk and puts a bubble-wrapped box into a cardboard box. Simulator, scripted policy and recorder: [MOBILAB-UDESC/g1-mujoco-real2sim](https://github.com/MOBILAB-UDESC/g1-mujoco-real2sim).

| | |
|---|---|
| Task | `put the wrapped box inside the cardboard box` |
| State / action | 15 joints: 14 arm joints + waist yaw (`Unitree_G1_ArmsWaist_3Cam`), action = commanded joint targets |
| Cameras | `cam_head`, `cam_left_wrist`, `cam_right_wrist`, 640 x 480, 30 fps |
| Format | LeRobot v3.0 (lerobot 0.4.1) |

## What changed from version 1 (`g1_box_into_box_sim_scripted`)

Version 1 was tried on the real robot and failed; version 2 was made from what that test showed.

| | Version 1 | Version 2 |
|---|---|---|
| Grasp | hands lowered over the box from above | hands come in beside the box, below its top, then close in sideways |
| Arms and torso | upper arm up to 1.7 cm inside the torso shell | at least 2.5 cm clear, planned and checked |
| Arms and the cardboard box | not checked | never closer than about 1 cm |
| Joint command speed | up to 2.1 rad/s | at most 0.85 rad/s (the robot-side runners clip at 1.0) |
| Box size | 22.6 x 22.6 x 26.6 cm | 3 to 12 % bigger, different in every part |
| Box placement | 8 x 5.5 cm, yaw +-12 deg, upright | about 11 x 26 cm, one of its four sides toward the robot +-10 deg, upright or upside down |
| Desk relative to robot | fixed | 5 cm nearer to 2 cm farther, 5 cm toward the centre to 2 cm away |
| Light | each light x0.75-1.25 | dim / normal / bright levels, per-light variation, warm-cool tint |
| Camera field of view | fixed | +-4 % per camera |

Episodes are kept only when the box ends in the cardboard box and the measured motion kept the limits above. To avoid a dataset of "easy" scenes only, every episode was assigned its scene strata beforehand (orientation, facing side, position cell, desk offset, light), each equally often.

## Limits

- Simulation only. Object poses come from the simulator; the policy that generated the motions is scripted.
- The robot's arms are short for this task: the box cannot stand farther than 44 cm from the robot, and the desk cannot be farther away than in the ranges above.
- Against the cardboard walls the wrapped box counts as 2 cm smaller on every side (bubble wrap squashes); its sides are spring-loaded and give about 8 mm in the grip.
