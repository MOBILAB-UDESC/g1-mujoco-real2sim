---
# LeRobot dataset card (Hub metadata spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md)
license: apache-2.0   # TODO: confirm the license with the lab before publishing
task_categories:
- robotics
tags:
- LeRobot
- unitree
- g1
- simulation
- mujoco
- synthetic
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# G1 box into box (MuJoCo sim, scripted, 15 joints)

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot) (format v3.0, lerobot 0.4.1).

The episodes are **simulated**. A MuJoCo Unitree G1 (29 DoF, gloved rubber hands, D405 wrist cameras) runs in a real2sim replica of the lab desk. It is driven over the same DDS + teleimager channels as the real robot. A code-as-policy script (`g1lab.scripted.box_into_box`) solves the task from the sim's privileged object poses through xr_teleoperate's IK. It squeezes the bubble-wrapped box between the palms, lifts it, turns the waist toward the cardboard box and lowers it inside. **Only episodes that pass the sim's success check are kept.** Every episode has a randomized start pose, speed, grasp parameters and object poses, and the sim adds visual randomization (`--randomize`).

- **Task:** `{{TASK}}`
- **Episodes / frames:** {{TOTAL_EPISODES}} / {{TOTAL_FRAMES}} at 30 fps
- **robot_type:** `Unitree_G1_ArmsWaist_3Cam`
- **Joints (15):** the 14 arm joints (left, then right: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) + `kWaistYaw` (G1 motor 12)

## Features

| key | type | shape | notes |
|---|---|---|---|
| `observation.state` | float32 | 15 | measured joint positions in sim (rad) |
| `action` | float32 | 15 | commanded joint targets (rad), as sent on rt/lowcmd |
| `observation.images.cam_head` | video (AV1, yuv420p) | 480x640x3 | rendered head camera |
| `observation.images.cam_left_wrist` | video | 480x640x3 | rendered left wrist camera |
| `observation.images.cam_right_wrist` | video | 480x640x3 | rendered right wrist camera |

The layout is the same as `<LAB_HF_ORG>/g1_pick_cube_real_waist`, so the two merge directly for real+sim training. The sim images are renders: expect a visual domain gap.

## Citation

[More Information Needed]
