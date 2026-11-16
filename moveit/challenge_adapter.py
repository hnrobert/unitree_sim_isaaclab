# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Map a MoveIt plan onto the UniBot challenge action interface.

The challenge API (unibot_submission) accepts action CHUNKS in one
of two mutually exclusive control_space modes:

  "joint": action.left_arm / action.right_arm  [T, 7] joint targets
  "ee":    action.left_ee_pose_gripper_base /
           action.right_ee_pose_gripper_base   [T, 6] xyz + rpy (base frame)

plus the shared action.left/right_gripper [T, 1] and action.pivot [T, 7].

joint mode is a direct re-packaging of the plan waypoints; ee mode
runs each waypoint through the arm FK (tools/arm_ik.py, base_link
torso_link - prepend your own base transform if the challenge base
frame differs). Usage from the repo (sim container or workstation):

  python moveit/challenge_adapter.py --in plan.json --mode ee --gripper 0.5
"""

import argparse
import json
import math
import os
import sys

REPO = "/home/code/unitree_sim_isaaclab"
sys.path.insert(0, os.path.join(REPO, "tools"))
from arm_ik import ArmKinematics  # noqa: E402

ARM_JOINTS = {
    "right": [f"right_{n}_joint" for n in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
               "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")],
    "left": [f"left_{n}_joint" for n in
             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
              "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")],
}


def _rpy_from_R(R):
    """Rotation matrix -> xyz rpy (extrinsic, rad) matching xyzw math."""
    sy = -R[2, 0]
    cy = math.sqrt(1 - min(1.0, sy * sy))
    if cy > 1e-6:
        p = math.atan2(R[2, 0], R[2, 2])
        r = math.atan2(R[1, 0], R[0, 0])
        y = math.atan2(sy, cy)
    else:  # gimbal lock
        p = math.atan2(-R[0, 1], R[1, 1])
        r = 0.0
        y = math.atan2(sy, cy)
    return r, p, y


def adapt(plan_json, mode, side, gripper, pivot=None):
    names = plan_json["names"]
    points = plan_json["points"]
    order = ARM_JOINTS[side]
    idx = [names.index(n) for n in order]

    out = {}
    if mode == "joint":
        out[f"action.{side}_arm"] = [[p[i] for i in idx] for p in points]
    else:  # ee
        kin = ArmKinematics(side, urdf=os.path.join(
            REPO, "assets", "urdf", "g1_d_description", "g1_d_with_dex1_1.urdf"))
        ees = []
        for p in points:
            q = [p[i] for i in idx]
            pos, R = kin.fk(q)
            r, pi_, y = _rpy_from_R(R)
            ees.append([float(pos[0]), float(pos[1]), float(pos[2]), r, pi_, y])
        out[f"action.{side}_ee_pose_gripper_base"] = ees
    t = len(points)
    out[f"action.{side}_gripper"] = [[gripper] for _ in range(t)]
    out["action.pivot"] = [list(pivot) if pivot else [0.0] * 7 for _ in range(t)]
    if side == "right":
        out["action.left_gripper"] = [[gripper] for _ in range(t)]
    else:
        out["action.right_gripper"] = [[gripper] for _ in range(t)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="plan json (from the sidecar)")
    ap.add_argument("--mode", choices=["joint", "ee"], default="joint")
    ap.add_argument("--side", default="right")
    ap.add_argument("--gripper", type=float, default=0.5, help="0 closed .. 1 open")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    with open(args.inp) as f:
        plan_json = json.load(f)
    out = adapt(plan_json, args.mode, args.side, args.gripper)
    text = json.dumps(out)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
