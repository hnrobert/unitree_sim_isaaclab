# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Headless test: the simulated Unitree G1-D Dex1 grippers actually GRASP and CARRY
(the Dex1 counterpart of tools/dex2_5_test.py, same methodology).

    python tools/dex1_pick_test.py --device cpu --viz none

Boots the OFFICIAL Isaac-PickPlace-Cylinder-G1D-Dex1-Joint task (its 0.4 kg
cylinder is the grasp object) without DDS: the action provider gets the
lowcmd a real client would send - arms / waist at 0, Dex1 grippers on the
INTERNAL slots (motors 31 left / 33 right, position control q 0 closed ..
5.4 open, enabled by mode 1 + kp) - from a stub. Prints ``TEST`` lines;
exit code 0 = all passed.

1. gripper control: q 0 closes both Dex1 grippers (position servo), the
   internal-slot wiring (mode / kp gating) is honoured
2. grasp: the task's cylinder is placed BETWEEN the fingers of each hand,
   the hand closes on it and must hold it against gravity - while the
   position command is active, with the command held, and while the lift
   column moves - and must drop it when opened
3. environment reset: grippers open again
"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("PROJECT_ROOT", REPO)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--task_name", default="Isaac-PickPlace-Cylinder-G1D-Dex1-Joint", help="a G1-D Dex1 task")
p.add_argument("--no_grasp", action="store_true", help="skip the grasp test (needs the task's rigid object 'object')")
add_launcher_args(p)
a = p.parse_args()

import tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(a.task_name, device=a.device, num_envs=1)
cfg.scene.world_camera = None  # no rendering needed
cfg.episode_length_s = 1.0e6  # no time-out reset during the test

FINGER_OPEN = 0.041       # [m] Dex1 prismatic finger travel (robots/unitree_g1d.py)
MOTOR_OPEN_Q = 5.4        # motor q of a fully open gripper
GRIP_KP = 800.0           # the URDF gripper actuator stiffness (kept coherent)
GRASP_LIFT = 0.20         # [m] the lift column is raised by this much first, so the cylinder hangs free
HOLD_S = 4.0              # [s] static hold duration


class LowCmdStub:
    """What dds.g1_robot_dds.G1RobotDDS.get_robot_command() returns for the lowcmd of a Dex1 client."""

    def __init__(self):
        self.pos = [0.0] * 35      # arms / waist / head at 0
        self.mode = [0] * 35
        self.kp = [0.0] * 35
        self.grip = {"left": 0.0, "right": 0.0}    # motor q command per side
        self.grip_on = {"left": False, "right": False}  # slot enable (mode 1 + kp)

    def set_grip(self, side, q, on=True):
        self.grip[side], self.grip_on[side] = float(q), bool(on)

    def get_robot_command(self):
        pos, kp, mode = list(self.pos), list(self.kp), list(self.mode)
        for slot, side in ((31, "left"), (33, "right")):
            if self.grip_on[side]:
                pos[slot], kp[slot], mode[slot] = self.grip[side], 5.0, 1
        return {"motor_cmd": {"modes": mode, "positions": pos, "velocities": [0.0] * 35,
                              "torques": [0.0] * 35, "kp": kp, "kd": [0.0] * 35}}

    def stop_communication(self):
        pass


class MobileStub:
    """rt/cmd_vel stand-in: only the lift speed (normalized z) is used, to raise the hands."""

    def __init__(self):
        self.column = 0.0

    def get_mobile_command(self):
        now = time.time()
        return {"lin_x": 0.0, "ang_z": 0.0, "move_stamp": 0.0,
                "height_speed": self.column, "height_stamp": now}

    def write_mobile_state(self, *args):
        pass

    def stop_communication(self):
        pass


with launch_simulation(cfg, a):
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    import tasks.common_observations.camera_state as cs

    cs._camera_cache["write_interval_steps"] = 10 ** 9  # no image shared memory
    env = gym.make(a.task_name, cfg=cfg).unwrapped
    env.reset()

    def T(x):
        return x.torch if hasattr(x, "torch") else x

    from action_provider.create_action_provider import create_action_provider
    from robots.unitree_g1d import G1D_LIFT_JOINTS
    from tasks.common_observations.g1d_state import ARM_JOINTS

    robot = env.scene["robot"]
    names = list(robot.data.joint_names)
    bodies = list(robot.data.body_names)
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        print(f"TEST {'ok  ' if cond else 'FAIL'} {msg}", flush=True)

    def info(msg):
        print(f"TEST info {msg}", flush=True)

    a.action_source, a.robot_type, a.task = "dds", "g1d", a.task_name
    a.enable_dex1_dds = a.enable_dex3_dds = a.enable_inspire_dds = False
    prov = create_action_provider(env, a)
    cmd = prov.robot_dds = LowCmdStub()
    mobile = prov.mobile_dds = MobileStub()
    dt = float(env.step_dt)
    arm_ids = [names.index(n) for n in ARM_JOINTS]
    lift_ids = [names.index(n) for n in G1D_LIFT_JOINTS]
    finger_ids = {s: [names.index(n) for n in (f"{s}_dex1_finger_joint_1", f"{s}_dex1_finger_joint_2")]
                  for s in ("left", "right")}

    def closure(side):
        """1 = fully closed, 0 = fully open (mean of the two prismatic fingers)."""
        q = T(robot.data.joint_pos)[0]
        c = sum(float(q[i]) for i in finger_ids[side]) / len(finger_ids[side])
        return (FINGER_OPEN - c) / FINGER_OPEN

    def run(seconds, each_step=None):
        for _ in range(max(1, round(seconds / dt))):
            if each_step is not None:
                each_step()
            env.step(prov.get_action(env))

    def fmt(v):
        return "[" + ", ".join(f"{float(x):+.2f}" for x in v) + "]"

    # ------------------------------------------------------------------ 1. position control
    run(1.0)
    c = {s: closure(s) for s in ("left", "right")}
    check(max(c.values()) < 0.05, f"default pose: both Dex1 grippers open (closure {fmt([c['left'], c['right']])})")

    for s in ("left", "right"):
        cmd.set_grip(s, 0.0)          # command fully closed
        run(1.5)
        c = closure(s)
        other = closure("right" if s == "left" else "left")
        check(c > 0.95 and other < 0.05, f"{s} gripper closes alone on motor q=0 (closure {c:.2f}, other {other:.2f})")
        cmd.set_grip(s, MOTOR_OPEN_Q)  # reopen
        run(1.0)
        check(closure(s) < 0.05, f"{s} gripper reopens on q=5.4 (closure {closure(s):.2f})")
        # slot gating: disabled slot (kp 0) must be ignored
        cmd.set_grip(s, 0.0, on=False)
        run(1.0)
        check(closure(s) < 0.05, f"{s} gripper ignores the slot when disabled (kp=0, closure {closure(s):.2f})")

    # ------------------------------------------------------------------ 2. grasp
    obj = env.scene["object"] if (not a.no_grasp and "object" in env.scene.keys()) else None
    if obj is None:
        info("grasp test skipped" + ("" if a.no_grasp else f": {a.task_name} has no rigid object 'object'"))
    else:
        env_ids = torch.tensor([0], device=env.device)
        obj_start = T(obj.data.root_pos_w)[0].clone()

        def lift_height():
            return float(T(robot.data.joint_pos)[0][lift_ids].sum())

        h0 = lift_height()
        mobile.column = 1.0
        for _ in range(round(10.0 / dt)):
            if lift_height() - h0 >= GRASP_LIFT:
                break
            env.step(prov.get_action(env))
        mobile.column = 0.0
        run(1.0)
        check(lift_height() - h0 > GRASP_LIFT - 0.02,
              f"lift column raised by {lift_height() - h0:.2f} m so the grasped cylinder hangs free of the table")

        for side in ("left", "right"):
            f1, f2 = (bodies.index(f"{side}_dex1_finger_link_1"), bodies.index(f"{side}_dex1_finger_link_2"))

            def finger_mid():
                """World midpoint between the two finger links (the natural grasp centre)."""
                b = T(robot.data.body_pos_w)[0]
                return (b[f1] + b[f2]) / 2.0

            def in_hand():
                """Cylinder centre relative to the finger midpoint (world axes)."""
                return T(obj.data.root_pos_w)[0] - finger_mid()

            def place():
                """Rest the cylinder with its TOP SECTION between the finger
                BODIES. finger_mid is at the finger ROOTS: the jaws extend
                ~6 cm forward from there (finger link +x), so walk 0.035 m
                along the finger axis into the jaw region; vertically the
                centre sits 0.155 m below so the top ~2 cm is between the
                jaws (higher pushes into the palm block, lower grips air)."""
                quat = T(robot.data.body_quat_w)[0][f1]
                fwd = quat_apply(quat, torch.tensor([0.035, 0.0, 0.0], device=env.device))
                pos = finger_mid() + fwd
                pos[2] -= 0.155
                pose = torch.cat([pos, torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)]).unsqueeze(0)
                obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
                obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=env_ids)

            cmd.set_grip(side, 0.0)
            run(1.5, each_step=place)  # the cylinder is held in place while the gripper closes on it
            c = closure(side)
            check(0.15 < c < 0.95, f"{side} gripper closes onto the cylinder and is stopped by it (closure {c:.2f})")
            before = in_hand().clone()
            run(1.0, each_step=None)  # released: held by the fingers only
            settled = in_hand().clone()
            run(HOLD_S)
            after = in_hand()
            z_obj = float(T(obj.data.root_pos_w)[0][2])
            drop = float((before - after)[2])
            creep = float((after - settled).norm())
            check(abs(drop) < 0.012 and creep < 0.004 and z_obj > obj_start[2] + 0.05,
                  f"{side} gripper holds the 0.4 kg cylinder for {HOLD_S:.0f} s in the raised column "
                  f"(slid {drop * 1000:+.1f} mm vertically, moved {creep * 1000:.1f} mm in the last 3 s, "
                  f"height {z_obj:.2f} m vs table start {float(obj_start[2]):.2f} m)")

            h_top = lift_height()
            mobile.column = -1.0  # carry it: column down and up again
            run(1.5)
            h_low = lift_height()
            mobile.column = 1.0
            run(1.5)
            mobile.column = 0.0
            run(1.0)
            carried = in_hand()
            check(h_top - h_low > 0.05 and float((carried - after).norm()) < 0.006,
                  f"{side} gripper carries the cylinder while the column moves {(h_top - h_low) * 100:.0f} cm down and up: "
                  f"moved {float((carried - after).norm()) * 1000:.1f} mm in the hand")

            z_obj = float(T(obj.data.root_pos_w)[0][2])
            cmd.set_grip(side, MOTOR_OPEN_Q)  # open: the cylinder must fall
            run(2.0)
            fell = z_obj - float(T(obj.data.root_pos_w)[0][2])
            check(fell > 0.05 and closure(side) < 0.1,
                  f"{side} gripper opened on q=5.4: cylinder released (fell {fell * 100:.0f} cm)")

    # ------------------------------------------------------------------ 3. reset
    run(1.0)
    env.reset()
    run(1.0)
    c = {s: closure(s) for s in ("left", "right")}
    check(max(c.values()) < 0.05, f"after an environment reset the grippers are open again (closure {fmt([c['left'], c['right']])})")
    if obj is not None:
        d = T(obj.data.root_pos_w)[0] - obj_start
        check(abs(float(d[2])) < 0.01 and float(d[:2].norm()) < 0.15,
              f"after the reset the cylinder stands on the table again "
              f"(height {float(d[2]) * 1000:+.0f} mm, {float(d[:2].norm()) * 100:.0f} cm from its first place)")

    print(f"TEST {'PASSED' if not failures else f'FAILED ({len(failures)})'}", flush=True)
    for f in failures:
        print(f"TEST   - {f}", flush=True)
    os._exit(0 if not failures else 1)
