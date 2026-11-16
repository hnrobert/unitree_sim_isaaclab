# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Execute MoveIt plans on the G1-D simulator through its own DDS path.

Runs INSIDE the sim container (next to sim_main.py, same network).
Usage:

  python tools/moveit_bridge.py --side right \
      --goal-world -8.43 1.42 0.88 0 -90 0      # world xyz + rpy deg
  python tools/moveit_bridge.py --gripper 0.5    # 0 closed .. 1 open

Planning is delegated to the MoveIt sidecar (`./run.sh moveit`,
ZMQ 12555). The returned joint trajectory is time-scaled, linearly
interpolated at 50 Hz and streamed to rt/lowcmd (motors 15..28,
kp 60 / kd 1.5 - the same values the keyboard teleop uses); the
gripper goes to rt/dex1/<side>/cmd. While idle the bridge holds
the CURRENT joint targets read from rt/lowstate, so taking over
never jumps.
"""

import argparse
import math
import sys
import threading
import time

sys.path.insert(0, "/home/code/unitree_sim_isaaclab")

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# lowcmd motor slots (README: arms are motors 15..28, left first)
ARM_MOTOR = {}
for _i, _n in enumerate(
        "left_shoulder_pitch left_shoulder_roll left_shoulder_yaw "
        "left_elbow left_wrist_roll left_wrist_pitch left_wrist_yaw "
        "right_shoulder_pitch right_shoulder_roll right_shoulder_yaw "
        "right_elbow right_wrist_roll right_wrist_pitch right_wrist_yaw".split()):
    ARM_MOTOR[_n + "_joint"] = 15 + _i
# whole-body extras (README / g1d_state.py): waist yaw = motor 12,
# waist pitch = 13; Dex1 grippers = lowcmd 31 (left) / 33 (right)
# with motor q 0 closed .. 5.4 open. The lift (LZ_mt/LZ_it, 0..0.21)
# has no lowcmd slot - it drives via rt/cmd_hispeed VELOCITY; the
# bridge runs a tiny P-loop for its position targets.
ARM_MOTOR["torso_Joint"] = 12
ARM_MOTOR["Yaw_Joint"] = 13
LIFT_JOINTS = {"LZ_mt_Joint", "LZ_it_Joint"}
GRIP_MOTOR = {"right_dex1_finger_joint_1": 33, "right_dex1_finger_joint_2": 33,
              "left_dex1_finger_joint_1": 31, "left_dex1_finger_joint_2": 31}
FINGER_TRAVEL = 0.041  # m, prismatic joint range
GRIP_MOTOR_OPEN = 5.4
GRIP_RANGE = (0.0, 1.0)


class Bridge:
    def __init__(self, domain=1, interface=None):
        ChannelFactoryInitialize(domain, interface) if interface else ChannelFactoryInitialize(domain)
        self.crc = CRC()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.grip_pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
        self.grip_pub.Init()
        self.lift_pub = ChannelPublisher("rt/cmd_hispeed", __import__(
            "unitree_sdk2py.idl.geometry_msgs.msg.dds_", fromlist=["Point32_"]).Point32_)
        self.lift_pub.Init()
        self.lift_target = None  # total lift height target (m)
        self.lowstate = None
        self.odom = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_lowstate, 1)
        # world->base uses --robot-base (the challenge robot reports its
        # own pose; the sim default matches the task layout spawn)
        self.q = [0.0] * 29
        self.mode_machine = 0
        self._run = True
        threading.Thread(target=self._publish_loop, daemon=True).start()

    def _on_lowstate(self, msg):
        self.lowstate = msg
        self.mode_machine = msg.mode_machine

    def wait_for_sim(self, timeout=10.0):
        t0 = time.time()
        while self.lowstate is None and time.time() - t0 < timeout:
            time.sleep(0.05)
        if self.lowstate is None:
            raise SystemExit("no rt/lowstate - run sim_main.py --robot_type g1d first")
        for i in range(29):
            self.q[i] = float(self.lowstate.motor_state[i].q)
        print(f"connected; mode_machine={self.mode_machine}")

    def _publish_loop(self):
        import unitree_sdk2py.idl.unitree_hg.msg.dds_ as hg
        import unitree_sdk2py.idl.unitree_go.msg.dds_ as go
        while self._run:
            try:
                msg = hg.LowCmd_()
            except TypeError:  # idl dataclass needs every field
                # NOTE the HG LowCmd slots are unitree_hg MotorCmd (its
                # reserve is a SCALAR) - the unitree_go MotorCmd's list
                # reserve breaks the CRC struct pack
                msg = hg.LowCmd_(
                    mode_pr=0, mode_machine=self.mode_machine,
                    motor_cmd=[hg.MotorCmd_(mode=1, q=0.0, dq=0.0, tau=0.0,
                                            kp=60.0, kd=1.5, reserve=0)
                               for _ in range(35)],
                    reserve=[0, 0, 0, 0], crc=0)  # uint32[4] fixed
            msg.mode_pr = 0
            msg.mode_machine = self.mode_machine
            for i in range(29):
                m = msg.motor_cmd[i]
                m.mode = 1
                m.q = float(self.q[i])
                m.dq = 0.0
                m.tau = 0.0
                m.kp = 60.0
                m.kd = 1.5
            msg.crc = self.crc.Crc(msg)
            self.pub.Write(msg)
            time.sleep(0.02)  # 50 Hz

    def _lift_velocity(self, height_m):
        """rt/cmd_hispeed is a VELOCITY command (teleop sends speeds);
        run a naive P-loop toward the requested lift height."""
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_
        p = Point32_()
        p.z = max(-0.05, min(0.05, 2.0 * (height_m - getattr(self, "_lift_now", 0.0))))
        self._lift_now = height_m
        self.lift_pub.Write(p)

    def gripper(self, open01):
        # q in real-motor units: 0 closed .. 5.4 open (README)
        q = 5.4 * float(min(max(open01, 0.0), 1.0))
        g = MotorCmds_(cmds=[MotorCmd_(mode=1, q=q, dq=0.0, tau=0.0,
                                      kp=5.0, kd=0.05, reserve=[0, 0, 0])])  # uint32[3]
        self.grip_pub.Write(g)

    def execute(self, names, times, points, time_scale=0.8, v_rad_s=0.6):
        """Interpolate the plan onto the arm motors and run it."""
        if len(points) > 1 and max(times) < 1e-6:
            # zero-timestamp plan: time-parameterize by joint speed
            times = [0.0]
            for a, b in zip(points, points[1:]):
                dq = max(abs(b[j] - a[j]) for j in range(len(a)))
                times.append(times[-1] + max(dq / v_rad_s, 0.04))
        t0 = time.time()
        total = (times[-1] if times else 0.0) / time_scale
        k = 0
        # snapshot the lift start height from lowstate (motor q)
        lift_start = self.q[12] * 0.0  # placeholder; lowstate lacks lift
        finger_names = [n for n in names if n in GRIP_MOTOR]
        while True:
            t = (time.time() - t0) * time_scale
            while k + 1 < len(times) and times[k + 1] < t:
                k += 1
            if k + 1 >= len(times):
                break
            t0k, t1k = times[k], times[k + 1]
            a = 0.0 if t1k <= t0k else min(1.0, (t - t0k) / (t1k - t0k))
            lift_sum = 0.0
            for j, name in enumerate(names):
                v = points[k][j] + a * (points[k + 1][j] - points[k][j])
                if name in GRIP_MOTOR:
                    # finger prismatic m -> gripper motor q (0 closed..5.4 open)
                    self.gripper(1.0 - min(1.0, max(0.0, v / FINGER_TRAVEL)))
                elif name in LIFT_JOINTS:
                    lift_sum += v
                else:
                    slot = ARM_MOTOR.get(name)
                    if slot is not None:
                        self.q[slot] = v
            if lift_sum > 0.0:
                self._lift_velocity(lift_sum)
            time.sleep(0.02)
        for j, name in enumerate(names):  # settle at the final point
            if name in GRIP_MOTOR or name in LIFT_JOINTS:
                continue
            slot = ARM_MOTOR.get(name)
            if slot is not None:
                self.q[slot] = points[-1][j]
        print(f"executed {len(points)} waypoints in {total:.1f}s")


def world_to_base(wx, wy, wz, bx, by, byaw):
    c, s = math.cos(byaw), math.sin(byaw)
    dx, dy = wx - bx, wy - by
    return (c * dx + s * dy, -s * dx + c * dy, wz)


def local_ik(side, goal_torso, seed_q):
    """DLS inverse kinematics LOCALLY (tools/arm_ik.py, numpy) - the
    same solver our demo uses; no ROS round-trip, no service flakiness.
    goal_torso: [x, y, z, roll, pitch, yaw] for the DEX1-BASE link
    (= wrist + 0.0415 m along the forearm axis, so tcp_offset below)."""
    from arm_ik import ArmKinematics
    import numpy as np
    kin = ArmKinematics(side, tcp_offset=(0.0415, 0.0, 0.0))
    x, y, z, r, p, yw = goal_torso
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(yw), math.sin(yw))
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    q, ep, er = kin.ik(np.array([x, y, z]), R, np.array(seed_q))
    if ep > 0.005 or er > 0.15:
        return {"ok": False, "error": f"no ik (perr {ep:.3f} rerr {er:.2f})"}
    return {"ok": True, "names": kin.names, "points": [list(q)], "times": [0.0]}


def plan(side, goal_base_rpy, seed=None, op="plan"):
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://127.0.0.1:12555")
    sock.send_json({"op": op, "side": side, "goal": list(goal_base_rpy),
                    "seed": seed or {}})
    rep = sock.recv_json()
    sock.close()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--goal-world", nargs=6, type=float, metavar=("X", "Y", "Z", "R", "P", "YAW"),
                    help="gripper-base goal in WORLD frame (rpy in degrees)")
    ap.add_argument("--goal-base", nargs=6, type=float, metavar=("X", "Y", "Z", "R", "P", "YAW"),
                    help="gripper-base goal in ROBOT-BASE frame")
    ap.add_argument("--robot-base", nargs=3, type=float, default=(-8.5, 0.35, 0.0),
                    metavar=("X", "Y", "YAW"), help="robot world pose (radians yaw)")
    ap.add_argument("--gripper", type=float, help="0 closed .. 1 open (no planning)")
    ap.add_argument("--ik", action="store_true",
                    help="skip OMPL: solve IK for the goal and move there directly")
    ap.add_argument("--ik-service", action="store_true",
                    help="use the sidecar /compute_ik service instead of local DLS")
    args = ap.parse_args()

    bridge = Bridge()
    bridge.wait_for_sim()
    if args.gripper is not None:
        bridge.gripper(args.gripper)
        time.sleep(1.0)
        return

    if args.goal_world:
        x, y, z = world_to_base(args.goal_world[0], args.goal_world[1], args.goal_world[2],
                                *args.robot_base)
        r, p, yw = (math.radians(v) for v in args.goal_world[3:])
    elif args.goal_base:
        x, y, z = args.goal_base[:3]
        r, p, yw = (math.radians(v) for v in args.goal_base[3:])
    else:
        ap.error("need --goal-world or --goal-base (or --gripper)")

    seed = {}
    for name, slot in ARM_MOTOR.items():
        seed[name] = bridge.q[slot]
    if args.ik:
        # LOCAL damped-least-squares IK (arm_ik.py) - deterministic,
        # no ROS hop; falls back to the sidecar service with --ik-service
        seed_q = [seed.get(n, 0.0) for n in
                  (f"{args.side}_{j}_joint" for j in
                   ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))]
        rep = local_ik(args.side, (x, y, z, r, p, yw), seed_q)
        if not rep.get("ok") and args.ik_service:
            rep = plan(args.side, (x, y, z, r, p, yw), seed, op="ik")
    else:
        rep = plan(args.side, (x, y, z, r, p, yw), seed)
    if not rep.get("ok"):
        raise SystemExit(f"planning failed: {rep.get('error')}")
    if args.ik:
        q = rep["points"][0]
        print(f"ik ok: q = ['" + "', '".join('%.3f' % v for v in q) + "']")
        # smooth 2 s move to the IK solution (seed -> solution)
        seed_q = [seed.get(n, 0.0) for n in rep["names"]]
        bridge.execute(rep["names"], [0.0, 2.0], [seed_q, q], time_scale=1.0)
        return
    print(f"plan ok: {len(rep['points'])} waypoints, {rep['times'][-1]:.1f}s")
    bridge.execute(rep["names"], rep["times"], rep["points"])


if __name__ == "__main__":
    main()
