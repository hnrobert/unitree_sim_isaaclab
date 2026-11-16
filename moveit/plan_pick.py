# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""MoveIt planning sidecar for the G1-D + Dex1 stack (ZMQ service).

Runs against the classic move_group stack (launch_g1d.py): a pure
rclpy client that calls the /plan_kinematic_path service
(moveit_msgs/srv/GetMotionPlan) and returns the joint trajectory.

Requests (REQ on tcp://*:12555):
  {"op": "plan",
   "side": "right",
   "goal": [x, y, z, roll, pitch, yaw],   # gripper-base pose, TORSO frame
   "seed": {"joint": value, ...}}          # start-state overrides
  {"op": "scene", "boxes": [[cx, cy, cz, hx, hy, hz], ...]}  # torso frame

Reply:
  {"ok": true, "names": [...], "times": [t...], "points": [[q...], ...]}
"""

import argparse
import json
import math
import os
import threading
import traceback

import rclpy
import zmq
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from moveit_msgs.msg import (CollisionObject, Constraints, MotionPlanRequest,
                             OrientationConstraint, PositionConstraint,
                             RobotState)
from moveit_msgs.srv import GetMotionPlan, GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

PORT = 12555


def _quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class Planner(Node):
    def __init__(self):
        super().__init__("unibot_plan_client")
        self.plan_srv = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.ik_srv = self.create_client(GetPositionIK, "/compute_ik")
        self.fk_srv = self.create_client(GetPositionFK, "/compute_fk")
        self.scene_pub = self.create_publisher(CollisionObject, "/collision_object", 10)
        while not self.plan_srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for move_group /plan_kinematic_path ...")

    def send_scene(self, boxes):
        now = self.get_clock().now().to_msg()
        for i, (cx, cy, cz, hx, hy, hz) in enumerate(boxes):
            obj = CollisionObject()
            obj.header.frame_id = "torso_link"
            obj.header.stamp = now
            obj.id = f"unibot_box_{i:03d}"
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(2 * hx), float(2 * hy), float(2 * hz)]
            obj.primitives.append(prim)
            obj.primitive_poses.append(Pose(
                position=Point(x=float(cx), y=float(cy), z=float(cz)),
                orientation=Quaternion(w=1.0)))
            obj.operation = CollisionObject.ADD
            self.scene_pub.publish(obj)
        self.get_logger().info(f"scene: {len(boxes)} boxes published")

    def _wait(self, fut, timeout=15.0):
        import time as _t
        _t0 = _t.time()
        while not fut.done() and _t.time() - _t0 < timeout:
            _t.sleep(0.05)
        return fut.done()

    def ik(self, req):
        """Pose -> joint solution (move_group /compute_ik, KDL)."""
        side = req.get("side", "right")
        goal = req["goal"]
        seed = req.get("seed") or {}
        from moveit_msgs.msg import PositionIKRequest
        from moveit_msgs.srv import GetPositionIK as _G
        r = PositionIKRequest()
        r.group_name = f"{side}_arm"
        r.ik_link_name = f"{side}_dex1_base_link"
        r.pose_stamped.header.frame_id = "torso_link"
        r.pose_stamped.pose.position = Point(x=goal[0], y=goal[1], z=goal[2])
        q = _quat(goal[3], goal[4], goal[5])
        r.pose_stamped.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        r.timeout.sec = 2
        if seed:
            js = JointState()
            js.name = list(seed.keys())
            js.position = [float(v) for v in seed.values()]
            r.robot_state = RobotState(joint_state=js)
        srv = _G.Request()
        srv.ik_request = r
        fut = self.ik_srv.call_async(srv)
        if not self._wait(fut):
            return {"ok": False, "error": "ik timeout"}
        res = fut.result()
        if res.error_code.val != 1:
            return {"ok": False, "error": f"ik error {res.error_code.val}"}
        js = res.solution.joint_state
        names, points = [], []
        for i, n in enumerate(js.name):
            if "arm" in n or ("_joint" in n and ("shoulder" in n or "elbow" in n or "wrist" in n)):
                names.append(n)
                points.append(float(js.position[i]))
        return {"ok": True, "names": names, "points": [points],
                "times": [0.0], "seed_used": len(seed) > 0}

    def fk(self, req):
        from moveit_msgs.srv import GetPositionFK as _F
        side = req.get("side", "right")
        joints = req.get("joints") or {}
        js = JointState()
        js.name = list(joints.keys())
        js.position = [float(v) for v in joints.values()]
        srv = _F.Request()
        srv.fk_link_names = [f"{side}_dex1_base_link"]
        srv.robot_state.joint_state = js
        srv.header.frame_id = "torso_link"
        fut = self.fk_srv.call_async(srv)
        if not self._wait(fut):
            return {"ok": False, "error": "fk timeout"}
        res = fut.result()
        if res.error_code.val != 1:
            return {"ok": False, "error": f"fk error {res.error_code.val}"}
        ps = res.pose_stamped[0].pose
        import math as _m
        # quaternion -> rpy
        w, x, y, z = ps.orientation.w, ps.orientation.x, ps.orientation.y, ps.orientation.z
        rpy = [_m.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
               _m.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
               _m.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))]
        return {"ok": True, "pose": [ps.position.x, ps.position.y, ps.position.z, *rpy]}

    def plan_joints(self, req):
        """Joint-space goal planning (any group incl. waist/grippers/whole)."""
        group = req["group"]
        target = req["target"]  # {joint: value}
        seed = req.get("seed") or {}
        mreq = MotionPlanRequest()
        if seed:
            js = JointState()
            js.name = list(seed.keys())
            js.position = [float(v) for v in seed.values()]
            mreq.start_state = RobotState(joint_state=js)
        mreq.group_name = group
        mreq.num_planning_attempts = 5
        mreq.allowed_planning_time = 4.0
        mreq.max_velocity_scaling_factor = 0.4
        mreq.max_acceleration_scaling_factor = 0.4
        mreq.goal_constraints.append(Constraints(
            name="joint_goal",
            joint_constraints=[__import__("moveit_msgs.msg", fromlist=["JointConstraint"]).JointConstraint(
                joint_name=n, position=float(v), tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
                for n, v in target.items()]))
        srv_req = GetMotionPlan.Request()
        srv_req.motion_plan_request = mreq
        fut = self.plan_srv.call_async(srv_req)
        if not self._wait(fut):
            return {"ok": False, "error": "planning service timeout"}
        res = fut.result()
        if res.motion_plan_response.error_code.val != 1:
            return {"ok": False, "error": f"planner error {res.motion_plan_response.error_code.val}"}
        jt = res.motion_plan_response.trajectory.joint_trajectory
        names = list(jt.joint_names)
        times, points = [], []
        for p in jt.points:
            times.append(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9)
            points.append([float(v) for v in p.positions])
        return {"ok": True, "names": names, "times": times, "points": points}

    def plan(self, req):
        side = req.get("side", "right")
        goal = req["goal"]
        seed = req.get("seed") or {}

        mreq = MotionPlanRequest()
        mreq.group_name = f"{side}_arm"
        mreq.num_planning_attempts = 5
        mreq.allowed_planning_time = 4.0
        mreq.max_velocity_scaling_factor = 0.4
        mreq.max_acceleration_scaling_factor = 0.4
        if seed:
            js = JointState()
            js.name = list(seed.keys())
            js.position = [float(v) for v in seed.values()]
            mreq.start_state = RobotState(joint_state=js)

        link = f"{side}_dex1_base_link"
        q = _quat(goal[3], goal[4], goal[5])
        pc = PositionConstraint()
        pc.header.frame_id = "torso_link"
        pc.link_name = link
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)  # Vector3, NOT Point (C assert!)
        pc.constraint_region.primitives.append(SolidPrimitive(
            type=SolidPrimitive.SPHERE, dimensions=[0.005]))
        pc.constraint_region.primitive_poses.append(Pose(
            position=Point(x=goal[0], y=goal[1], z=goal[2]),
            orientation=Quaternion(w=1.0)))
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = "torso_link"
        oc.link_name = link
        oc.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        oc.absolute_x_axis_tolerance = 0.05
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0
        goal_c = Constraints()
        goal_c.name = "unibot_goal"
        goal_c.position_constraints.append(pc)
        goal_c.orientation_constraints.append(oc)
        mreq.goal_constraints.append(goal_c)

        srv_req = GetMotionPlan.Request()
        srv_req.motion_plan_request = mreq
        fut = self.plan_srv.call_async(srv_req)
        # NOTE no spin_until_future_complete here: a background thread
        # already spins this node (double-spin deadlocks rclpy)
        import time as _t
        _t0 = _t.time()
        while not fut.done() and _t.time() - _t0 < 15.0:
            _t.sleep(0.05)
        if not fut.done():
            return {"ok": False, "error": "planning service timeout"}
        res = fut.result()
        if res.motion_plan_response.error_code.val != 1:
            return {"ok": False,
                    "error": f"planner error {res.motion_plan_response.error_code.val}"}
        jt = res.motion_plan_response.trajectory.joint_trajectory
        names = list(jt.joint_names)
        times, points = [], []
        for p in jt.points:
            times.append(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9)
            points.append([float(v) for v in p.positions])
        return {"ok": True, "names": names, "times": times, "points": points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    planner = Planner()
    threading.Thread(target=rclpy.spin, args=(planner,), daemon=True).start()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{PORT}")
    print(f"[plan] serving on {PORT}", flush=True)
    while True:
        req = json.loads(sock.recv())
        try:
            op = req.get("op", "plan")
            if op == "scene":
                planner.send_scene(req["boxes"])
                rep = {"ok": True}
            elif op == "ik":
                rep = planner.ik(req)
            elif op == "fk":
                rep = planner.fk(req)
            elif op == "plan_joints":
                rep = planner.plan_joints(req)
            else:
                rep = planner.plan(req)
        except Exception:
            traceback.print_exc()
            rep = {"ok": False, "error": "plan error (see sidecar log)"}
        sock.send_json(rep)


if __name__ == "__main__":
    main()
