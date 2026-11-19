#!/usr/bin/env python3
# T20 paper-cup stacking station server: MCP-exported camera + arm interfaces.
#
# run_T20.sh parks the robot at the table_03 station (UNIBOT_SPAWN) with
# ALL navigation disabled - the base never moves. This server, running in
# the sim container, exports everything a remote policy needs:
#
#   cameras   the 4 challenge views (head stereo L/R + wrist L/R) grabbed
#             from the existing teleimager ZMQ JPEG streams
#   arms      rt/lowcmd 50 Hz stream (the e2e-proven moveit_bridge.Bridge)
#             + OFFICIAL arm_ik (tools/arm_ik.py) for ee targets
#   grippers  rt/dex1/<side>/cmd (motor q 0 closed .. 5.4 open)
#   lift      rt/cmd_hispeed P-loop toward a height target
#
# exported as a minimal streamable-HTTP MCP endpoint on :12580 (JSON-RPC,
# stdlib only). The host driver (tools/t20_driver.py) calls glm-5.3-flash
# with the 4 views and dispatches its tool calls through this endpoint;
# any other MCP client (claude --mcp-config) can drive the same tools.

import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402

from arm_ik import ArmKinematics  # noqa: E402  (official solver)
from moveit_bridge import ARM_MOTOR, Bridge  # noqa: E402 (proven lowcmd stream)

MCP_PORT = 12580
# "main" = the MAIN WINDOW camera stream (:55562, sim_main viewport grab):
# whatever framing the GUI viewport shows - for run_T20.sh that is the
# close over-the-shoulder view of the table workspace (UNIBOT_VIEW).
# It only streams when the sim runs with UNIBOT_MAINVIEW=1 (--vlm) or
# UNIBOT_RECORD=1; without it the view is simply absent from get_views.
VIEWS = ("main", "head_left", "head_right", "wrist_left", "wrist_right")
MAIN_PORT = 55562
FRAME_HELP = "torso frame: x = forward (toward the table), y = left, z = up; metres"

# ---- camera grabber (latest-frame CONFLATE subs, like camera_windows) ----
_frames = {v: None for v in VIEWS}


def _cam_worker():
    ctx = zmq.Context()

    def sub(port):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://127.0.0.1:{port}")
        return s

    stereo, lw, rw = sub(55555), sub(55556), sub(55557)
    main = sub(MAIN_PORT)
    while True:
        try:
            if main.poll(0):
                _frames["main"] = main.recv()
            if stereo.poll(50):
                buf = stereo.recv()
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[1] >= 2:
                    half = img.shape[1] // 2
                    _frames["head_left"] = cv2.imencode(".jpg", img[:, :half],
                                                        [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                    _frames["head_right"] = cv2.imencode(".jpg", img[:, half:],
                                                         [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            for sock, view in ((lw, "wrist_left"), (rw, "wrist_right")):
                if sock.poll(0):
                    _frames[view] = sock.recv()
        except Exception:
            time.sleep(0.1)


# ---- robot bridge ------------------------------------------------------------
# Dex1-base TCP convention of moveit_bridge.local_ik: 0.0415 m ahead of the wrist
KIN = {s: ArmKinematics(s, tcp_offset=(0.0415, 0.0, 0.0)) for s in ("left", "right")}
BRIDGE = None  # created in main()


def _arm_q(side):
    """Actual 7 joint angles (rad, arm_ik name order) from rt/lowstate."""
    ls = BRIDGE.lowstate
    if ls is None:
        return None
    names = KIN[side].names
    slots = [ARM_MOTOR[n] for n in names]
    return [float(ls.motor_state[s].q) for s in slots]


def _home_q(side):
    from robots.unitree_g1d import G1D_ARM_JOINT_POS
    import re
    out = []
    for n in KIN[side].names:
        val = 0.0
        for pat, v in G1D_ARM_JOINT_POS.items():
            if re.fullmatch(pat, n):
                val = v
                break
        out.append(val)
    return out


def _stream_joints(side, q_targets, v_rad_s=0.4):
    """Interpolate the arm to q_targets (arm_ik name order) on the Bridge stream.

    Targets are CLAMPED to the URDF joint limits first: VLM / policy outputs
    can leave the valid range (an early ACT checkpoint produced out-of-limit
    targets and the arm twisted toward them)."""
    kin = KIN[side]
    q_targets = [max(lo, min(hi, q)) for q, lo, hi in
                 zip(q_targets, kin.lower, kin.upper)]
    names = KIN[side].names
    slots = [ARM_MOTOR[n] for n in names]
    q0 = _arm_q(side) or _home_q(side)
    dq = max(abs(a - b) for a, b in zip(q0, q_targets))
    n_steps = max(2, int(math.ceil(dq / v_rad_s / 0.04)))
    for i in range(1, n_steps + 1):
        a = i / n_steps
        for s, qa, qb in zip(slots, q0, q_targets):
            BRIDGE.q[s] = qa + a * (qb - qa)
        time.sleep(0.04)
    for s, qb in zip(slots, q_targets):
        BRIDGE.q[s] = qb


Z_FLOOR = -0.01  # tcp height floor: below this the hand plows the tabletop
                 # (observed: VLM descents to z -0.02..-0.05 tipped every cup)

# TCP workspace bounds (torso frame) - the collision safety layer of the
# pose planner: every path WAYPOINT must pass _tcp_safe before it is
# executed. Keeps the hand out of the torso volume, behind the body and
# below the tabletop; joint-level self-collision remains PhysX-guarded
# (enabled_self_collisions=True in robots/unitree_g1d.py).
TCP_BOUNDS = ((-0.05, 0.55), (-0.45, 0.45), (Z_FLOOR, 0.60))  # x, y, z
TORSO_XY_EXCLUDE = (0.10, 0.20)  # x < 0.10 AND |y| < 0.20 = inside the torso


def _tcp_safe(p) -> str | None:
    """None if the waypoint is in the safe workspace, else the reason."""
    for v, (lo, hi) in zip(p, TCP_BOUNDS):
        if not lo <= v <= hi:
            return f"{v:+.2f} outside the workspace [{lo}..{hi}]"
    if p[0] < TORSO_XY_EXCLUDE[0] and abs(p[1]) < TORSO_XY_EXCLUDE[1]:
        return f"inside the torso volume (x {p[0]:+.2f}, y {p[1]:+.2f})"
    return None


def _rpy_to_R(roll, pitch, yaw):
    """Rotation matrix from torso-frame RPY (Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _R_to_rpy(R):
    """Current orientation as (roll, pitch, yaw) matching _rpy_to_R."""
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def op_set_ee_pose(side, x, y, z, roll=None, pitch=None, yaw=None, speed=0.5):
    """Plan and execute ONE absolute gripper-pose command.

    The MODEL only states the goal pose; the motion itself is planned here:
    a straight TCP path resampled into <= 4 cm waypoints, every waypoint
    checked against the safe workspace (no torso, no below-table, no
    behind-body), solved by the OFFICIAL arm_ik (null-space pulled toward
    the home posture so the elbow never swings into the body) and executed
    joint-interpolated at a bounded speed. Optional roll/pitch/yaw reorient
    the wrist (absolute, torso frame; omitted values keep the current
    orientation) - e.g. pitch about +pi to invert a cup."""
    q = _arm_q(side)
    if q is None:
        return "error: no rt/lowstate yet"
    p0, R0 = KIN[side].fk(q)
    tgt = np.array([float(x), float(y), float(z)])
    why = _tcp_safe(tgt)
    if why:
        return f"error: target ({tgt[0]:+.2f},{tgt[1]:+.2f},{tgt[2]:+.2f}) rejected - {why}"
    r0, p0r, y0 = _R_to_rpy(R0)
    R = _rpy_to_R(r0 if roll is None else float(roll),
                  p0r if pitch is None else float(pitch),
                  y0 if yaw is None else float(yaw))
    dist = float(np.linalg.norm(tgt - p0))
    n_wp = max(1, int(math.ceil(dist / 0.04)))
    q_prev = q
    for i in range(1, n_wp + 1):
        pi = p0 + (tgt - p0) * (i / n_wp)
        why = _tcp_safe(pi)
        if why:
            return (f"aborted at waypoint {i}/{n_wp} ({pi[0]:+.2f},{pi[1]:+.2f},{pi[2]:+.2f}) "
                    f"- path is unsafe: {why}; arm left at the last safe pose")
        q_i, ep, er = KIN[side].ik(pi, R, q_prev, q_rest=_home_q(side), iters=40)
        if ep > 0.012:  # seed stuck in a local minimum - retry from home
            q_i, ep, er = KIN[side].ik(pi, R, _home_q(side), q_rest=_home_q(side), iters=60)
        if ep > 0.012:
            return (f"aborted at waypoint {i}/{n_wp}: unreachable (pos err {ep:.3f} m) "
                    f"- try a closer target or change the lift")
        _stream_joints(side, list(q_i), v_rad_s=max(0.3, min(0.8, float(speed))))
        q_prev = q_i
    p1, _ = KIN[side].fk(q_prev)
    return (f"{side} tcp at ({p1[0]:+.3f},{p1[1]:+.3f},{p1[2]:+.3f}) [{FRAME_HELP}] "
            f"(planned {n_wp} waypoints over {dist:.2f} m, collision-checked); "
            f"requested ({x:+.2f},{y:+.2f},{z:+.2f})")


def op_move_ee(side, dx, dy, dz, speed=0.4):
    q = _arm_q(side)
    if q is None:
        return "error: no rt/lowstate yet"
    p, R = KIN[side].fk(q)
    tgt = p + np.array([dx, dy, dz], dtype=float)
    why = _tcp_safe(tgt)
    if why:
        return (f"error: target ({tgt[0]:+.3f},{tgt[1]:+.3f},{tgt[2]:+.3f}) "
                f"rejected - {why}; the move is NOT executed")
    v = max(0.03, min(0.4, float(speed)))  # rad/s joint speed
    q_new, ep, er = KIN[side].ik(tgt, R, q, q_rest=_home_q(side))
    if ep > 0.012:
        return (f"error: target ({tgt[0]:+.3f},{tgt[1]:+.3f},{tgt[2]:+.3f}) "
                f"unreachable (pos err {ep:.3f} m) - move closer / use the lift")
    _stream_joints(side, list(q_new), v_rad_s=v)
    p2, _ = KIN[side].fk(_arm_q(side) or q_new)
    return (f"moved {side} tcp -> ({p2[0]:+.3f},{p2[1]:+.3f},{p2[2]:+.3f}) "
            f"[{FRAME_HELP}]; requested d=({dx:+.3f},{dy:+.3f},{dz:+.3f}) "
            f"at {v:.2f} rad/s")


# grippers: own publishers for BOTH sides (Bridge ships right only)
class Grippers:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
        self.pub = {}
        for side in ("left", "right"):
            p = ChannelPublisher(f"rt/dex1/{side}/cmd", MotorCmds_)
            p.Init()
            self.pub[side] = p

    def set(self, side, open01):
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_
        q = 5.4 * float(min(1.0, max(0.0, open01)))  # motor q: 0 closed .. 5.4 open
        g = MotorCmds_(cmds=[MotorCmd_(mode=1, q=q, dq=0.0, tau=0.0, kp=5.0, kd=0.05, reserve=[0, 0, 0])])
        self.pub[side].Write(g)
        return f"{side} gripper {'open' if open01 > 0.5 else 'closed'} (q={q:.2f})"


# lift: cmd_hispeed P-loop toward a height target, feedback rt/hispeed_state (y = m)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_  # noqa: E402


class Lift:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        self.state = {"h": None}
        self.target = None
        ChannelSubscriber("rt/hispeed_state", Point32_).Init(
            lambda m: self.state.__setitem__("h", float(m.y)), 1)
        self.pub = ChannelPublisher("rt/cmd_hispeed", Point32_)
        self.pub.Init()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            h = self.state.get("h")
            if self.target is not None and h is not None:
                p = Point32_(x=0.0, y=0.0, z=max(-0.05, min(0.05, 1.5 * (self.target - h))))
                self.pub.Write(p)
            time.sleep(0.05)

    def set(self, height_m):
        if height_m is None:
            self.target = None
            return "lift control released"
        self.target = float(min(0.42, max(0.0, height_m)))
        t0 = time.time()
        while time.time() - t0 < 8.0:
            h = self.state.get("h")
            if h is not None and abs(h - self.target) < 0.01:
                break
            time.sleep(0.2)
        return f"lift at {self.state.get('h'):.3f} m (target {self.target:.3f}, 0..0.42)"


GRIP = None
LIFT = None


def op_state():
    out = {"frame": FRAME_HELP}
    for side in ("left", "right"):
        q = _arm_q(side)
        if q is None:
            out[side] = "no lowstate"
            continue
        p, _ = KIN[side].fk(q)
        out[side] = {
            "tcp_xyz": [round(float(v), 3) for v in p],
            "joints_rad": [round(v, 3) for v in q],
        }
    out["lift_m"] = round(LIFT.state.get("h") or 0.0, 3)
    return json.dumps(out, ensure_ascii=False)


# ---- MCP tools ----------------------------------------------------------------
TOOLS = [
    {"name": "get_views", "description": "The 5 camera views: main (close over-the-shoulder of the "
                                         "table workspace - judge the global scene from it), "
                                         "head_left, head_right, wrist_left, wrist_right.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_state", "description": f"Arm TCP poses + joints ({FRAME_HELP}), lift height. "
                                         f"Use the tcp_xyz values to plan set_ee_pose targets.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "set_ee_pose", "description": f"PRIMARY motion tool: send one gripper to an ABSOLUTE "
        f"pose ({FRAME_HELP}). The arm motion is planned and collision-checked on this side "
        f"(straight TCP path, waypoint safety screen - no torso / below-table / behind-body, "
        f"bounded joint speed) - BIG steps of 0.1-0.3 m in ONE command are safe and encouraged. "
        f"Optional roll/pitch/yaw (radians, absolute, omitted axes keep the current orientation) "
        f"reorient the wrist - e.g. pitch about +3.1 flips the gripper to point straight down "
        f"(invert a held cup). Plan targets from get_state's tcp_xyz.",
     "inputSchema": {"type": "object", "required": ["side", "x", "y", "z"],
                     "properties": {"side": {"type": "string", "enum": ["left", "right"]},
                                    "x": {"type": "number"}, "y": {"type": "number"},
                                    "z": {"type": "number"},
                                    "roll": {"type": "number"}, "pitch": {"type": "number"},
                                    "yaw": {"type": "number"},
                                    "speed": {"type": "number"}}}},
    {"name": "move_ee", "description": f"Fine adjustment: move one gripper's tool point by a small "
                                       f"RELATIVE offset (<= 5 cm, {FRAME_HELP}), keeping the "
                                       f"orientation. Same collision screen as set_ee_pose; for "
                                       f"larger moves prefer set_ee_pose (one planned command).",
     "inputSchema": {"type": "object", "required": ["side", "dx", "dy", "dz"],
                     "properties": {"side": {"type": "string", "enum": ["left", "right"]},
                                    "dx": {"type": "number"}, "dy": {"type": "number"},
                                    "dz": {"type": "number"},
                                    "speed": {"type": "number"}}}},
    {"name": "set_gripper", "description": "open01: 1 = fully open, 0 = closed.",
     "inputSchema": {"type": "object", "required": ["side", "open"],
                     "properties": {"side": {"type": "string", "enum": ["left", "right"]},
                                    "open": {"type": "number"}}}},
    {"name": "set_lift", "description": "Column height target in metres (0 .. 0.42). Lift to reach "
                                        "higher cups; the base itself never moves.",
     "inputSchema": {"type": "object", "required": ["height_m"],
                     "properties": {"height_m": {"type": "number"}}}},
    {"name": "home", "description": "Both arms back to the default pose, grippers open.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "task_done", "description": "Declare the stacking task finished.",
     "inputSchema": {"type": "object", "required": ["success"],
                     "properties": {"success": {"type": "boolean"},
                                    "note": {"type": "string"}}}},
]
DONE = {"done": False}


def call_tool(name, args):
    if name == "get_views":
        import base64
        content = []
        for v in VIEWS:
            jp = _frames.get(v)
            if jp:
                # Anthropic-format image block: the driver passes these through
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(jp).decode()}})
                content.append({"type": "text", "text": f"^ {v}"})
        missing = [v for v in VIEWS if not _frames.get(v)]
        if missing:
            content.append({"type": "text", "text": f"missing views: {missing}"})
        return {"content": content or [{"type": "text", "text": "no frames yet"}]}
    if name == "get_state":
        return {"content": [{"type": "text", "text": op_state()}]}
    if name == "set_ee_pose":
        return {"content": [{"type": "text", "text": op_set_ee_pose(
            args["side"], args["x"], args["y"], args["z"],
            args.get("roll"), args.get("pitch"), args.get("yaw"),
            args.get("speed", 0.5))}]}
    if name == "move_ee":
        return {"content": [{"type": "text", "text": op_move_ee(
            args["side"], args["dx"], args["dy"], args["dz"], args.get("speed", 0.4))}]}
    if name == "move_joints":
        # not in TOOLS (the model plans in gripper-pose space now), but
        # the MCP endpoint also serves the ACT runner, which legitimately
        # streams joint targets (its own layer clamps to joint limits)
        _stream_joints(args["side"], [float(x) for x in args["q"]])
        return {"content": [{"type": "text", "text": op_state()}]}
    if name == "set_gripper":
        return {"content": [{"type": "text", "text": GRIP.set(args["side"], args["open"])}]}
    if name == "set_lift":
        return {"content": [{"type": "text", "text": LIFT.set(args.get("height_m"))}]}
    if name == "home":
        for side in ("left", "right"):
            _stream_joints(side, _home_q(side))
            GRIP.set(side, 1.0)
        return {"content": [{"type": "text", "text": "at home (arms default, grippers open)"}]}
    if name == "task_done":
        DONE["done"] = True
        DONE["success"] = bool(args.get("success"))
        return {"content": [{"type": "text", "text": f"recorded: success={DONE['success']}"}]}
    return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        mid, method = req.get("id"), req.get("method", "")
        if method == "initialize":
            body = {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "t20-arm", "version": "1.0"}}}
        elif method.startswith("notifications/"):
            self.send_response(202)
            self.end_headers()
            return
        elif method == "tools/list":
            body = {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            p = req.get("params", {})
            body = {"jsonrpc": "2.0", "id": mid, "result": call_tool(p.get("name"), p.get("arguments") or {})}
        else:
            body = {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"no such method {method}"}}
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    global BRIDGE, GRIP, LIFT
    BRIDGE = Bridge(domain=1)
    # the launcher starts this server the moment the camera streams answer,
    # which can be BEFORE the sim's DDS provider publishes rt/lowstate -
    # wait patiently instead of dying in that window
    for _ in range(120):
        try:
            BRIDGE.wait_for_sim(timeout=2.0)
            break
        except SystemExit:
            time.sleep(1.0)
    else:
        raise SystemExit("no rt/lowstate after 4 min - is the sim running?")
    GRIP = Grippers()
    LIFT = Lift()
    # grippers OPEN from the start: the demos (and the ACT pre-pose) begin
    # with open grippers, and any earlier close command lingering on the
    # wire must not survive (motor q: 0 closed .. 5.4 open)
    for side in ("left", "right"):
        GRIP.set(side, 5.4 / 5.4)
    threading.Thread(target=_cam_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", MCP_PORT), Handler)
    print(f"[t20] MCP endpoint http://0.0.0.0:{MCP_PORT}/mcp - tools: "
          + ", ".join(t["name"] for t in TOOLS), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
