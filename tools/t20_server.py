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
VIEWS = ("head_left", "head_right", "wrist_left", "wrist_right")
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
    while True:
        try:
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


def op_move_ee(side, dx, dy, dz):
    q = _arm_q(side)
    if q is None:
        return "error: no rt/lowstate yet"
    p, R = KIN[side].fk(q)
    tgt = p + np.array([dx, dy, dz], dtype=float)
    q_new, ep, er = KIN[side].ik(tgt, R, q, q_rest=_home_q(side))
    if ep > 0.012:
        return (f"error: target ({tgt[0]:+.3f},{tgt[1]:+.3f},{tgt[2]:+.3f}) "
                f"unreachable (pos err {ep:.3f} m) - move closer / use the lift")
    _stream_joints(side, list(q_new))
    p2, _ = KIN[side].fk(_arm_q(side) or q_new)
    return (f"moved {side} tcp -> ({p2[0]:+.3f},{p2[1]:+.3f},{p2[2]:+.3f}) "
            f"[{FRAME_HELP}]; requested d=({dx:+.3f},{dy:+.3f},{dz:+.3f})")


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
    {"name": "get_views", "description": "The 4 challenge camera views: head_left, head_right, "
                                         "wrist_left, wrist_right (head stereo + wrists).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_state", "description": f"Arm TCP poses + joints ({FRAME_HELP}), lift height.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "move_ee", "description": f"Move one gripper's tool point by a RELATIVE offset "
                                       f"({FRAME_HELP}). Keeps the current orientation. ~5 cm steps "
                                       f"work well; errors if unreachable (adjust or use set_lift).",
     "inputSchema": {"type": "object", "required": ["side", "dx", "dy", "dz"],
                     "properties": {"side": {"type": "string", "enum": ["left", "right"]},
                                    "dx": {"type": "number"}, "dy": {"type": "number"},
                                    "dz": {"type": "number"}}}},
    {"name": "move_joints", "description": "Absolute 7-joint target (rad, shoulder_pitch..wrist_yaw order).",
     "inputSchema": {"type": "object", "required": ["side", "q"],
                     "properties": {"side": {"type": "string", "enum": ["left", "right"]},
                                    "q": {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7}}}},
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
    if name == "move_ee":
        return {"content": [{"type": "text", "text": op_move_ee(args["side"], args["dx"], args["dy"], args["dz"])}]}
    if name == "move_joints":
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
    BRIDGE.wait_for_sim()
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
