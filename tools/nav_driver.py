# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Base navigation for the G1-D in the official sim - OUR demo's
phased control logic, on the sim's native DDS channels.

Channels (same as the keyboard teleop / real robot):
  rt/cmd_vel_no_limit   geometry_msgs/Twist   (linear.x = v, angular.z = w)
  rt/slamware_ros_sdk_server_node/odom  nav_msgs/Odometry  (world pose)

Control logic ported from the demo's g1d_driver.py: face the target
first, then creep forward with proportional heading + lateral
homing (capped), then spin-in-place to the final heading, then
settle still and verify. Tolerances: pos 0.05 m, yaw 0.08 rad.

Stations: directly in front of each work/dressing table's CENTER,
facing the table squarely (the corridor-side face is 1.25 m from
the aisle; the AGV parks at +/-0.90 m so its nose clears the table
face and the near items sit inside the 0.55 m arm envelope).

Usage (inside the sim container):
  python tools/nav_driver.py --probe            # read odom, 1 s creep
  python tools/nav_driver.py --table 1          # station at table_01
  python tools/nav_driver.py --pose -8.5 0.9 90 # x, y, yaw_deg
  python tools/nav_driver.py --stations         # list stations
"""

import argparse
import math
import time

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_, Vector3_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_

# ---- layout constants (mirror unibot_warehouse_scene.py) ---------------
DIVIDER_X = (-8.5, -11.0, -13.75, -15.25, -18.0, -19.5)
ROW_DY = 1.7
TABLE_POS = tuple((float(x), float(dy)) for x in DIVIDER_X for dy in (-ROW_DY, ROW_DY))
DRESSING_TABLES = ((6.0, 3.0), (10.0, 3.2), (6.0, -3.0), (12.0, -3.4))

STATION_STANDOFF = 0.90      # |y| of the parking spot
# TRANSIT LANE: with the tables rotated (long edge to the corridor),
# the north/south LEG lines sit at |y| = 0.89..1.25 - driving across
# at the station height rams the legs (measured stall at x -9.98).
# Stay at |y| <= 0.3 while crossing, pull in only at the station x.
LANE_Y = 0.30

# odom is RELATIVE to the robot's spawn pose (the mobile DDS reports
# home-rotated dx/dy); the spawn is our task layout's ROBOT_POS
HOME_X, HOME_Y, HOME_YAW = -8.5, 0.35, 0.0

# control constants (from the demo's driver)
V_CREEP = 0.15
V_TRANSIT = 0.50  # the sim slews the base, so command high - actual ~0.1-0.2
W_TURN = 0.6
K_LAT = 0.8                  # lateral homing gain
K_HEAD = 1.2                 # heading P gain while creeping
POS_TOL = 0.05
YAW_TOL = 0.08
LATERAL_DEAD = 0.05


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def stations():
    """table_idx -> (x, y, yaw_deg): centered, square to the table."""
    out = {}
    for i, (tx, ty) in enumerate(TABLE_POS):
        out[i] = (tx, math.copysign(STATION_STANDOFF, ty), 90.0 if ty > 0 else -90.0)
    for j, (tx, ty) in enumerate(DRESSING_TABLES):
        out[len(TABLE_POS) + j] = (
            tx, ty + (-STATION_STANDOFF if ty > 0 else STATION_STANDOFF),
            90.0 if ty > 0 else -90.0)
    return out


class NavDriver:
    def __init__(self, domain=1):
        ChannelFactoryInitialize(domain)
        self.cmd = ChannelPublisher("rt/cmd_vel_no_limit", Twist_)
        self.cmd.Init()
        self.odom = None
        ChannelSubscriber("rt/slamware_ros_sdk_server_node/odom", Odometry_).Init(
            lambda m: setattr(self, "odom", m), 1)
        self.reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
        self.reset_pub.Init()
        self.zero()

    def reset_pose(self, category="2", wait=8.0):
        """Full scene reset (rt/reset_pose/cmd: '1' object, '2' all) -
        the velocity-driven base LOCKS UP irreversibly on contact;
        resetting is the only reliable unlock."""
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_ as _S
        m = _S(data=category)
        self.reset_pub.Write(m)
        time.sleep(wait)

    @staticmethod
    def _twist(v, w):
        return Twist_(linear=Vector3_(x=float(v), y=0.0, z=0.0),
                      angular=Vector3_(x=0.0, y=0.0, z=float(w)))

    def zero(self):
        self.cmd.Write(self._twist(0.0, 0.0))

    def send(self, v, w):
        self.cmd.Write(self._twist(v, w))

    # ---- feedback -------------------------------------------------------
    def wait_odom(self, timeout=10.0):
        t0 = time.time()
        while self.odom is None and time.time() - t0 < timeout:
            time.sleep(0.05)
        if self.odom is None:
            raise SystemExit("no odom on rt/slamware_ros_sdk_server_node/odom - sim running?")
        return True

    def pose(self):
        """WORLD pose (odom is spawn-relative; yaw 0 = world +x)."""
        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(HOME_YAW), math.sin(HOME_YAW)
        wx = HOME_X + c * p.x - s * p.y
        wy = HOME_Y + s * p.x + c * p.y
        return wx, wy, wrap(yaw + HOME_YAW)

    # ---- demo control logic ---------------------------------------------
    def spin_to(self, yaw, timeout=12.0, tol=YAW_TOL):
        t0 = time.time()
        while time.time() - t0 < timeout:
            _, _, cur = self.pose()
            err = wrap(yaw - cur)  # demo convention: err = desired - yaw
            if abs(err) < tol:
                self.zero()
                return True
            self.send(0.0, max(0.08, min(W_TURN, 1.5 * abs(err))) * (1 if err > 0 else -1))
            time.sleep(0.05)
        self.zero()
        return False

    def forward_to(self, x, y, final_yaw, timeout=240.0):
        """Creep to (x, y): face the target, proportional heading +
        lateral homing (the demo's forward_line), no-overshoot legs."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            cx, cy, cyaw = self.pose()
            dx, dy = x - cx, y - cy
            dist = math.hypot(dx, dy)
            if dist < POS_TOL:
                self.zero()
                return True
            bearing = math.atan2(dy, dx)
            head_err = wrap(bearing - cyaw)
            # spin to face the target when far off (saves nose-scraping)
            if abs(head_err) > 0.6 and dist > 0.4:
                self.spin_to(bearing)
                continue
            v = V_CREEP if dist < 0.5 else V_TRANSIT
            # heading correction + lateral homing, capped like the demo
            w = K_HEAD * head_err
            w = max(-0.5, min(0.5, w))
            self.send(v, w)
            time.sleep(0.05)
        self.zero()
        return False

    def leave_station(self, lane_y, timeout=30.0):
        """REVERSE straight out of a station to the transit lane.

        Inside a station (|y| > ~0.6) the robot sits IN the table-leg
        band - any spin or diagonal scrape-locks on the legs (the
        measured stalls). Backing up along the current heading keeps
        every corner moving straight away from the legs."""
        t0 = time.time()
        target_sign = 1.0 if lane_y > 0 else -1.0
        while time.time() - t0 < timeout:
            cx, cy, cyaw = self.pose()
            if (cy - lane_y) * target_sign >= -POS_TOL and abs(lane_y - cy) < 0.12:
                self.zero()
                return True
            v = -0.15 * target_sign  # reverse toward the lane
            # hold the station heading while reversing - an angled
            # reverse drifts sideways INTO the leg band
            face = math.radians(90.0) * (1 if cy > 0 else -1)
            w = max(-0.3, min(0.3, 0.8 * wrap(face - cyaw)))
            self.send(v, w)
            time.sleep(0.05)
        self.zero()
        return False

    def drive_to(self, x, y, yaw_deg, via_lane=None):
        """Phased (demo logic + leg-band awareness): reverse out of a
        station -> lane transit along the corridor -> pull in at the
        station x -> square up -> settle."""
        if via_lane is not None:
            lane_y = LANE_Y * (1 if y > 0 else -1)
            # 1) reverse out of a station straight (never spin inside
            #    the leg band - contacts lock the velocity-driven base)
            if abs(self.pose()[1]) > 0.6:
                self.leave_station(lane_y)
            # 2) transit ALONG the open lane to the station x
            lx = x if via_lane == "x" else via_lane
            if not self.forward_to(lx, lane_y, yaw_deg, timeout=300.0):
                print(f"  transit leg missed lane ({self.pose()[0]:.2f},{self.pose()[1]:.2f})")
            # 3) square up to the table IN THE LANE (open space), then
            #    4) pull straight in - NO spinning inside the station
            self.spin_to(math.radians(yaw_deg))
            ok = self.forward_to(x, y, yaw_deg, timeout=60.0)
        else:
            ok = self.forward_to(x, y, yaw_deg)
            ok &= self.spin_to(math.radians(yaw_deg))
        # settle still (the demo braces before arm work)
        self.zero()
        time.sleep(0.4)
        self.zero()
        cx, cy, cyaw = self.pose()
        ex, ey = cx - x, cy - y
        print(f"  station ({cx:.3f},{cy:.3f},{math.degrees(cyaw):.1f}deg) "
              f"err=({ex:+.3f},{ey:+.3f}) yaw_err={math.degrees(wrap(math.radians(yaw_deg) - cyaw)):+.1f}")
        return ok and abs(ex) < 0.1 and abs(ey) < 0.1


def patrol_route(which="all"):
    """Sensible sweep: north row west-bound, south row east-bound,
    east dressing zone last; reverse for the way back."""
    north = [1, 3, 5, 7, 9, 11]        # x -8.5 .. -19.5
    south = [10, 8, 6, 4, 2, 0]        # x -19.5 .. -8.5
    dressing = [12, 13, 14, 15]        # east zone x 6..12
    if which == "work":
        return north + south
    if which == "dressing":
        return dressing
    return north + south + dressing


class PatrolReport:
    def __init__(self):
        self.rows = []

    def add(self, tag, x, y, yd, ex, ey, yerr, result, secs):
        self.rows.append((tag, x, y, yd, ex, ey, yerr, result, secs))

    def dump(self):
        print("\n" + "=" * 78)
        print(f"{'station':16s} {'target':>18s} {'err_xy(m)':>12s} {'yaw_err':>8s} {'result':>7s} {'time':>6s}")
        print("-" * 78)
        n_ok = 0
        for tag, x, y, yd, ex, ey, yerr, result, secs in self.rows:
            print(f"{tag:16s} ({x:+7.2f},{y:+6.2f},{yd:+3.0f}) "
                  f"({ex:+.2f},{ey:+.2f})  {yerr:+5.1f}deg  {result:>7s} {secs:5.1f}s")
            n_ok += result == "PASS"
        print("-" * 78)
        print(f"{n_ok}/{len(self.rows)} stations PASS")


def patrol(d, which="all", rounds=1):
    """Visit every station in route order, then reverse - the way
    back. Per station: arrival verify (0.10 m / 8 deg), one retry
    with a back-off on stall/miss."""
    route = patrol_route(which)
    full = route + list(reversed(route)) if rounds >= 1 else route
    rep = PatrolReport()
    d.wait_odom()
    # start from a CLEAN scene: a previously contact-locked base
    # would fail every leg (the lock is irreversible without reset)
    d.reset_pose()
    print(f"patrol: {len(full)} station stops, route={which}", flush=True)
    for si, ti in enumerate(full):
        st = stations()[ti]
        tag = f"[{si+1}/{len(full)}] table_{ti:02d}"
        x, y, yd = st
        t0 = time.time()
        result = "FAIL"
        for attempt in (1, 2):
            ok = d.drive_to(x, y, yd, via_lane="x")
            cx, cy, cyaw = d.pose()
            ex, ey = cx - x, cy - y
            yerr = math.degrees(wrap(math.radians(yd) - cyaw))
            good = ok and abs(ex) < 0.10 and abs(ey) < 0.10 and abs(yerr) < 8.0
            if good:
                result = "PASS"
                if attempt == 2:
                    result = "PASS*"
                break
            # stalled or missed: back off straight and retry once;
            # a second miss means the base is contact-locked -> full
            # scene reset (the only reliable unlock), mark RESET, move on
            print(f"  {tag} attempt {attempt} missed "
                  f"({cx:+.2f},{cy:+.2f},{math.degrees(cyaw):+.0f}) - back off", flush=True)
            if attempt == 2:
                print(f"  {tag} contact-locked - resetting scene", flush=True)
                d.reset_pose()
                result = "RESET"
                break
            t1 = time.time()
            while time.time() - t1 < 1.5:
                d.send(-0.20, 0.0)
                time.sleep(0.05)
            d.zero()
            time.sleep(0.3)
        cx, cy, cyaw = d.pose()
        rep.add(tag, x, y, yd, cx - x, cy - y,
                math.degrees(wrap(math.radians(yd) - cyaw)), result, time.time() - t0)
        print(f"  {tag} -> {result}", flush=True)
    rep.dump()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patrol", type=int, nargs="?", const=1, default=None,
                    help="patrol all stations (arg = extra reverse rounds)")
    ap.add_argument("--route", choices=["all", "work", "dressing"], default="all")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--stations", action="store_true")
    ap.add_argument("--table", type=int)
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"))
    args = ap.parse_args()

    if args.stations:
        for i, (x, y, yd) in stations().items():
            tag = f"table_{i:02d}" if i < len(TABLE_POS) else f"dressing_{i - len(TABLE_POS)}"
            print(f"{i:2d} {tag:12s} -> ({x:+7.2f},{y:+6.2f}) yaw {yd:+.0f}")
        return

    if args.patrol is not None:
        d = NavDriver()
        d.wait_odom()
        patrol(d, which=args.route, rounds=args.patrol)
        return

    d = NavDriver()
    d.wait_odom()
    if args.probe:
        print(f"odom pose: ({d.pose()[0]:.3f},{d.pose()[1]:.3f},{math.degrees(d.pose()[2]):.1f}deg)")
        print("creep v=0.10 for 1.5 s (republished - cmd_vel TIMESTAMPS OUT) ...")
        t0 = time.time()
        while time.time() - t0 < 1.5:
            d.send(0.10, 0.0)
            time.sleep(0.05)
        d.zero()
        x0 = d.pose()
        print(f"after creep: ({x0[0]:.3f},{x0[1]:.3f},{math.degrees(x0[2]):.1f}deg)")
        return

    if args.table is not None:
        st = stations()[args.table]
        print(f"driving to table_{args.table:02d} station {st} (via lane y={LANE_Y})")
        ok = d.drive_to(st[0], st[1], st[2], via_lane="x")
        print("ARRIVED" if ok else "MISSED")
        return

    if args.pose:
        ok = d.drive_to(args.pose[0], args.pose[1], args.pose[2])
        print("ARRIVED" if ok else "MISSED")
        return

    ap.error("need --patrol / --probe / --stations / --table N / --pose X Y YAW")


if __name__ == "__main__":
    main()
