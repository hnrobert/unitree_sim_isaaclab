# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Viewport camera that follows the robot (chase view).

The viewport is NEVER bound to a scene prim: in Isaac Sim 6.2 a
looked-through scene camera ignores viewport navigation entirely
(mouse drags produce no camera writes at all - the "fixed main
view" reports). The kit's native FREE camera /OmniverseKit_Persp
(session layer) is the one camera the viewport is designed to
edit, so the chase view drives THAT prim: same framing behaviour,
and the user's drag lands directly on it - we detect the foreign
transform and stop following from then on.
"""

from __future__ import annotations

import math

FOLLOW_CAMERA_PATH = "/OmniverseKit_Persp"


class FollowCamera:
    def __init__(self, env, asset_name: str = "robot", prim_path: str = FOLLOW_CAMERA_PATH,
                 distance: float = 2.2, height: float = 1.6, look_height: float = 0.9, look_ahead: float = 0.4,
                 focal_length: float = 16.5, yaw_smoothing: float = 0.15):
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        self._env = env
        self._robot = env.scene[asset_name]
        self._distance, self._height = distance, height
        self._look_height, self._look_ahead = look_height, look_ahead
        self._alpha = yaw_smoothing
        self._yaw = None
        self._last = None
        stage = omni.usd.get_context().get_stage()
        self._stage = stage
        self._session = stage.GetSessionLayer()
        # the free camera belongs to the SESSION layer - the viewport
        # creates it lazily, so define it early-bind-safe (no-op if
        # it exists) and pin the framing to the sensor-camera lens
        with Usd.EditContext(stage, self._session):
            cam = UsdGeom.Camera.Define(stage, prim_path)
            cam.CreateFocalLengthAttr(focal_length)
            cam.CreateHorizontalApertureAttr(20.955)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1.0e5))
            self._op = UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform()
        self._xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
        self.path = prim_path
        self._written = None  # local transform WE last wrote (user edits differ)
        self._paused = False
        self.update(force=True)

    def _root_pose(self):
        d = self._robot.data
        pos = d.root_pos_w.torch[0] if hasattr(d.root_pos_w, "torch") else d.root_pos_w[0]
        quat = d.root_quat_w.torch[0] if hasattr(d.root_quat_w, "torch") else d.root_quat_w[0]
        x, y, z, w = (float(v) for v in quat)  # xyzw
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(pos[0]), float(pos[1]), yaw

    def update(self, force: bool = False) -> None:
        from pxr import Gf, Usd

        # The viewport camera fight: orbiting/panning in the viewport
        # edits THIS camera's transform (it is the kit free camera), and
        # the next update() would overwrite it right back (the "camera
        # snaps back to center" behavior). Detect the external edit -
        # the composed LOCAL transform no longer equals the one we
        # wrote, whoever wrote which ops - and stop following from
        # then on (until the simulation restarts).
        current = self._xf.GetLocalTransformation()
        if not self._paused and self._written is not None:
            diff = max(abs(a - b) for ra, rb in zip(current, self._written) for a, b in zip(ra, rb))
            if diff > 1e-6:
                self._paused = True
                print("[camera] follow view paused - viewport camera moved by the user")
        if self._paused:
            return

        px, py, yaw = self._root_pose()
        if self._yaw is None or force:
            self._yaw = yaw
        else:  # low-pass the heading (shortest angle)
            self._yaw += self._alpha * math.atan2(math.sin(yaw - self._yaw), math.cos(yaw - self._yaw))
        key = (round(px, 3), round(py, 3), round(self._yaw, 3))
        if key == self._last and not force:
            return  # robot did not move: no USD write
        self._last = key
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        eye = Gf.Vec3d(px - self._distance * c, py - self._distance * s, self._height)
        target = Gf.Vec3d(px + self._look_ahead * c, py + self._look_ahead * s, self._look_height)
        view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))  # world -> camera (USD / OpenGL camera)
        cam_to_world = view.GetInverse()
        with Usd.EditContext(self._stage, self._session):  # free camera = session layer
            self._op.Set(cam_to_world)
        self._written = self._xf.GetLocalTransformation()
