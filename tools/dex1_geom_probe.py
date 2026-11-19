"""Measure the REAL Dex1 finger geometry (no guessing): STL mesh bboxes +
live world poses of the finger links -> exact jaw centre and clearance."""
import os
import struct
import sys

REPO = "/home/code/unitree_sim_isaaclab"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("PROJECT_ROOT", REPO)

from isaaclab.app import add_launcher_args, launch_simulation
import argparse

p = argparse.ArgumentParser()
p.add_argument("--task_name", default="Isaac-PickPlace-Cylinder-G1D-Dex1-Joint")
add_launcher_args(p)
a = p.parse_args()

import tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

cfg = parse_env_cfg(a.task_name, device=a.device, num_envs=1)
cfg.scene.world_camera = None
cfg.episode_length_s = 1.0e6


def stl_bbox(path):
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        mn = [1e9] * 3
        mx = [-1e9] * 3
        for _ in range(n):
            f.read(12)
            for _v in range(3):
                for i, v in enumerate(struct.unpack("<3f", f.read(12))):
                    mn[i] = min(mn[i], v)
                    mx[i] = max(mx[i], v)
            f.read(2)
    return mn, mx


with launch_simulation(cfg, a):
    import gymnasium as gym
    import tasks.common_observations.camera_state as cs

    cs._camera_cache["write_interval_steps"] = 10 ** 9
    env = gym.make(a.task_name, cfg=cfg).unwrapped
    env.reset()

    import functools
    _print = print
    print = functools.partial(_print, flush=True)
    robot = env.scene["robot"]
    T = lambda x: x.torch if hasattr(x, "torch") else x
    names = list(robot.data.body_names)
    bodies = list(robot.data.body_names)

    for side in ("left", "right"):
        print(f"===== {side} =====")
        for link in ("dex1_base_link", "dex1_finger_link_1", "dex1_finger_link_2"):
            i = bodies.index(f"{side}_{link}")
            p_ = T(robot.data.body_pos_w)[0][i]
            q_ = T(robot.data.body_quat_w)[0][i]
            print(f"{link:22s} wpos ({float(p_[0]):+.3f},{float(p_[1]):+.3f},{float(p_[2]):+.3f}) "
                  f"quat xyzw ({float(q_[0]):+.3f},{float(q_[1]):+.3f},{float(q_[2]):+.3f},{float(q_[3]):+.3f})")

    mesh_dir = f"{REPO}/assets/urdf/g1_d_description/meshes/meshes"
    for name in ("Dex1_finger_link_1.STL", "Dex1_finger_link_2.STL", "Dex1_base_link.STL"):
        mn, mx = stl_bbox(f"{mesh_dir}/{name}")
        print(f"mesh {name:26s} bbox min ({mn[0]:+.3f},{mn[1]:+.3f},{mn[2]:+.3f}) "
              f"max ({mx[0]:+.3f},{mx[1]:+.3f},{mx[2]:+.3f})")

    obj = env.scene["object"]
    po = T(obj.data.root_pos_w)[0]
    print(f"cylinder wpos ({float(po[0]):+.3f},{float(po[1]):+.3f},{float(po[2]):+.3f})")
    os._exit(0)
