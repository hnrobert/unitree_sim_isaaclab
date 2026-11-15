# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Termination for the unibot task: the object stays near its table.

The reference task's success bound (reset_object_estimate) checks
ABSOLUTE scene coordinates of the original 6-table layout - with our
warehouse tables at x -8.5 the check fails every step and the env
reset-loops (the "cylinder jumping around" bug). This variant is
RELATIVE: the object must stay within `radius` of its own default
(spawn) pose and above `min_height` (i.e. still on/about the
tabletop); carrying it away or dropping it to the floor terminates
the episode, matching the original semantics.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_stays_near_default(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    radius: float = 0.4,
    min_height: float = 0.55,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    pos = object.data.root_pos_w.torch[:, :3]           # world xyz
    default = object.data.default_root_state[:, :3]      # spawn xyz (env frame)
    near = torch.linalg.norm(pos - default, dim=-1) < radius
    high = pos[:, 2] > min_height
    return ~(near & high)
