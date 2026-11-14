# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""G1-D + Dex1 pick-place in the unibot warehouse hall."""

import gymnasium as gym

from . import pickplace_cylinder_g1d_dex1_unibot_env_cfg

gym.register(
    id="Isaac-PickPlace-Cylinder-G1D-Dex1-Unibot-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_cylinder_g1d_dex1_unibot_env_cfg.PickPlaceG1DDex1UnibotEnvCfg,
    },
    disable_env_checker=True,
)
