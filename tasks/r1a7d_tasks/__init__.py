# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Unitree R1-A7-D (wheeled base + lift + R1 upper body) task module."""

import importlib

_TASKS = (
    "pick_place_cylinder_r1a7d_dex1",
    "pick_place_cylinder_r1a7d_dex2_5",
    "pick_place_redblock_r1a7d_dex1",
    "pick_place_redblock_r1a7d_dex2_5",
    "pick_redblock_into_drawer_r1a7d_dex1",
    "pick_redblock_into_drawer_r1a7d_dex2_5",
    "stack_rgyblock_r1a7d_dex1",
    "stack_rgyblock_r1a7d_dex2_5",
)

# robot variants whose asset is not published in the fetch_assets dataset yet register
# a None cfg; their env cfgs raise FileNotFoundError at import - skip those tasks.
__all__ = []
for _task in _TASKS:
    try:
        globals()[_task] = importlib.import_module(f"{__name__}.{_task}")
        __all__.append(_task)
    except FileNotFoundError as _e:
        print(f"[tasks.r1a7d_tasks] skipped {_task} (robot asset missing): {_e}")
