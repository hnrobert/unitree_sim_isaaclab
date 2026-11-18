# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Unitree G1-D (wheeled base + lift + G1 upper body) task module."""

import importlib

_TASKS = (
    "pick_place_cylinder_g1d_dex1",
    "pick_place_cylinder_g1d_dex1_unibot",
    "pick_place_cylinder_g1d_dex3",
    "pick_place_redblock_g1d_dex1",
    "pick_place_redblock_g1d_dex3",
    "pick_redblock_into_drawer_g1d_dex1",
    "pick_redblock_into_drawer_g1d_dex3",
    "stack_rgyblock_g1d_dex1",
    "stack_rgyblock_g1d_dex3",
)

# robot variants whose asset is not published in the fetch_assets dataset yet register
# a None cfg; their env cfgs raise FileNotFoundError at import - skip those tasks.
__all__ = []
for _task in _TASKS:
    try:
        globals()[_task] = importlib.import_module(f"{__name__}.{_task}")
        __all__.append(_task)
    except FileNotFoundError as _e:
        print(f"[tasks.g1d_tasks] skipped {_task} (robot asset missing): {_e}")
