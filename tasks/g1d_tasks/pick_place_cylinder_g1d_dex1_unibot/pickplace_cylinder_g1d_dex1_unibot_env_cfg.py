# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""G1-D + Dex1 pick-place in the UNIBOT demo's warehouse hall.

Everything (actions / observations / events / DDS) is inherited from
the reference cylinder task; only the SCENE is swapped for the
unibot warehouse (tasks/common_scene/unibot_warehouse_scene.py) and
the TASK OBJECT comes from the challenge layout config
(unibot_task_layout.yaml: `pick:` selects a table + item - by
default the T01 mug). The success bound is RELATIVE to the object's
spawn pose (the reference one is absolute-scene and reset-loops
outside its 6-table layout).
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from tasks.common_scene.unibot_warehouse_scene import PICK_OBJECT, UnibotWarehouseSceneCfg
from tasks.g1d_tasks.pick_place_cylinder_g1d_dex1.pickplace_cylinder_g1d_dex1_joint_env_cfg import (
    PickPlaceG1DDEX1JointEnvCfg)

from . import mdp as unibot_mdp

# robot parks at the slot standoff (SLOT_STANDOFF 0.55 south of the
# slot), facing +y like the reference task.
# UNIBOT_SPAWN="x,y" (metres) teleports the START pose to any station -
# e.g. run_T20.sh boots directly at the table_03 paper-cup-stacking
# station ("T20 纸杯堆叠") with ALL navigation disabled: the robot
# spawns already parked at the table and never moves the base.
ROBOT_POS = (-8.5, 0.35, 0.1114)
try:  # env override wins when set (sim_main passes the environment through)
    import os as _os
    _sp = _os.environ.get("UNIBOT_SPAWN", "").split(",")
    if len(_sp) == 2:
        ROBOT_POS = (float(_sp[0]), float(_sp[1]), ROBOT_POS[2])
except Exception:
    pass

_ROBOT_ENTITIES = (
    "robot", "front_camera", "front_right_camera", "left_wrist_camera", "right_wrist_camera")


@configclass
class UnibotTerminationsCfg:
    # object must stay near its tabletop spawn (see mdp.py: the
    # reference reset_object_estimate is ABSOLUTE-scene and would
    # reset-loop at our x -8.5 tables)
    success = DoneTerm(func=unibot_mdp.object_stays_near_default)


@configclass
class PickPlaceG1DDex1UnibotEnvCfg(PickPlaceG1DDEX1JointEnvCfg):
    """The reference G1-D Dex1 task inside our warehouse."""

    terminations = UnibotTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        old = self.scene
        wh = UnibotWarehouseSceneCfg(
            num_envs=old.num_envs, env_spacing=old.env_spacing, replicate_physics=True)
        for name in _ROBOT_ENTITIES:
            setattr(wh, name, getattr(old, name))
        init = wh.robot.init_state
        wh.robot = wh.robot.replace(init_state=init.replace(pos=ROBOT_POS))
        # the env "object" = the challenge layout's pick item (the T01
        # mug by default). The layout loader EXCLUDED it from the
        # scene fields, so this is the single instance of it.
        wh.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=list(PICK_OBJECT["pos"]), rot=list(PICK_OBJECT["rot"])),
            spawn=PICK_OBJECT["spawn"],
        )
        self.scene = wh
