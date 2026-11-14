# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""G1-D + Dex1 pick-place in the UNIBOT demo's warehouse hall.

Everything (actions / observations / events / terminations / DDS)
is inherited from the reference cylinder task; only the SCENE is
swapped for the unibot warehouse (tasks/common_scene/
unibot_warehouse_scene.py): the robot works the corridor-side slot
of the first divider table (x -8.5, +y row).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from tasks.common_scene.unibot_warehouse_scene import UnibotWarehouseSceneCfg
from tasks.g1d_tasks.pick_place_cylinder_g1d_dex1.pickplace_cylinder_g1d_dex1_joint_env_cfg import (
    PickPlaceG1DDEX1JointEnvCfg)

# robot parks at the slot standoff (SLOT_STANDOFF 0.55 south of the
# slot), facing +y like the reference task; the cylinder stands on
# the table's corridor-facing slot (TABLE_TOP 0.78 + half height)
ROBOT_POS = (-8.5, 0.35, 0.1114)
OBJECT_POS = (-8.5, 0.9, 0.955)

_ROBOT_ENTITIES = (
    "robot", "front_camera", "front_right_camera", "left_wrist_camera", "right_wrist_camera")


@configclass
class PickPlaceG1DDex1UnibotEnvCfg(PickPlaceG1DDEX1JointEnvCfg):
    """The reference G1-D Dex1 cylinder task inside our warehouse."""

    def __post_init__(self):
        super().__post_init__()
        old = self.scene
        wh = UnibotWarehouseSceneCfg(
            num_envs=old.num_envs, env_spacing=old.env_spacing, replicate_physics=True)
        for name in _ROBOT_ENTITIES:
            setattr(wh, name, getattr(old, name))
        init = wh.robot.init_state
        wh.robot = wh.robot.replace(init_state=init.replace(pos=ROBOT_POS))
        wh.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=list(OBJECT_POS), rot=[0, 0, 0, 1]),
            spawn=sim_utils.CylinderCfg(
                radius=0.018,
                height=0.35,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    restitution_combine_mode="min",
                    static_friction=1.5,
                    dynamic_friction=1.5,
                    restitution=0.0,
                ),
            ),
        )
        self.scene = wh
