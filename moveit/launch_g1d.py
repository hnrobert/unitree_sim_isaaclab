# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Launch the classic MoveIt2 move_group stack for the G1-D (plan only).

robot_state_publisher (URDF + static /joint_states defaults) +
move_group (OMPL planning, no controllers - plans are pulled via
the MoveGroup action by plan_pick.py). Parameters are injected as
strings directly from ../config, no xacro needed.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config")


def _read(name):
    with open(os.path.join(CONFIG, name)) as f:
        return f.read()


def generate_launch_description():
    urdf = _read("g1d_moveit.urdf")
    srdf = _read("g1d.srdf")
    with open(os.path.join(CONFIG, "kinematics.yaml")) as f:
        import yaml
        kin = yaml.safe_load(f)
    with open(os.path.join(CONFIG, "ompl_planning.yaml")) as f:
        ompl = yaml.safe_load(f)

    common = {
        "robot_description": urdf,
        "robot_description_semantic": srdf,
        "robot_description_kinematics": kin,
        # NOTE no controller manager at all: planning only, trajectories
        # leave via ZMQ. (An EMPTY list parameter - controller_names: []
        # - becomes a tuple in launch_ros parameter evaluation and is
        # rejected with "Expected 'value' to be one of ...".)
    }

    # move_group's /compute_ik & /compute_fk need the TF tree, which
    # robot_state_publisher only broadcasts WHEN /joint_states flows -
    # without a publisher the IK/FK services fail with NO_IK_SOLUTION
    jsp = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{"robot_description": urdf, "rate": 30}],
        output="screen",
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": urdf}],
        output="screen",
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        parameters=[common, ompl],
        output="screen",
    )

    return LaunchDescription([jsp, rsp, move_group])


if __name__ == "__main__":
    from launch import LaunchService
    ls = LaunchService()
    ls.include_launch_description(generate_launch_description())
    ls.run()
