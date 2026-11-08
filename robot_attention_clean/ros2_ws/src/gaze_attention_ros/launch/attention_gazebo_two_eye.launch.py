"""
attention_gazebo_two_eye.launch.py — starts Gazebo Sim with the minimal
three-person attention scene (two_eye_demo.sdf), bridges the single
iris-yaw command topic into it (read by BOTH left_iris_joint and
right_iris_joint's JointPositionController plugins), and starts
eye_gaze_node. Does NOT start the webcam process (main_webcam.py) — get
manual angle commands working first:

    ros2 launch ros2_ws/src/gaze_attention_ros/launch/attention_gazebo_two_eye.launch.py
    # in another terminal, after sourcing /opt/ros/humble/setup.bash:
    ros2 topic pub --once /attention_target_angle std_msgs/msg/Float32 "{data: -0.65}"
    ros2 topic pub --once /attention_target_angle std_msgs/msg/Float32 "{data: 0.0}"
    ros2 topic pub --once /attention_target_angle std_msgs/msg/Float32 "{data: 0.65}"

Designed to run directly by path (no colcon build required), matching the
project's "don't block the demo on ROS tooling" constraint -- same pattern
as attention_gazebo.launch.py (the separate single-whole-head-yaw demo this
does not replace).

Args:
    world_path   default: multi_person/gazebo/worlds/two_eye_demo.sdf
    headless     default: false (pass true for -s, server-only, no GUI)
"""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# multi_person/ros2_ws/src/gaze_attention_ros/launch/this_file.py -> multi_person/gazebo/worlds/...
_PACKAGE_DIR = Path(__file__).resolve().parent.parent  # .../gaze_attention_ros/
_DEFAULT_WORLD = str(_PACKAGE_DIR.parent.parent.parent / "gazebo" / "worlds" / "two_eye_demo.sdf")

_NODE_ENV = dict(os.environ)
_NODE_ENV["PYTHONPATH"] = str(_PACKAGE_DIR) + os.pathsep + _NODE_ENV.get("PYTHONPATH", "")


def generate_launch_description() -> LaunchDescription:
    world_arg = DeclareLaunchArgument("world_path", default_value=_DEFAULT_WORLD)
    headless_arg = DeclareLaunchArgument("headless", default_value="false")

    gz_sim_gui = ExecuteProcess(
        cmd=["gz", "sim", "-r", LaunchConfiguration("world_path")],
        condition=UnlessCondition(LaunchConfiguration("headless")),
        output="screen",
    )
    gz_sim_headless = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", LaunchConfiguration("world_path")],
        condition=IfCondition(LaunchConfiguration("headless")),
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/gaze_attention/iris_yaw_cmd@std_msgs/msg/Float64]gz.msgs.Double"
        ],
        output="screen",
    )

    eye_gaze = ExecuteProcess(
        cmd=["python3", "-m", "gaze_attention_ros.eye_gaze_node"],
        additional_env=_NODE_ENV,
        output="screen",
    )

    return LaunchDescription([world_arg, headless_arg, gz_sim_gui, gz_sim_headless, bridge, eye_gaze])
