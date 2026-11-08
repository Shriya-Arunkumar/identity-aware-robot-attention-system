"""
attention_gazebo.launch.py — starts Gazebo Sim with the attention demo
world, bridges the head-yaw command topic into it, and starts
gazebo_response_node. Does NOT start the webcam process (main_webcam.py) —
run that separately, per run_demo.sh, so it can be swapped/restarted
independently of the sim.

Designed to run directly by path (no colcon build required), matching the
project's "don't block the demo on ROS tooling" constraint:

    ros2 launch ros2_ws/src/gaze_attention_ros/launch/attention_gazebo.launch.py

Args:
    world_path     default: multi_person/gazebo/worlds/attention_demo.sdf
    headless       default: false (pass true for -s, server-only, no GUI)
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
_DEFAULT_WORLD = str(_PACKAGE_DIR.parent.parent.parent / "gazebo" / "worlds" / "attention_demo.sdf")

# gaze_attention_ros is deliberately run as a plain script here (not via a
# `Node(package=...)` lookup) so this launch file works straight from source
# without a colcon build — consistent with the project's "don't block the
# demo on ROS tooling" constraint. ros_gz_bridge, unlike this package, is a
# real installed system package, so it uses the normal Node() lookup.
_NODE_ENV = dict(os.environ)
_NODE_ENV["PYTHONPATH"] = str(_PACKAGE_DIR) + os.pathsep + _NODE_ENV.get("PYTHONPATH", "")


def generate_launch_description() -> LaunchDescription:
    world_arg = DeclareLaunchArgument("world_path", default_value=_DEFAULT_WORLD)
    headless_arg = DeclareLaunchArgument("headless", default_value="false")

    # Two statically-argv'd variants gated by condition, rather than building
    # the flag string at runtime — passing "-s -r" as one concatenated
    # substitution reaches gz as a single malformed argv token, not two flags.
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
        arguments=["/gaze_attention/head_yaw_cmd@std_msgs/msg/Float64]gz.msgs.Double"],
        output="screen",
    )

    response_node = ExecuteProcess(
        cmd=["python3", "-m", "gaze_attention_ros.gazebo_response_node"],
        additional_env=_NODE_ENV,
        output="screen",
    )

    return LaunchDescription([world_arg, headless_arg, gz_sim_gui, gz_sim_headless, bridge, response_node])
