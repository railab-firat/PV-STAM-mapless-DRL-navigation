#!/usr/bin/env python3
"""
benchmark_launch.py — Launch a benchmark world with TurtleBot3 for STAM evaluation.

Unlike phase3_launch.py, benchmark worlds don't have the robot baked in,
so we spawn it via spawn_entity.

Usage:
    WORLD_FILE=~/tubitak_2209_ws/worlds/benchmark_dqn_stage4.world \
        ros2 launch tb3_drl_nav benchmark_launch.py

    # Or specify directly:
    ros2 launch tb3_drl_nav benchmark_launch.py \
        world:=$HOME/tubitak_2209_ws/worlds/benchmark_dqn_stage4.world
"""
import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, SetEnvironmentVariable,
    DeclareLaunchArgument, ExecuteProcess
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

_WS = os.path.expanduser("~/tubitak_2209_ws")
_DEFAULT_WORLD = os.path.join(_WS, "worlds", "benchmark_dqn_stage4.world")

# Allow override via WORLD_FILE env var
WORLD = os.environ.get("WORLD_FILE", _DEFAULT_WORLD)
WORLD = os.path.expanduser(WORLD)


def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, "launch")
    tb3_desc_dir = get_package_share_directory("turtlebot3_description")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # URDF for spawn_entity
    urdf_file = os.path.join(
        tb3_desc_dir, "urdf", "turtlebot3_waffle_pi.urdf")

    print(f"[benchmark_launch] World: {WORLD}")
    print(f"[benchmark_launch] Robot URDF: {urdf_file}")

    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),

        # ── Gazebo server ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")),
            launch_arguments={
                "world": WORLD,
                "verbose": "false",
                "pause": "false",
            }.items(),
        ),

        # ── Spawn TurtleBot3 at origin ──
        ExecuteProcess(
            cmd=[
                "ros2", "run", "gazebo_ros", "spawn_entity.py",
                "-entity", "waffle_pi",
                "-file", urdf_file,
                "-x", "0.0", "-y", "0.0", "-z", "0.01",
            ],
            output="screen",
        ),

        # ── Robot state publisher ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_launch_dir, "robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),
    ])
