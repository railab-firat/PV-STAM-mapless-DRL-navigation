#!/usr/bin/env python3
"""
eval_tagd.launch.py
===================================================
Unified launch for Experiment 1: TAGD Comparison Arena evaluation.

Launches:
  1. Gazebo (headless) — open_arena_12x8.world
  2. Robot state publisher
  3. TurtleBot3 spawn at (−4.5, 0.0)
  4. nav_environment
  5. social_force_obstacle_controller (6 obstacles @ 0.18 m/s)
  6. tagd_goals (200-episode manager)
  7. canonical_eval — variant=v11, seed=42, 200 episodes

Usage:
    ros2 launch tb3_drl_nav eval_tagd.launch.py
"""
import os
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, SetEnvironmentVariable,
                             ExecuteProcess, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

WORLD_PATH = os.path.expanduser(
    "~/tubitak_2209_ws/worlds/open_arena_12x8.world")
MODEL_DIR  = os.environ.get("SAC_MODEL_DIR",
                             os.path.expanduser("~/tb3_drl_models/sac"))


def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, "launch")
    tb3_desc_dir   = get_package_share_directory("turtlebot3_description")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")

    sdf_file = os.path.join(
        tb3_gazebo_dir, "models", "turtlebot3_waffle_pi", "model.sdf")

    return LaunchDescription([
        # ── Environment ───────────────────────────────────────────────────────
        SetEnvironmentVariable("TURTLEBOT3_MODEL",  "waffle_pi"),
        SetEnvironmentVariable("EVAL_ONLY",          "true"),
        SetEnvironmentVariable("EVAL_TAG",           "tagd_open_arena"),
        SetEnvironmentVariable("SAC_MODEL_DIR",      MODEL_DIR),
        SetEnvironmentVariable("SEED",               "42"),
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            "/opt/ros/humble/share/turtlebot3_gazebo/models:"
            + os.path.expanduser("~/tubitak_2209_ws/models")
            + ":" + os.environ.get("GAZEBO_MODEL_PATH", ""),
        ),

        # ── 1. Gazebo headless ────────────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")),
            launch_arguments={
                "world":   WORLD_PATH,
                "verbose": "false",
                "pause":   "false",
            }.items(),
        ),

        # ── 2. Robot state publisher ──────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_launch_dir, "robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": "true"}.items(),
        ),

        # ── 3. Spawn robot at (−4.5, 0.0) ─────────────────────────────────────
        ExecuteProcess(
            cmd=[
                "ros2", "run", "gazebo_ros", "spawn_entity.py",
                "-entity", "waffle_pi",
                "-file",   sdf_file,
                "-x", "-4.5", "-y", "0.0", "-z", "0.01",
            ],
            output="screen",
        ),

        # ── Wait 8 s for Gazebo, then start ROS nodes ─────────────────────────
        TimerAction(period=8.0, actions=[

            # 4. nav_environment
            Node(
                package="tb3_drl_nav",
                executable="nav_environment",
                parameters=[{"start_x": -4.5, "start_y": 0.0}],
                output="screen",
            ),

            # 5. Social-force obstacle controller
            Node(
                package="tb3_drl_nav",
                executable="social_force_obstacle_controller",
                output="screen",
            ),

            # 6. TAGD goal manager
            Node(
                package="tb3_drl_nav",
                executable="tagd_goals",
                output="screen",
            ),

            # 7. Canonical evaluator: v11 seed=42
            Node(
                package="tb3_drl_nav",
                executable="canonical_eval",
                parameters=[{
                    "variant":      "v11",
                    "seed":         42,
                    "max_episodes": 200,
                    "eval_tag":     "tagd_open_arena",
                }],
                output="screen",
            ),
        ]),
    ])
