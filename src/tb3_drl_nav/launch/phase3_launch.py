#!/usr/bin/env python3
"""
phase3_launch.py v7 Phase 3

IMPORTANT: The world file (phase3_arena_.world) already contains:
  - The waffle_pi robot with all sensor plugins (lidar, diff_drive, etc.)
  - All dynamic obstacle models (dyn_obs_1..6, dyn_obs_9..12, s1/s2/s6/s7/s8/s10, etc.)
  - The goal_pole marker is NOT in the world — spawned at runtime by goal_manager_dynamic
So we do NOT spawn a second robot here — that caused Gazebo to crash.

v7: gzclient (GUI) REMOVED — it was segfaulting (exit -11) on the Quadro M2000M
    GPU due to OpenGL rendering issues, bringing down gzserver with it.
    Training only needs gzserver (physics). Headless = stable overnight runs.

This launch starts:
  1. gzserver — physics + world file (headless, no GUI)
  2. robot_state_publisher — publishes TF from URDF + /joint_states
"""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Detect which world file to use
# Override with WORLD_FILE env var for benchmark testing:
#   WORLD_FILE=~/tubitak_2209_ws/worlds/benchmark_dqn_stage4.world
_WS = os.path.expanduser("~/tubitak_2209_ws")
_WORLD_OVERRIDE = os.environ.get("WORLD_FILE", "")
_WORLD_UNDERSCORE = os.path.join(_WS, "worlds", "phase3_arena_.world")
_WORLD_PLAIN      = os.path.join(_WS, "worlds", "phase3_arena.world")

if _WORLD_OVERRIDE and os.path.exists(os.path.expanduser(_WORLD_OVERRIDE)):
    WORLD = os.path.expanduser(_WORLD_OVERRIDE)
elif os.path.exists(_WORLD_UNDERSCORE):
    WORLD = _WORLD_UNDERSCORE
elif os.path.exists(_WORLD_PLAIN):
    WORLD = _WORLD_PLAIN
else:
    raise FileNotFoundError(
        "No world file found. Expected one of:\n"
        f"  {_WORLD_UNDERSCORE}\n"
        f"  {_WORLD_PLAIN}")

print(f"[phase3_launch] Using world: {WORLD}")


def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, "launch")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")
    use_sim_time   = LaunchConfiguration("use_sim_time", default="true")

    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),

        # ── Gazebo server (loads world file which already has the robot) ──────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")),
            launch_arguments={
                "world":   WORLD,
                "verbose": "false",
                "pause":   "false",
            }.items(),
        ),

        # ── Gazebo GUI — DISABLED (headless mode) ────────────────────────────
        # gzclient was segfaulting (exit -11) on Quadro M2000M, crashing gzserver.
        # Training needs only physics (gzserver). Remove for stable overnight runs.

        # ── Robot state publisher (URDF → TF, needed for rviz/nav tools) ─────
        # The robot is already in the world file with its own joint state plugin,
        # so robot_state_publisher just converts those joint states to TF.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_launch_dir, "robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),

        # ── Obstacle Controller (Dynamic Obstacle Movement) ──────────────────
        Node(
            package='tb3_drl_nav',
            executable='obstacle_controller',
            name='obstacle_controller',
            output='log',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        # ── Goal Manager (Curriculum Phase & Goal Spawning) ──────────────────
        Node(
            package='tb3_drl_nav',
            executable='goal_manager_dynamic',
            name='goal_manager_dynamic',
            output='log',
            parameters=[{
                'use_sim_time': use_sim_time,
                'start_phase': LaunchConfiguration('start_phase', default='0'),
            }]
        ),

        # ── Environment Node (Observations & Rewards) ───────────────────────
        Node(
            package='tb3_drl_nav',
            executable='environment_ppo',
            name='environment_ppo',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        # NOTE: spawn_turtlebot3 is intentionally removed.
        # The world file already contains the full waffle_pi model with all
        # sensor plugins. Spawning again would create a duplicate and crash Gazebo.
    ])

