#!/usr/bin/env python3
"""
train_sac_v11.launch.py  Single launch for v11 training
=============================================================================
Launches everything in one command:
  1. gzserver (headless physics)
  2. robot_state_publisher
  3. obstacle_controller + goal_manager_dynamic + environment_ppo  (after 3 s)
  4. train_agent_sac_v11  (after 8 s)

Usage:
  # Fresh start:
  ros2 launch tb3_drl_nav train_sac_v11.launch.py fresh:=true

  # Resume from last checkpoint:
  ros2 launch tb3_drl_nav train_sac_v11.launch.py

  # Custom run ID:
  ros2 launch tb3_drl_nav train_sac_v11.launch.py run_id:=sac_v11_exp2 fresh:=true
"""
import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

_WS = os.path.expanduser("~/tubitak_2209_ws")

_WORLD_UNDERSCORE = os.path.join(_WS, "worlds", "phase3_arena_.world")
_WORLD_PLAIN      = os.path.join(_WS, "worlds", "phase3_arena.world")
if os.path.exists(_WORLD_UNDERSCORE):
    WORLD = _WORLD_UNDERSCORE
elif os.path.exists(_WORLD_PLAIN):
    WORLD = _WORLD_PLAIN
else:
    raise FileNotFoundError("No phase3 world file found.")

# Run at parse time — before any process starts
_phase_file = os.path.expanduser("~/tb3_drl_logs/phase3/current_phase.txt")
_curric_state = os.path.expanduser("~/tb3_drl_logs/phase3/curriculum_state.json")
for _f, _label in [(_phase_file, "current_phase.txt"), (_curric_state, "curriculum_state.json")]:
    if os.path.exists(_f):
        os.remove(_f)
        print(f"[train_sac_v11.launch] Cleared stale {_label}")

# Kill any orphaned gzserver / training processes from a previous crashed run
os.system("pkill -9 -f gzserver 2>/dev/null; pkill -9 -f train_sac_r_stam 2>/dev/null; sleep 1")
print("[train_sac_v11.launch] Killed any orphaned gzserver/trainer processes")

print(f"[train_sac_v11.launch] world: {WORLD}")


def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, "launch")
    gazebo_ros_dir  = get_package_share_directory("gazebo_ros")
    use_sim_time    = LaunchConfiguration("use_sim_time", default="true")
    run_id          = LaunchConfiguration("run_id")
    fresh           = LaunchConfiguration("fresh")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("run_id",       default_value="sac_v11_fixed"),
        DeclareLaunchArgument("fresh",        default_value="false"),

        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("ROS_LOG_LEVEL",    "WARN"),

        # ── Step 1: Gazebo server ──────────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")),
            launch_arguments={
                "world":   WORLD,
                "verbose": "false",
                "pause":   "false",
            }.items(),
        ),

        # ── Step 2: Robot state publisher ─────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_launch_dir, "robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),

        # ── Step 3: Environment nodes (wait 3 s for Gazebo to settle) ─────────
        TimerAction(period=3.0, actions=[

            Node(
                package="tb3_drl_nav",
                executable="obstacle_controller",
                name="obstacle_controller",
                output="log",
                parameters=[{"use_sim_time": use_sim_time}],
            ),

            Node(
                package="tb3_drl_nav",
                executable="goal_manager_dynamic",
                name="goal_manager_dynamic",
                output="log",
                parameters=[{
                    "use_sim_time": use_sim_time,
                    "start_phase":  1,
                }],
            ),

            Node(
                package="tb3_drl_nav",
                executable="nav_environment",
                name="nav_environment",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]),

        # ── Step 4: Training agent (wait 8 s — environment must be ready) ─────
        TimerAction(period=8.0, actions=[
            Node(
                package="tb3_drl_nav",
                executable="train_sac_r_stam",
                name="train_sac_r_stam",
                output="screen",
                parameters=[{
                    "run_id": run_id,
                    "fresh":  fresh,
                }],
            ),
        ]),
    ])
