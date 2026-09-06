#!/usr/bin/env python3
"""
env_only.launch.py
=======================================
Starts Gazebo + environment nodes ONLY (no trainer).
Use this when you want to run a specific ablation agent manually:

  # Terminal 1 — environment:
  ros2 launch tb3_drl_nav env_only.launch.py

  # Terminal 2 — agent (your choice):
  SEED=42 ros2 run tb3_drl_nav sac_v8     --ros-args -p run_id:=v8_s42    -p fresh:=true
  SEED=42 ros2 run tb3_drl_nav sac_v10    --ros-args -p run_id:=v10_s42   -p fresh:=true
  SEED=42 ros2 run tb3_drl_nav sac_v11    --ros-args -p run_id:=v11_s42   -p fresh:=true
  SEED=42 ros2 run tb3_drl_nav sac_baseline --ros-args -p run_id:=bl_s42  -p fresh:=true
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

# Clean stale phase files on every launch (prevents auto-resume to wrong phase)
_phase_file = os.path.expanduser("~/tb3_drl_logs/phase3/current_phase.txt")
_curric_state = os.path.expanduser("~/tb3_drl_logs/phase3/curriculum_state.json")
for _f in (_phase_file, _curric_state):
    if os.path.exists(_f):
        os.remove(_f)

# Kill orphaned processes from previous crashed runs
os.system("pkill -9 -f gzserver 2>/dev/null; sleep 1")

print(f"[env_only.launch] world: {WORLD}")


def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, "launch")
    gazebo_ros_dir  = get_package_share_directory("gazebo_ros")
    use_sim_time    = LaunchConfiguration("use_sim_time", default="true")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),

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
                output="log",          # goal spam goes to log file, not terminal
                parameters=[{
                    "use_sim_time": use_sim_time,
                    # FRESH env var: "true" = Phase 1 (fresh), else = auto-resume
                    "start_phase": 1 if os.environ.get("FRESH", "").lower() == "true" else 0,
                }],
            ),

            Node(
                package="tb3_drl_nav",
                executable="environment_ppo",
                name="environment_ppo",
                output="log",
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]),

        # ── NO TRAINER — run your agent manually in a separate terminal ────────
    ])
