#!/usr/bin/env python3
"""train_sac_pv_stam_no_omega.launch.py — Experiment 4: No-ω ablation training."""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

WORLD = os.path.expanduser("~/tubitak_2209_ws/worlds/phase3_arena_.world")

def generate_launch_description():
    tb3_gz  = get_package_share_directory("turtlebot3_gazebo")
    tb3_lch = os.path.join(tb3_gz, "launch")
    gz_ros  = get_package_share_directory("gazebo_ros")
    sdf     = os.path.join(tb3_gz, "models", "turtlebot3_waffle_pi", "model.sdf")
    seed    = os.environ.get("SEED", "42")
    run_id  = LaunchConfiguration("run_id")

    return LaunchDescription([
        DeclareLaunchArgument("run_id", default_value=f"sac_pv_stam_no_omega_s{seed}"),
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("SEED", seed),
        SetEnvironmentVariable("RUN_ID", run_id),
        # Normal world (RTF=1.0). STEP_SLEEP_S unset → default 0.15s matches baseline MDP exactly.
        # Switched from fast world: fast world's 50 ODE iterations caused physics artifacts
        # (spawn-area false collisions), invalidating 97% of episodes. Normal world is reliable.
        SetEnvironmentVariable("GAZEBO_MODEL_PATH",
            "/opt/ros/humble/share/turtlebot3_gazebo/models:"
            + os.path.expanduser("~/tubitak_2209_ws/models")
            + ":" + os.environ.get("GAZEBO_MODEL_PATH", "")),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gz_ros,"launch","gzserver.launch.py")),
            launch_arguments={"world": WORLD, "verbose": "false", "pause": "false"}.items()),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_lch,"robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": "true"}.items()),

        ExecuteProcess(cmd=["ros2","run","gazebo_ros","spawn_entity.py",
            "-entity","waffle_pi","-file",sdf,"-x","-2.0","-y","-0.5","-z","0.01"],
            output="screen"),

        TimerAction(period=6.0, actions=[
            Node(package="tb3_drl_nav", executable="nav_environment",
                 parameters=[{"start_x": -2.0, "start_y": -0.5}], output="screen"),
            Node(package="tb3_drl_nav", executable="goal_manager_dynamic",
                 parameters=[{"start_phase": 0}], output="screen"),
            Node(package="tb3_drl_nav", executable="obstacle_controller", output="screen"),
            Node(package="tb3_drl_nav", executable="sac_pv_stam_no_omega",
                 parameters=[{"run_id": run_id, "fresh": False}], output="screen"),
        ]),
    ])
