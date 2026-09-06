#!/usr/bin/env python3
"""finetune_benchC.launch.py — Experiment 3: Fine-tune v11 s42 on Benchmark C."""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

WORLD      = os.path.expanduser("~/tubitak_2209_ws/worlds/benchmark_tb3_world.world")
BASE_CKPT  = "/home/anas/tb3_drl_models/sac/sac_v11_s42/ckpt_shutdown.pt"
MODEL_DIR  = os.environ.get("SAC_MODEL_DIR", os.path.expanduser("~/tb3_drl_models/sac"))

def generate_launch_description():
    tb3_gz  = get_package_share_directory("turtlebot3_gazebo")
    tb3_lch = os.path.join(tb3_gz, "launch")
    gz_ros  = get_package_share_directory("gazebo_ros")
    sdf     = os.path.join(tb3_gz, "models", "turtlebot3_waffle_pi", "model.sdf")

    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("SEED",             "42"),
        SetEnvironmentVariable("LOAD_WEIGHTS_PATH", BASE_CKPT),
        SetEnvironmentVariable("MAX_STEPS",        "50000"),
        SetEnvironmentVariable("WARMUP",           "0"),
        SetEnvironmentVariable("SAC_MODEL_DIR",    MODEL_DIR),
        SetEnvironmentVariable("RUN_ID",           "sac_v11_s42_finetune_benchC"),
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

        TimerAction(period=8.0, actions=[
            Node(package="tb3_drl_nav", executable="nav_environment",
                 parameters=[{"start_x": -2.0, "start_y": -0.5}], output="screen"),
            Node(package="tb3_drl_nav", executable="benchmark_goals", output="screen"),
            Node(package="tb3_drl_nav", executable="benchmark_obstacle_controller", output="screen"),
            Node(package="tb3_drl_nav", executable="train_sac_r_stam",
                 parameters=[{"run_id": "sac_v11_s42_finetune_benchC", "fresh": False}],
                 output="screen"),
        ]),
    ])
