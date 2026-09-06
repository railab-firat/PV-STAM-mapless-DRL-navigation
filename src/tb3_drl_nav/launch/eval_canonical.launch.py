#!/usr/bin/env python3
"""
eval_canonical.launch.py
=============================================================================
ROS2 Launch file for isolated canonical evaluation.
Launches Gazebo, environment core, goal managers, and the canonical_eval node.
"""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, "launch")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")
    sdf_file = os.path.join(tb3_gazebo_dir, "models", "turtlebot3_waffle_pi", "model.sdf")

    # Arguments
    variant = LaunchConfiguration("variant")
    seed = LaunchConfiguration("seed")
    max_episodes = LaunchConfiguration("max_episodes")
    world_path = LaunchConfiguration("world")
    eval_tag = LaunchConfiguration("eval_tag")
    start_x = LaunchConfiguration("start_x")
    start_y = LaunchConfiguration("start_y")
    dz_mode = LaunchConfiguration("dz_mode")

    return LaunchDescription([
        DeclareLaunchArgument("variant", default_value="v10", description="baseline, mlp_fs, v8, v10, or v11"),
        DeclareLaunchArgument("seed", default_value="42", description="42, 777, or 123"),
        DeclareLaunchArgument("max_episodes", default_value="100", description="Number of episodes to evaluate"),
        DeclareLaunchArgument("dz_mode", default_value="stam", description="Danger zone mode"),
        DeclareLaunchArgument("world", default_value=os.path.expanduser("~/tubitak_2209_ws/worlds/benchmark_dqn_stage4.world"), description="Absolute path to benchmark world"),
        DeclareLaunchArgument("eval_tag", default_value="benchmark_dqn_stage4", description="Tag appended to eval CSV filename"),
        DeclareLaunchArgument("start_x", default_value="-2.0", description="Robot spawn x"),
        DeclareLaunchArgument("start_y", default_value="-0.5", description="Robot spawn y"),

        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("EVAL_ONLY", "true"),
        SetEnvironmentVariable("DZ_MODE", dz_mode),
        SetEnvironmentVariable("EVAL_TAG", eval_tag),
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            "/opt/ros/humble/share/turtlebot3_gazebo/models:"
            + os.path.expanduser("~/tubitak_2209_ws/models")
            + ":" + os.environ.get("GAZEBO_MODEL_PATH", "")
        ),

        # 1. Gazebo Server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")),
            launch_arguments={"world": world_path, "verbose": "false", "pause": "false"}.items(),
        ),

        # 2. State Publisher
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_launch_dir, "robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": "true"}.items(),
        ),
        
        # 3. Spawn TurtleBot3
        ExecuteProcess(
            cmd=[
                "ros2", "run", "gazebo_ros", "spawn_entity.py",
                "-entity", "waffle_pi",
                "-file", sdf_file,
                "-x", start_x, "-y", start_y, "-z", "0.01",
            ],
            output="screen",
        ),

        # Delay starting core nodes to allow Gazebo server to launch stably
        TimerAction(
            period=8.0,
            actions=[
                # 4. Environment Core
                Node(
                    package="tb3_drl_nav",
                    executable="nav_environment",
                    parameters=[{"start_x": start_x, "start_y": start_y}],
                    output="screen"
                ),
                
                # 5. Benchmark Goal Manager
                Node(
                    package="tb3_drl_nav",
                    executable="benchmark_goals",
                    parameters=[{"seed": seed}],
                    output="screen"
                ),

                # 6. Benchmark Obstacle Controller
                Node(
                    package="tb3_drl_nav",
                    executable="benchmark_obstacle_controller",
                    output="screen"
                ),

                # 7. Canonical Agent Node
                Node(
                    package="tb3_drl_nav", 
                    executable="canonical_eval",
                    parameters=[{
                        "variant": variant,
                        "seed": seed,
                        "max_episodes": max_episodes,
                        "eval_tag": eval_tag,
                        "log_dir": "/home/anas/tb3_drl_logs/canonical"
                    }],
                    output="screen"
                ),
            ]
        )
    ])
