#!/usr/bin/env python3
"""
eval_benchmark.launch.py
===================
A unified launch file for completely isolated evaluation on the STANDARD benchmark
environment (turtlebot3_dqn_stage4.world). 

This launches EVERYTHING in one command:
 1. Gazebo Server (Headless physics) loading stage4 world.
 2. Robot Spawner & State Publisher
 3. Environment Core (`environment_ppo`)
 4. Benchmark Goal Manager (`gazebo_goals`) with start_phase=2 (so all standard goals are active)
 5. The specified SAC Agent in EVAL_ONLY=true mode

Usage example:
    ros2 launch tb3_drl_nav eval_benchmark.launch.py agent_version:=v10 run_id:=sac_v10_s42
    ros2 launch tb3_drl_nav eval_benchmark.launch.py agent_version:=baseline run_id:=sac_baseline_s42
    ros2 launch tb3_drl_nav eval_benchmark.launch.py agent_version:=v11 run_id:=sac_v11_s42
"""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, "launch")
    tb3_desc_dir = get_package_share_directory("turtlebot3_description")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")
    
    sdf_file = os.path.join(tb3_gazebo_dir, "models", "turtlebot3_waffle_pi", "model.sdf")

    # Arguments
    agent_version = LaunchConfiguration("agent_version")
    run_id = LaunchConfiguration("run_id")
    
    agent_cmd_str = PythonExpression([
        "'sac_mlp' if '", agent_version, "' == 'baseline' else ",
        "'sac_stam' if '", agent_version, "' == 'v8' else ",
        "'sac_stam_huber' if '", agent_version, "' == 'v10' else ",
        "'sac_r_stam' if '", agent_version, "' == 'v11' else ",
        "('train_agent_sac_' + '", agent_version, "') if '", agent_version, "' in ['v9a', 'v9b'] else '",
        agent_version,
        "'"
    ])

    world_path = LaunchConfiguration("world")
    eval_tag = LaunchConfiguration("eval_tag")
    start_x = LaunchConfiguration("start_x")
    start_y = LaunchConfiguration("start_y")

    return LaunchDescription([
        DeclareLaunchArgument("agent_version", default_value="v10", description="baseline, v8, v9a, v9b, v10, v11, or exact executable name"),
        DeclareLaunchArgument("run_id", default_value="sac_v10_s42", description="Exact run_id folder name to load weights from"),
        DeclareLaunchArgument("dz_mode", default_value="stam", description="Danger zone preprocessing mode"),
        DeclareLaunchArgument("world", default_value=os.path.expanduser("~/tubitak_2209_ws/worlds/benchmark_dqn_stage4.world"), description="Absolute path to benchmark world"),
        DeclareLaunchArgument("eval_tag", default_value="benchmark_dqn_stage4", description="Tag appended to eval CSV filename"),
        DeclareLaunchArgument("start_x", default_value="-2.0", description="Robot spawn x"),
        DeclareLaunchArgument("start_y", default_value="-0.5", description="Robot spawn y"),

        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("EVAL_ONLY", "true"),  # Force agents to act deterministically
        SetEnvironmentVariable("DZ_MODE", LaunchConfiguration("dz_mode")),
        SetEnvironmentVariable("EVAL_TAG", eval_tag),
        SetEnvironmentVariable("SAC_MODEL_DIR", os.environ.get("SAC_MODEL_DIR", os.path.expanduser("~/tb3_drl_models/sac"))),
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            "/opt/ros/humble/share/turtlebot3_gazebo/models:"
            + os.path.expanduser("~/tubitak_2209_ws/models")
            + ":" + os.environ.get("GAZEBO_MODEL_PATH", "")
        ),

        # 1. Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")),
            launch_arguments={"world": world_path, "verbose": "false", "pause": "false"}.items(),
        ),

        # 2. State Publisher
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_launch_dir, "robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": "true"}.items(),
        ),
        
        # 3. Spawn TurtleBot3 at origin
        ExecuteProcess(
            cmd=[
                "ros2", "run", "gazebo_ros", "spawn_entity.py",
                "-entity", "waffle_pi",
                "-file", sdf_file,
                "-x", start_x, "-y", start_y, "-z", "0.01",
            ],
            output="screen",
        ),

        # Give Gazebo 8 seconds to start — benchmark worlds load slower than main arena
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
                
                # 5. Benchmark Goal Manager (goals verified for 5×5m ROBOTIS worlds)
                Node(
                    package="tb3_drl_nav",
                    executable="benchmark_goals",
                    output="screen"
                ),

                # 6. Benchmark Obstacle Controller (moves dynamic obstacles)
                Node(
                    package="tb3_drl_nav",
                    executable="benchmark_obstacle_controller",
                    output="screen"
                ),

                # 7. Evaluation Agent
                Node(
                    package="tb3_drl_nav", 
                    executable=agent_cmd_str,
                    parameters=[{"run_id": run_id, "fresh": False}],
                    output="screen"
                ),
            ]
        )
    ])
