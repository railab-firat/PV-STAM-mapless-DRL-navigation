#!/usr/bin/env python3
"""
eval_sac.launch.py
===================
A unified launch file for completely isolated and deterministic evaluation.
This launches EVERYTHING in one command:
 1. Gazebo Server (Headless physics)
 2. Robot State Publisher
 3. Environment Core (`environment_ppo`)
 4. Fixed Goal Manager (`goal_manager_fixed` - no impossible goals!)
 5. Obstacle Controller
 6. The specified SAC Agent in EVAL_ONLY=true mode

Usage example:
  ros2 launch tb3_drl_nav eval_sac.launch.py agent_version:=v10 run_id:=sac_v10_stam
  ros2 launch tb3_drl_nav eval_sac.launch.py agent_version:=v8 run_id:=sac_v8_stam
  ros2 launch tb3_drl_nav eval_sac.launch.py agent_version:=baseline run_id:=sac_baseline
    ros2 launch tb3_drl_nav eval_sac.launch.py agent_version:=v11 run_id:=sac_v11_stam
    ros2 launch tb3_drl_nav eval_sac.launch.py agent_version:=sac_v8 run_id:=sac_v8_s42
"""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, "launch")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")
    pkg_dir = get_package_share_directory("tb3_drl_nav")

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

    world_path = os.path.expanduser("~/tubitak_2209_ws/worlds/phase3_arena_.world")

    return LaunchDescription([
        DeclareLaunchArgument("agent_version", default_value="v10", description="baseline, v8, v9a, v9b, v10, v11, or exact executable name"),
        DeclareLaunchArgument("run_id", default_value="sac_v10_stam", description="Exact run_id folder name to load weights from"),
        DeclareLaunchArgument("dz_mode", default_value="stam", description="Danger zone preprocessing mode"),
        DeclareLaunchArgument("eval_phase", default_value="7", description="Phase to lock eval at (5=slow dynamic, 6=medium, 7=full speed dynamic — default 7)"),

        # [domain-isolation] every node must inherit the SAME domain, otherwise two
        # parallel evaluations share a DDS domain and their goal managers cross-talk:
        # episodes from one run land in the other run's CSV. Verified failure mode.
        SetEnvironmentVariable("ROS_DOMAIN_ID",
                               os.environ.get("ROS_DOMAIN_ID", "0")),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("EVAL_ONLY", "true"),  # Force agents to act deterministically and skip learning
        SetEnvironmentVariable("DZ_MODE", LaunchConfiguration("dz_mode")),
        SetEnvironmentVariable("EVAL_PHASE", LaunchConfiguration("eval_phase")),
        SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
        SetEnvironmentVariable("SAC_MODEL_DIR", os.environ.get("SAC_MODEL_DIR", os.path.expanduser("~/tb3_drl_models/sac"))),

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

        # Give Gazebo 3 seconds to start before slamming the CPU with DRL nodes
        TimerAction(
            period=3.0,
            actions=[
                # 3. Environment Core
                Node(package="tb3_drl_nav", executable="nav_environment", output="log"),
                
        # 4. Goal Manager (dynamic, 7-phase) locked to eval_phase with no advancement
                Node(
                    package="tb3_drl_nav", executable="goal_manager_dynamic", output="log",
                    parameters=[{
                        "start_phase": LaunchConfiguration("eval_phase"),
                        "eval_mode": True,
                    }]
                ),
                
                # 5. Obstacle Controller
                Node(package="tb3_drl_nav", executable="obstacle_controller", output="log"),
                
                # 6. Evaluation Agent (using the dynamically chosen executable name based on agent_version)
                Node(
                    package="tb3_drl_nav", 
                    executable=agent_cmd_str,
                    parameters=[{"run_id": run_id, "fresh": False}],
                    output="screen"
                ),
            ]
        )
    ])
