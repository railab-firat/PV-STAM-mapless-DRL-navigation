from setuptools import setup
import os

package_name = 'tb3_drl_nav'

setup(
    name=package_name,
    version='3.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch file — found by: ros2 launch tb3_drl_nav phase3_launch.py
        (os.path.join('share', package_name, 'launch'),
            ['launch/phase3_launch.py', 'launch/benchmark_launch.py', 'launch/eval_sac.launch.py',
             'launch/eval_benchmark.launch.py', 'launch/train_sac_v11.launch.py',
             'launch/env_only.launch.py', 'launch/eval_canonical.launch.py',
             'launch/eval_tagd.launch.py', 'launch/train_sac_lstm.launch.py',
             'launch/finetune_benchC.launch.py', 'launch/train_sac_pv_stam_no_omega.launch.py',
             'launch/train_sac_v10_matched.launch.py', 'launch/train_sac_lstm_forced.launch.py']),
        # Config files — one per phase
        (os.path.join('share', package_name, 'config'),
            ['config/phase1_dqn.yaml', 'config/phase3_ppo.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anas',
    maintainer_email='anas@todo.todo',
    description='Phase 3 PPO DRL Navigation with dynamic obstacles',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ── Phase 1 — DQN (unchanged, do not remove) ─────────────────────
            'gazebo_goals         = tb3_drl_nav.gazebo_goals:main',
            'environment          = tb3_drl_nav.environment:main',
            'train_agent          = tb3_drl_nav.train_agent_dqn:main',
            'train_agent_dqn      = tb3_drl_nav.train_agent_dqn:main',

            # ── Phase 3 — PPO + dynamic arena ────────────────────────────────
            'nav_environment      = tb3_drl_nav.nav_environment:main',
            'train_agent_ppo      = tb3_drl_nav.train_agent_ppo:main',
            'goal_manager_dynamic = tb3_drl_nav.goal_manager_dynamic:main',
            'goal_manager_fixed   = tb3_drl_nav.goal_manager_fixed:main',
            'obstacle_controller  = tb3_drl_nav.obstacle_controller:main',
            'benchmark_goals      = tb3_drl_nav.benchmark_goals:main',
            'benchmark_obstacle_controller = tb3_drl_nav.benchmark_obstacle_controller:main',
            'live_plot            = tb3_drl_nav.live_plot:main',
            'check_hardware       = tb3_drl_nav.check_hardware:main',

            # ── Phase 3 — SAC (off-policy replacement for PPO) ───────────────
            'train_agent_sac      = tb3_drl_nav.train_agent_sac:main',
            'train_sac_stam       = tb3_drl_nav.train_sac_stam:main',
            'train_agent_sac_v9a  = tb3_drl_nav.train_agent_sac_v9a:main',
            'train_agent_sac_v9b  = tb3_drl_nav.train_agent_sac_v9b:main',
            'train_sac_stam_huber = tb3_drl_nav.train_sac_stam_huber:main',
            'train_sac_r_stam     = tb3_drl_nav.train_sac_r_stam:main',

            # ── Ablation study — clean comparable implementations ─────────────
            'sac_mlp              = tb3_drl_nav.sac_mlp:main',
            'sac_mlp_fs           = tb3_drl_nav.sac_mlp_fs:main',
            'sac_stam             = tb3_drl_nav.sac_stam:main',
            'sac_stam_huber       = tb3_drl_nav.sac_stam_huber:main',
            'sac_r_stam           = tb3_drl_nav.sac_r_stam:main',
            'canonical_eval       = tb3_drl_nav.canonical_eval:main',

            # ── Journal experiments (2026-06) ─────────────────────────────────
            'social_force_obstacle_controller = tb3_drl_nav.social_force_obstacle_controller:main',
            'tagd_goals           = tb3_drl_nav.tagd_goals:main',
            'sac_lstm             = tb3_drl_nav.sac_lstm:main',
            'sac_pv_stam_no_omega = tb3_drl_nav.sac_pv_stam_no_omega:main',

            # ── Task 8 revision runs (MDPI major revision, 2026-08) ────────────
            'sac_v10_matched       = tb3_drl_nav.sac_v10_matched:main',
            'goal_manager_dynamic_forced = tb3_drl_nav.goal_manager_dynamic_forced:main',

            # ── Real-robot deployment (all variants) ──────────────────────────
            'real_robot_nav       = tb3_drl_nav.real_robot_nav:main',
        ],
    },
)
