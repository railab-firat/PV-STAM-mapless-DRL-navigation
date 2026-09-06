#!/usr/bin/env python3
"""
benchmark_obstacle_controller.py  — Move dynamic obstacles in benchmark worlds
===============================================================================
Benchmark worlds have 4-6 dynamic obstacles with planar_move plugins.
This controller sends sinusoidal velocity commands to make them move,
providing a dynamic obstacle avoidance challenge during evaluation.

No curriculum dependency — obstacles always move at a fixed speed.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Speed tuned to ~65% of robot max (0.26 m/s) — challenging but fair
V_PEAK = 0.12

# Each obstacle: (namespace, axis, period_s, phase_offset)
# Covers up to 6 obstacles (stage3 has 4, stage4 and tb3_world have 6).
# Publishing to non-existent namespaces is harmless (no subscriber = no effect).
OBSTACLES = [
    ("dyn_obs_1", "x", 10.0, 0.00),
    ("dyn_obs_2", "y", 12.0, math.pi / 3),
    ("dyn_obs_3", "y", 11.0, math.pi / 2),
    ("dyn_obs_4", "x", 13.0, math.pi),
    ("dyn_obs_5", "x",  9.0, math.pi / 4),
    ("dyn_obs_6", "y", 14.0, 2 * math.pi / 3),
]


class BenchmarkObstacleController(Node):
    def __init__(self):
        super().__init__("benchmark_obstacle_controller")
        self._pubs = {
            ns: self.create_publisher(Twist, f"/{ns}/cmd_vel", 10)
            for ns, *_ in OBSTACLES
        }
        self._t0 = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(0.05, self._tick)

        self.get_logger().info("=" * 60)
        self.get_logger().info("  Benchmark Obstacle Controller")
        self.get_logger().info(f"  {len(OBSTACLES)} obstacles @ V_peak={V_PEAK} m/s")
        self.get_logger().info("=" * 60)

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self._t0
        for ns, axis, period, phase in OBSTACLES:
            msg = Twist()
            v = V_PEAK * math.sin(2 * math.pi / period * t + phase)
            if axis == "x":
                msg.linear.x = v
            else:
                msg.linear.y = v
            self._pubs[ns].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkObstacleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
