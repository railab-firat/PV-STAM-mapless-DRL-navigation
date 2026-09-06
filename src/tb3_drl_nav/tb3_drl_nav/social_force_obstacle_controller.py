#!/usr/bin/env python3
"""
social_force_obstacle_controller.py
=====================================================================
Controls 6 dynamic obstacles in the 12×8 m TAGD comparison arena
using random-direction sinusoidal trajectories (social-force approximation).

Motion model per obstacle i at tick t:
    v_i(t)  = V_PEAK * sin(2*pi/PERIOD * t + phi_i)
    vx_i(t) = v_i(t) * cos(theta_i)
    vy_i(t) = v_i(t) * sin(theta_i)

Every RERANDOMIZE_TICKS ticks:  theta_i ~ U(0, 2*pi),  phi_i ~ U(0, 2*pi)

Usage:
    SEED=42 ros2 run tb3_drl_nav social_force_obstacle_controller
"""
import math
import os
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ── Configuration ─────────────────────────────────────────────────────────────
SEED              = int(os.environ.get("SEED", 42))
V_PEAK            = 0.18          # m/s  (matches TAGD evaluation speed)
PERIOD            = 50            # ticks per full sinusoidal cycle
RERANDOMIZE_TICKS = 50            # ticks between direction/phase re-draws
TIMER_HZ          = 20.0          # controller frequency (20 Hz → 0.05 s / tick)

OBSTACLE_NAMES = [
    "dyn_obs_1",
    "dyn_obs_2",
    "dyn_obs_3",
    "dyn_obs_4",
    "dyn_obs_5",
    "dyn_obs_6",
]

# ── Main Node ─────────────────────────────────────────────────────────────────
class SocialForceObstacleController(Node):
    def __init__(self):
        super().__init__("social_force_obstacle_controller")

        # Seed for reproducibility
        random.seed(SEED)

        # Per-obstacle state: direction theta (rad), phase phi (rad), tick counter
        n = len(OBSTACLE_NAMES)
        self._theta = [random.uniform(0.0, 2 * math.pi) for _ in range(n)]
        self._phi   = [random.uniform(0.0, 2 * math.pi) for _ in range(n)]
        self._tick  = 0

        # Publishers
        self._pubs = []
        for name in OBSTACLE_NAMES:
            pub = self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            self._pubs.append(pub)

        # Main timer
        self._dt = 1.0 / TIMER_HZ
        self.create_timer(self._dt, self._step)

        self.get_logger().info(
            f"[SocialForce] Controlling {n} obstacles @ {TIMER_HZ} Hz  "
            f"V_PEAK={V_PEAK} m/s  SEED={SEED}"
        )

    def _step(self):
        """Called every 1/TIMER_HZ seconds."""
        t = self._tick

        # Re-draw direction and phase every RERANDOMIZE_TICKS ticks
        if t > 0 and t % RERANDOMIZE_TICKS == 0:
            for i in range(len(OBSTACLE_NAMES)):
                self._theta[i] = random.uniform(0.0, 2 * math.pi)
                self._phi[i]   = random.uniform(0.0, 2 * math.pi)
            self.get_logger().debug(f"[SocialForce] Re-randomised at tick {t}")

        # Publish velocity for each obstacle
        for i, pub in enumerate(self._pubs):
            v_mag = V_PEAK * math.sin(2 * math.pi / PERIOD * t + self._phi[i])
            msg = Twist()
            msg.linear.x  = v_mag * math.cos(self._theta[i])
            msg.linear.y  = v_mag * math.sin(self._theta[i])
            msg.angular.z = 0.0
            pub.publish(msg)

        self._tick += 1


def main(args=None):
    rclpy.init(args=args)
    node = SocialForceObstacleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop all obstacles cleanly on shutdown
        stop = Twist()
        for pub in node._pubs:
            pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
