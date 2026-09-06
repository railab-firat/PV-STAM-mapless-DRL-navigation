#!/usr/bin/env python3
"""
obstacle_controller.py  CURRICULUM v2.0

NEW CURRICULUM STRUCTURE:
  STATIC STAGE (Phases 1-4): ALL obstacles FROZEN - learn pure navigation
  DYNAMIC STAGE (Phases 5-7): Gradual obstacle activation and speed increase

Phase Breakdown:
  Phase 1-4: ALL obstacles FROZEN (V=0) - robot learns navigation without distraction
  Phase 5:   6 perimeter obstacles, SLOW speed (V=0.08 m/s, 44% of robot)
  Phase 6:   6 perimeter + 3 center (9 total), MEDIUM speed (V=0.12 m/s, 65% of robot)
  Phase 7:   ALL obstacles (15 total), FULL speed (V=0.18 m/s, 70% of robot)

This gradual approach prevents the SR crash seen when moving directly from
static to full-speed dynamic obstacles.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

# ═══════════════════════════════════════════════════════════════════════════════
# OBSTACLE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# 6 perimeter obstacles — activated first in Phase 5
OBSTACLES_PERIMETER = [
    ("dyn_obs_1",  "x", 12.0, 0.00),       # NW  at (-3.5, +3.5)
    ("dyn_obs_2",  "x", 13.0, math.pi),    # NE  at (+3.5, +3.5)
    ("dyn_obs_3",  "y", 11.0, 0.00),       # SW  at (-3.5, -3.5)
    ("dyn_obs_4",  "y", 14.0, math.pi),    # SE  at (+3.5, -3.5)
    ("dyn_obs_5",  "y", 10.0, 0.00),       # W   at (-2.5,  0.0)
    ("dyn_obs_6",  "y", 10.0, math.pi),    # E   at (+2.5,  0.0)
]

# 4 corridor blockers — blocked "behind the blue goals" (Phase 6)
OBSTACLES_CORRIDORS = [
    ("s10", "x",  8.0, 0.00),    # North corridor (0, +3.5)
    ("s2",  "x",  8.0, math.pi), # South corridor (0, -3.5) -- moved in world
    ("s6",  "y",  8.0, 0.00),    # West corridor (-3.5, 0)
    ("s7",  "y",  8.0, math.pi), # East corridor (+3.5, 0)
]

# 4 corner-room deep blockers (Phase 7)
OBSTACLES_DEEP_CORNERS = [
    ("dyn_obs_9",  "y", 10.0, 0.00),    # NW corner
    ("dyn_obs_10", "y", 10.0, math.pi), # NE corner
    ("dyn_obs_11", "x", 10.0, 0.00),    # SW corner
    ("dyn_obs_12", "x", 10.0, math.pi), # SE corner
]

# 6 additional obstacles — added in Phase 7
OBSTACLES_CENTER_EXTRA = [
    ("s1",  "y", 20.0, 0.00),              # centre-left   (-1.5,  0.0)
    ("s8",  "x", 15.0, 0.00),              # upper-centre  (-0.8, +1.8)
    ("uc_clone",       "y",  8.0, 0.00),   # inner (-0.8, -1.6)
    ("uc_clone_1",     "x",  9.0, math.pi),# inner (+0.6, +2.0)
    ("uc_clone_clone", "y",  8.0, math.pi),# inner (+0.9, -1.4)
]

# Combined sets for each phase
ALL_OBSTACLES = (OBSTACLES_PERIMETER + OBSTACLES_CORRIDORS + 
                 OBSTACLES_DEEP_CORNERS + OBSTACLES_CENTER_EXTRA)

# ═══════════════════════════════════════════════════════════════════════════════
# SPEED SETTINGS BY PHASE
# ═══════════════════════════════════════════════════════════════════════════════
# Robot max speed: 0.26 m/s

V_SLOW   = 0.08   # Phase 5: 44% of robot max — easy to outrun
V_MEDIUM = 0.12   # Phase 6: 65% of robot max — requires attention
V_FULL   = 0.18   # Phase 7: 70% of robot max — challenging but fair

PHASE_SPEEDS = {
    1: 0.0,        # FROZEN
    2: 0.0,        # FROZEN
    3: 0.0,        # FROZEN
    4: 0.0,        # FROZEN
    5: V_SLOW,     # Slow dynamics
    6: V_MEDIUM,   # Medium dynamics
    7: V_FULL,     # Full dynamics
}


class ObstacleController(Node):
    def __init__(self):
        super().__init__("obstacle_controller")
        self._phase = 1   # tracks curriculum phase
        self._v_peak = 0.0  # current speed (0 for static phases)

        # Create publishers for ALL obstacles (even if frozen initially)
        self._pubs = {
            name: self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            for name, *_ in ALL_OBSTACLES
        }
        self._t0 = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(0.05, self._tick)

        # Subscribe with TRANSIENT_LOCAL — gets last published phase even if late to start
        _latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Int32, "/tb3_drl/curriculum_phase",
                                 self._on_phase, _latched)

        self.get_logger().info("=" * 60)
        self.get_logger().info("  ObstacleController v2.0 — CURRICULUM ALIGNED")
        self.get_logger().info("  Phases 1-4: ALL FROZEN (static navigation)")
        self.get_logger().info(f"  Phase 5: 6 obstacles @ V={V_SLOW} m/s (slow)")
        self.get_logger().info(f"  Phase 6: 9 obstacles @ V={V_MEDIUM} m/s (medium)")
        self.get_logger().info(f"  Phase 7: 15 obstacles @ V={V_FULL} m/s (full)")
        self.get_logger().info("=" * 60)

    def _on_phase(self, msg: Int32):
        new = msg.data
        if new == self._phase:
            return
        old_phase = self._phase
        self._phase = new
        self._v_peak = PHASE_SPEEDS.get(new, 0.0)

        self.get_logger().info("=" * 60)
        if new <= 4:
            self.get_logger().info(f"  PHASE {new} — STATIC STAGE (all obstacles FROZEN)")
            self.get_logger().info(f"  Robot learns navigation without moving obstacles")
        elif new == 5:
            self.get_logger().info(f"  PHASE 5 — SLOW DYNAMICS")
            self.get_logger().info(f"  6 perimeter obstacles @ V={V_SLOW} m/s")
            self.get_logger().info(f"  (44% of robot speed — easy to outrun)")
        elif new == 6:
            self.get_logger().info(f"  PHASE 6 — MEDIUM DYNAMICS")
            self.get_logger().info(f"  9 obstacles (6 perim + 3 center) @ V={V_MEDIUM} m/s")
            self.get_logger().info(f"  (65% of robot speed — requires attention)")
        elif new >= 7:
            self.get_logger().info(f"  PHASE 7 — FULL DYNAMICS")
            self.get_logger().info(f"  15 obstacles (ALL) @ V={V_FULL} m/s")
            self.get_logger().info(f"  (70% of robot speed — maximum challenge)")
        self.get_logger().info("=" * 60)

        # Stop all obstacles when transitioning to static phase
        if new <= 4:
            self._stop_all_obstacles()

    def _stop_all_obstacles(self):
        """Send zero velocity to all obstacles."""
        stop_msg = Twist()
        for name in self._pubs:
            self._pubs[name].publish(stop_msg)

    def _get_active_obstacles(self):
        """Return list of obstacles that should be moving in current phase."""
        if self._phase <= 4:
            return []  # STATIC STAGE - no moving obstacles
        elif self._phase == 5:
            return OBSTACLES_PERIMETER  # 6 perimeter only
        elif self._phase == 6:
            return OBSTACLES_PERIMETER + OBSTACLES_CORRIDORS
        else:  # Phase 7+
            return ALL_OBSTACLES  # Everything moving

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self._t0

        # Get active obstacles for current phase
        active = self._get_active_obstacles()
        active_names = {name for name, *_ in active}

        # Move active obstacles, stop inactive ones
        for name, axis, period, phase_offset in ALL_OBSTACLES:
            msg = Twist()

            if name in active_names and self._v_peak > 0:
                # Calculate sinusoidal velocity
                v = self._v_peak * math.sin(2 * math.pi / period * t + phase_offset)
                if axis == "x":
                    msg.linear.x = v
                else:
                    msg.linear.y = v
            # else: msg stays zero (obstacle frozen)

            self._pubs[name].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
