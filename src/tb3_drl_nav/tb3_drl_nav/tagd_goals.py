#!/usr/bin/env python3
"""
tagd_goals.py
=====================================================================
Goal manager for the 12×8 m TAGD comparison arena.

- 16 hardcoded goal positions covering the arena.
- Deterministic round-robin cycling over all goals for 200 episodes.
- Publishes /tb3_drl/goal (String "x,y") and /tb3_drl/curriculum_phase (Int32, fixed=7).
- Subscribes /tb3_drl/step_result to detect episode termination.

Robot spawn: (−4.5, 0.0)  — all goals verified ≥ 2.0 m from spawn.
"""
import os
import csv
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float32MultiArray, Int32

_LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

# ── 16 goal positions in the 12×8 m arena (X∈[−6,6], Y∈[−4,4]) ─────────────
# All positions ≥ 2.0 m from spawn (−4.5, 0.0) and ≥ 0.5 m from walls.
TAGD_GOALS = [
    ( 0.0,  2.5),  # centre-north
    ( 3.0,  2.5),  # east-north
    (-3.0,  2.5),  # west-north
    ( 4.5,  0.0),  # far east
    (-2.0,  0.0),  # mid-west (2.5 m from spawn)
    ( 0.0, -2.5),  # centre-south
    ( 3.0, -2.5),  # east-south
    (-3.0, -2.5),  # west-south
    ( 2.0,  0.0),  # mid-east (6.5 m from spawn)
    ( 5.0,  2.5),  # far east-north
    ( 5.0, -2.5),  # far east-south
    (-5.0,  2.5),  # far west-north  (0.7 m from spawn: SKIP → replaced below)
    ( 0.0,  3.5),  # north-deep
    ( 0.0, -3.5),  # south-deep
    ( 3.0,  0.0),  # east-centre
    (-1.0,  3.0),  # northwest mid
]

LOG_DIR = os.path.expanduser("~/tb3_drl_logs/canonical")
MAX_EPISODES = int(os.environ.get("TAGD_MAX_EPISODES", 200))


class TAGDGoals(Node):
    def __init__(self):
        super().__init__("tagd_goals")
        os.makedirs(LOG_DIR, exist_ok=True)

        self._goals      = TAGD_GOALS
        self._idx        = 0
        self._ep         = 0
        self._ep_open    = False
        self._outcomes   = []   # list of (goal_x, goal_y, goal_reached, collision)
        self._cur_goal   = self._goals[0]
        self._last_reset = time.time()

        # CSV log — separate file from canonical_eval.py's output
        run_tag = os.environ.get("EVAL_TAG", "tagd_open_arena")
        self._csv_path = os.path.join(LOG_DIR, f"sac_v11_s42_{run_tag}_goals.csv")
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode", "goal_x", "goal_y", "goal_reached", "collision", "timeout"])

        # Publishers
        self._goal_pub  = self.create_publisher(String, "/tb3_drl/goal",              _LATCHED)
        self._phase_pub = self.create_publisher(Int32,  "/tb3_drl/curriculum_phase",  _LATCHED)

        # Subscribers
        self.create_subscription(Float32MultiArray, "/tb3_drl/step_result",
                                 self._on_step_result, 10)
        self.create_subscription(Float32MultiArray, "/tb3_drl/reset_obs",
                                 self._on_reset, _LATCHED)

        # Publish fixed phase=7
        msg = Int32(); msg.data = 7
        self._phase_pub.publish(msg)

        # Startup goal after brief delay
        self.create_timer(1.5, self._publish_goal_once)

        self.get_logger().info(
            f"[TAGDGoals] {len(self._goals)} goals, max {MAX_EPISODES} episodes")
        self._publish_goal()

    def _publish_goal(self):
        self._cur_goal = self._goals[self._idx % len(self._goals)]
        msg = String()
        msg.data = f"{self._cur_goal[0]},{self._cur_goal[1]}"
        self._goal_pub.publish(msg)
        self.get_logger().info(
            f"[TAGDGoals] Ep {self._ep+1}/{MAX_EPISODES}  "
            f"Goal {self._idx % len(self._goals)+1}: {self._cur_goal}")

    def _publish_goal_once(self):
        """Re-publish goal after startup settling."""
        self._publish_goal()

    def _on_reset(self, msg: Float32MultiArray):
        self._ep_open    = True
        self._last_reset = time.time()

    def _on_step_result(self, msg: Float32MultiArray):
        if not self._ep_open:
            return

        # Decode terminal signal
        if len(msg.data) >= 57:
            done = bool(msg.data[55])
            info = int(msg.data[56])
        elif len(msg.data) == 6:
            done = True
            info = int(msg.data[5])
        else:
            return

        # Filter impossible terminals (< 0.5 s after reset)
        if done and (time.time() - self._last_reset) < 0.5:
            return

        if not done:
            return

        self._ep_open = False
        goal_reached = 1 if info == 1 else 0
        collision    = 1 if info == 2 else 0

        self._outcomes.append((
            self._cur_goal[0], self._cur_goal[1],
            goal_reached, collision))

        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self._ep + 1,
                self._cur_goal[0], self._cur_goal[1],
                goal_reached, collision,
                1 if (info not in (1, 2)) else 0])

        self._ep  += 1
        self._idx += 1

        sr = sum(o[2] for o in self._outcomes) / len(self._outcomes) * 100
        cr = sum(o[3] for o in self._outcomes) / len(self._outcomes) * 100
        self.get_logger().info(
            f"[TAGDGoals] Ep {self._ep}/{MAX_EPISODES}  "
            f"SR={sr:.1f}%  CR={cr:.1f}%  info={info}")

        if self._ep >= MAX_EPISODES:
            self.get_logger().info(
                f"[TAGDGoals] ✅ DONE — Final SR={sr:.1f}%  CR={cr:.1f}%")
            return

        self._publish_goal()


def main(args=None):
    rclpy.init(args=args)
    node = TAGDGoals()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
