#!/usr/bin/env python3
"""
gazebo_goals.py  — v7  (curriculum learning)
==============================================
CURRICULUM PHASES:
  Phase 1 — EASY  : Only Ring A (4 goals, 0.45m from spawn)
                     Robot learns "follow the arrow → get reward"
                     Unlocks Phase 2 when rolling success rate >= 40%

  Phase 2 — FULL  : All 12 goals (A + C rings)
                     Robot generalises to all directions and distances

WHY THIS HELPS:
  With random 12-goal sampling, the robot only sees each position
  ~1 in 12 episodes. Success events are rare (7 in 186 eps = 3.7%).
  The policy cannot strongly learn "goal_angle → go there → +50"
  from such a weak signal.

  By starting with 4 nearby goals, the robot hits the success
  signal much more often in early training. Once it reliably
  follows nearby goals, generalising to far goals is easy.
"""

import collections
import random
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String as StringMsg
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

_LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)


# ── Goal positions ────────────────────────────────────────────────────
GOALS_A = [
    ( 0.32,  0.32),   # A1: NE inner gap
    (-0.32,  0.32),   # A2: NW inner gap
    (-0.32, -0.32),   # A3: SW inner gap
    ( 0.32, -0.32),   # A4: SE inner gap
]

GOALS_C = [
    ( 0.32,  1.00),   # C1: NNE corridor
    (-0.32,  1.00),   # C2: NNW corridor
    ( 0.32, -1.00),   # C3: SSE corridor
    (-0.32, -1.00),   # C4: SSW corridor
    ( 1.00,  0.32),   # C5: ENE corridor
    (-1.00,  0.32),   # C6: WNW corridor
    ( 1.00, -0.32),   # C7: ESE corridor
    (-1.00, -0.32),   # C8: WSW corridor
]

PHASE1_WINDOW    = 20    # rolling window size
PHASE1_THRESHOLD = 0.40  # 40% over 20 eps → unlock Phase 2


# ── SDF models ───────────────────────────────────────────────────────
SPHERE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><sphere><radius>0.15</radius></sphere></geometry>
        <material>
          <ambient>0 0.9 0 1</ambient>
          <diffuse>0 0.9 0 1</diffuse>
          <emissive>0 0.6 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

RING_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder><radius>{radius}</radius><length>0.025</length></cylinder>
        </geometry>
        <material>
          <ambient>1 0.9 0 1</ambient>
          <diffuse>1 0.9 0 1</diffuse>
          <emissive>0.7 0.6 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

POLE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.04</radius><length>1.4</length></cylinder>
        </geometry>
        <material>
          <ambient>0 1 0.1 1</ambient>
          <diffuse>0 1 0.1 1</diffuse>
          <emissive>0 0.7 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


class GazeboGoals(Node):
    def __init__(self):
        super().__init__('tb3_drl_gazebo_goals')

        self.declare_parameter('republish_sec', 3.0)
        self.declare_parameter('goal_radius',   0.35)
        self.declare_parameter('seed',          123)
        self.declare_parameter('start_phase',   1)   # 1=curriculum, 2=all goals from start (for DQN)

        seed = int(self.get_parameter('seed').value)
        if seed >= 0:
            random.seed(seed)

        self.goal_radius = float(self.get_parameter('goal_radius').value)
        start_phase      = int(self.get_parameter('start_phase').value)

        # ── Curriculum state ─────────────────────────────────────────
        self.phase         = max(1, min(3, start_phase))
        self.result_window = collections.deque(maxlen=PHASE1_WINDOW)
        self.ep_done       = False
        self.total_eps     = 0
        self.total_succs   = 0

        # ── Publishers / Subscribers ──────────────────────────────────
        self.goal_pub      = self.create_publisher(PoseStamped, '/tb3_drl/goal_pose', 10)
        self.goal_str_pub  = self.create_publisher(StringMsg, '/tb3_drl/goal', _LATCHED)
        self.need_sub  = self.create_subscription(
            Bool, '/tb3_drl/need_goal', self.on_need_goal, 10)
        self.step_sub  = self.create_subscription(
            Float32MultiArray, '/tb3_drl/step_result',
            self.on_step_result, 10)

        # ── Gazebo markers ────────────────────────────────────────────
        self.spawn_cli  = self.create_client(SpawnEntity,  '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')
        self._gz_ok     = self.spawn_cli.wait_for_service(timeout_sec=3.0)

        self.last_goal      = None
        self._prev_idx      = -1
        self._generation    = 0
        self._pending_x     = None
        self._pending_y     = None
        self._markers_exist = False

        self.timer = self.create_timer(
            float(self.get_parameter('republish_sec').value), self.republish)

        self.get_logger().info("=" * 55)
        if self.phase == 1:
            self.get_logger().info("  CURRICULUM — Phase 1 (EASY: Ring A only)")
            self.get_logger().info(f"  4 goals, 0.45m from spawn")
            self.get_logger().info(f"  Unlocks Phase 2 at >={int(PHASE1_THRESHOLD*100)}% "
                                   f"success over {PHASE1_WINDOW} eps")
            for i, (x, y) in enumerate(GOALS_A):
                self.get_logger().info(f"  A{i+1}: ({x:+.2f}, {y:+.2f})")
        elif self.phase == 3:
            self.get_logger().info("  HARD GOALS — Phase 3 (Ring C only)")
            self.get_logger().info(f"  8 goals, 1.05m from spawn")
            self.get_logger().info(f"  DQN will focus on its weakest points")
            for i, (x, y) in enumerate(GOALS_C):
                self.get_logger().info(f"  C{i+1}: ({x:+.2f}, {y:+.2f})")
        else:
            self.get_logger().info("  ALL GOALS — Phase 2 from start (DQN / ablation mode)")
            self.get_logger().info(f"  12 verified safe positions: Ring A + Ring C")
            self.get_logger().info(f"  No curriculum — all positions active immediately")
            for i, (x, y) in enumerate(GOALS_A + GOALS_C):
                label = f"A{i+1}" if i < 4 else f"C{i-3}"
                self.get_logger().info(f"  {label}: ({x:+.2f}, {y:+.2f})")
        self.get_logger().info("=" * 55)

        self.publish_next_goal()

    # ── Curriculum tracking ───────────────────────────────────────────

    def on_step_result(self, msg: Float32MultiArray):
        # environment_ppo format: obs(54) + [reward, done, info/success]
        # done is at index -2, success is at index -1
        if len(msg.data) < 3:
            return
        done    = msg.data[-2] > 0.5
        success = msg.data[-1] == 1.0  # info==1 means goal reached (not collision=2)

        if done and not self.ep_done:
            self.ep_done = True
            self.total_eps  += 1
            self.total_succs += int(success)
            self.result_window.append(1 if success else 0)

            rate = sum(self.result_window) / len(self.result_window) if self.result_window else 0.0
            # Log progress every 5 episodes during phase 1
            if self.phase == 1 and self.total_eps % 5 == 0:
                self.get_logger().info(
                    f"  [Curriculum] ep={self.total_eps}  "
                    f"window_sr={rate*100:.0f}%  "
                    f"total_succ={self.total_succs}")

            if self.phase == 1 and len(self.result_window) >= PHASE1_WINDOW and rate >= PHASE1_THRESHOLD:
                self._unlock_phase2(rate)

    def _unlock_phase2(self, rate: float):
        self.phase = 2
        self.result_window.clear()
        self._prev_idx = -1
        self.get_logger().info("=" * 55)
        self.get_logger().info(f"  ★★★ PHASE 2 UNLOCKED! ★★★")
        self.get_logger().info(f"  Success rate = {rate*100:.0f}% over {PHASE1_WINDOW} eps")
        self.get_logger().info(f"  Now using all 12 goals (A + C rings)")
        self.get_logger().info(f"  Total episodes: {self.total_eps}  "
                               f"Total successes: {self.total_succs}")
        self.get_logger().info("=" * 55)

    # ── Goal selection ────────────────────────────────────────────────

    def on_need_goal(self, msg: Bool):
        if msg.data:
            self.ep_done = False
            self.publish_next_goal()

    def publish_next_goal(self):
        if self.phase == 1:
            pool = GOALS_A
        elif self.phase == 3:
            pool = GOALS_C
        else:
            pool = GOALS_A + GOALS_C
            
        n     = len(pool)
        cands = [i for i in range(n) if i != self._prev_idx]
        idx   = random.choice(cands)
        self._prev_idx = idx

        gx, gy = pool[idx]
        if self.phase == 1:
            label = f"A{idx+1}"
        elif self.phase == 3:
            label = f"C{idx+1}"
        else:
            label = f"A{idx+1}" if pool is GOALS_A else (f"A{idx+1}" if idx < 4 else f"C{idx-3}")

        sr_str = ""
        if self.result_window:
            sr = sum(self.result_window) / len(self.result_window)
            sr_str = f"  [{len(self.result_window)}/{PHASE1_WINDOW} eps, sr={sr*100:.0f}%]"

        msg = PoseStamped()
        msg.header.frame_id    = 'odom'
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.pose.position.x    = gx
        msg.pose.position.y    = gy
        msg.pose.position.z    = 0.0
        msg.pose.orientation.w = 1.0

        self.last_goal = msg
        self.goal_pub.publish(msg)
        self.goal_str_pub.publish(StringMsg(data=f"{gx:.4f},{gy:.4f}"))
        self.get_logger().info(
            f"[Ph{self.phase}] Goal [{label}]: x={gx:+.2f}  y={gy:+.2f}{sr_str}")
        self._place_markers(gx, gy)

    def republish(self):
        if self.last_goal is None:
            return
        self.last_goal.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(self.last_goal)
        gx = self.last_goal.pose.position.x
        gy = self.last_goal.pose.position.y
        self.goal_str_pub.publish(StringMsg(data=f"{gx:.4f},{gy:.4f}"))

    # ── Marker pipeline ───────────────────────────────────────────────

    def _place_markers(self, x: float, y: float):
        if not self._gz_ok:
            return
        self._generation += 1
        gen = self._generation
        self._pending_x = x
        self._pending_y = y
        if not self._markers_exist:
            self._spawn_all(gen, x, y)
            self._markers_exist = True
            return
        self._delete_one("goal_sphere", lambda: self._after_del_sphere(gen))

    def _after_del_sphere(self, gen):
        self._delete_one("goal_ring", lambda: self._after_del_ring(gen))

    def _after_del_ring(self, gen):
        self._delete_one("goal_pole", lambda: self._after_del_pole(gen))

    def _after_del_pole(self, gen):
        if gen == self._generation:
            self._spawn_all(gen, self._pending_x, self._pending_y)

    def _delete_one(self, name, callback):
        req = DeleteEntity.Request()
        req.name = name
        self.delete_cli.call_async(req).add_done_callback(lambda f: callback())

    def _spawn_one(self, name, sdf, x, y, z):
        req = SpawnEntity.Request()
        req.name = name
        req.xml  = sdf
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        req.reference_frame = "world"
        self.spawn_cli.call_async(req)

    def _spawn_all(self, gen, x, y):
        if gen != self._generation:
            return
        self._spawn_one("goal_sphere",
                        SPHERE_SDF.format(name="goal_sphere"), x, y, 0.20)
        self._spawn_one("goal_ring",
                        RING_SDF.format(name="goal_ring", radius=self.goal_radius),
                        x, y, 0.012)
        self._spawn_one("goal_pole",
                        POLE_SDF.format(name="goal_pole"), x, y, 0.70)
        self.get_logger().info(f"Markers → x={x:.2f}  y={y:.2f}")


def main():
    rclpy.init()
    node = GazeboGoals()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
