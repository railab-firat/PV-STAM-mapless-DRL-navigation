#!/usr/bin/env python3
"""
benchmark_goals.py  — Goal manager for cross-environment benchmark evaluation
==============================================================================
Publishes goals appropriate for the standard ROBOTIS 5×5m worlds
(turtlebot3_dqn_stage3, turtlebot3_dqn_stage4, turtlebot3_world).

These worlds have outer walls at ±2.5m. The robot spawns at (0, 0).
Goals are hand-verified navigable positions that avoid walls and static
obstacles across all three standard benchmark environments.

Ring B (near):  ~1.0m from origin — easy goals
Ring D (far):   ~1.8m from origin — harder goals
"""

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

# ── Goal positions for 5×5m ROBOTIS worlds ─────────────────────────────
# All positions verified clear of walls in stage3, stage4, and tb3_world.
# Outer walls at ~±2.5m; static obstacles vary by world but these spots
# are in open corridors shared by all three.

GOALS_NEAR = [
    ( 0.7,  0.7),   # B01: NE
    (-0.7,  0.7),   # B02: NW
    (-0.7, -0.7),   # B03: SW
    ( 0.7, -0.7),   # B04: SE
    ( 0.0,  1.0),   # B05: N
    ( 0.0, -1.0),   # B06: S
    ( 1.0,  0.0),   # B07: E
    (-1.0,  0.0),   # B08: W
]

GOALS_FAR = [
    ( 1.5,  1.5),   # D01: far NE
    (-1.5,  1.5),   # D02: far NW
    (-1.5, -1.5),   # D03: far SW
    ( 1.5, -1.5),   # D04: far SE
    ( 0.0,  1.8),   # D05: far N
    ( 0.0, -1.8),   # D06: far S
    ( 1.8,  0.0),   # D07: far E
    (-1.8,  0.0),   # D08: far W
]

ALL_GOALS = GOALS_NEAR + GOALS_FAR

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


class BenchmarkGoals(Node):
    def __init__(self):
        super().__init__('tb3_drl_gazebo_goals')

        self.declare_parameter('republish_sec', 3.0)
        self.declare_parameter('goal_radius', 0.45)
        self.declare_parameter('seed', 123)

        seed = int(self.get_parameter('seed').value)
        if seed >= 0:
            random.seed(seed)

        self.goal_radius = float(self.get_parameter('goal_radius').value)

        # State
        self.ep_done = False
        self.total_eps = 0
        self.total_succs = 0
        self._prev_idx = -1
        self._generation = 0
        self._pending_x = None
        self._pending_y = None
        self._markers_exist = False

        # Publishers / Subscribers
        self.goal_pub = self.create_publisher(PoseStamped, '/tb3_drl/goal_pose', 10)
        self.goal_str_pub = self.create_publisher(StringMsg, '/tb3_drl/goal', _LATCHED)
        self.need_sub = self.create_subscription(
            Bool, '/tb3_drl/need_goal', self.on_need_goal, 10)
        self.step_sub = self.create_subscription(
            Float32MultiArray, '/tb3_drl/step_result',
            self.on_step_result, 10)

        # Gazebo markers
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')
        self._gz_ok = self.spawn_cli.wait_for_service(timeout_sec=3.0)

        self.last_goal = None

        self.timer = self.create_timer(
            float(self.get_parameter('republish_sec').value), self.republish)

        self.get_logger().info("=" * 60)
        self.get_logger().info("  BENCHMARK GOALS — 16 positions for 5×5m ROBOTIS worlds")
        self.get_logger().info(f"  Near ring (8 goals, ~1.0m):")
        for i, (x, y) in enumerate(GOALS_NEAR):
            self.get_logger().info(f"    B{i+1:02d}: ({x:+.1f}, {y:+.1f})")
        self.get_logger().info(f"  Far ring (8 goals, ~1.8m):")
        for i, (x, y) in enumerate(GOALS_FAR):
            self.get_logger().info(f"    D{i+1:02d}: ({x:+.1f}, {y:+.1f})")
        self.get_logger().info("=" * 60)

        self.publish_next_goal()

    def on_step_result(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return
        done = msg.data[-2] > 0.5
        success = msg.data[-1] == 1.0  # info==1 means goal reached (not collision=2)

        if done and not self.ep_done:
            self.ep_done = True
            self.total_eps += 1
            self.total_succs += int(success)

    def on_need_goal(self, msg: Bool):
        if msg.data:
            self.ep_done = False
            self.publish_next_goal()

    def publish_next_goal(self):
        pool = ALL_GOALS
        n = len(pool)
        cands = [i for i in range(n) if i != self._prev_idx]
        idx = random.choice(cands)
        self._prev_idx = idx

        gx, gy = pool[idx]
        label = f"B{idx+1:02d}" if idx < len(GOALS_NEAR) else f"D{idx - len(GOALS_NEAR) + 1:02d}"

        msg = PoseStamped()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        self.last_goal = msg
        self.goal_pub.publish(msg)
        self.goal_str_pub.publish(StringMsg(data=f"{gx:.4f},{gy:.4f}"))
        sr = (self.total_succs / self.total_eps * 100) if self.total_eps > 0 else 0.0
        self.get_logger().info(
            f"Goal [{label}]: x={gx:+.2f}  y={gy:+.2f}  "
            f"(ep={self.total_eps}, sr={sr:.0f}%)")
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
        req.xml = sdf
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
    node = BenchmarkGoals()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
