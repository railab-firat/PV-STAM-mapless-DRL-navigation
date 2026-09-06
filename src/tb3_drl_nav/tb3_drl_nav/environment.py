#!/usr/bin/env python3
import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, Bool
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty


def yaw_from_quat(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class TB3DrlEnv(Node):
    def __init__(self):
        super().__init__("tb3_drl_environment")

        # Timing / obs
        self.declare_parameter("dt", 0.2)
        self.declare_parameter("settle_sec", 1.0)
        self.declare_parameter("n_sectors", 24)
        self.declare_parameter("max_range", 3.5)

        # Termination
        self.declare_parameter("collision_dist", 0.18)
        self.declare_parameter("goal_radius", 0.25)
        self.declare_parameter("goal_dist_norm", 3.0)
        self.declare_parameter("max_steps_env", 600)

        # Reset robustness
        self.declare_parameter("reset_msg_timeout", 8.0)
        self.declare_parameter("reset_collision_margin", 0.05)
        self.declare_parameter("reset_goal_margin", 0.10)
        self.declare_parameter("reset_retry_limit", 30)

        # Reward shaping (key improvements)
        self.declare_parameter("time_penalty", 0.01)         # per-step negative
        self.declare_parameter("progress_gain", 5.0)         # reward for reducing goal distance
        self.declare_parameter("safe_dist", 0.35)            # start penalizing below this lidar min
        self.declare_parameter("safe_gain", 1.5)             # proximity penalty gain
        self.declare_parameter("turn_penalty", 0.02)         # penalize pure turning actions
        self.declare_parameter("jerk_penalty", 0.01)         # penalize action switching
        self.declare_parameter("timeout_penalty", 30.0)      # penalty if hit max_steps_env
        self.declare_parameter("stuck_window", 25)           # steps window
        self.declare_parameter("stuck_eps", 0.05)            # if max-min dist < this => stuck
        self.declare_parameter("stuck_penalty", 25.0)

        self.dt = float(self.get_parameter("dt").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.n_sectors = int(self.get_parameter("n_sectors").value)
        self.max_range = float(self.get_parameter("max_range").value)

        self.collision_dist = float(self.get_parameter("collision_dist").value)
        self.goal_radius = float(self.get_parameter("goal_radius").value)
        self.goal_dist_norm = float(self.get_parameter("goal_dist_norm").value)
        self.max_steps_env = int(self.get_parameter("max_steps_env").value)

        self.reset_msg_timeout = float(self.get_parameter("reset_msg_timeout").value)
        self.reset_collision_margin = float(self.get_parameter("reset_collision_margin").value)
        self.reset_goal_margin = float(self.get_parameter("reset_goal_margin").value)
        self.reset_retry_limit = int(self.get_parameter("reset_retry_limit").value)

        self.time_penalty = float(self.get_parameter("time_penalty").value)
        self.progress_gain = float(self.get_parameter("progress_gain").value)
        self.safe_dist = float(self.get_parameter("safe_dist").value)
        self.safe_gain = float(self.get_parameter("safe_gain").value)
        self.turn_penalty = float(self.get_parameter("turn_penalty").value)
        self.jerk_penalty = float(self.get_parameter("jerk_penalty").value)
        self.timeout_penalty = float(self.get_parameter("timeout_penalty").value)
        self.stuck_window = int(self.get_parameter("stuck_window").value)
        self.stuck_eps = float(self.get_parameter("stuck_eps").value)
        self.stuck_penalty = float(self.get_parameter("stuck_penalty").value)

        # Pub/Sub
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.reset_obs_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/reset_obs", 10)
        self.step_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/step_result", 10)
        self.need_goal_pub = self.create_publisher(Bool, "/tb3_drl/need_goal", 10)

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.goal_sub = self.create_subscription(PoseStamped, "/tb3_drl/goal", self.on_goal, 10)
        self.action_sub = self.create_subscription(Int32, "/tb3_drl/action", self.on_action, 10)

        self.reset_srv = self.create_service(Empty, "/tb3_drl/reset", self.on_reset_srv)
        self.reset_client = self.create_client(Empty, "/reset_simulation")

        # State
        self.scan = None
        self.odom = None
        self.goal = None

        # Counters for freshness
        self.scan_seq = 0
        self.odom_seq = 0
        self.reset_scan_seq = 0
        self.reset_odom_seq = 0

        self.episode_id = 0
        self.step_id = 0

        self.pending_action = None
        self.step_active = False
        self.step_start_t = 0.0
        self.prev_goal_dist = None
        self.last_action = 0
        self.prev_action = None

        # Stuck detector
        self.dist_hist = deque(maxlen=self.stuck_window)

        # Reset state machine
        self.reset_requested = False
        self.reset_state = "IDLE"
        self.reset_future = None
        self.reset_t0 = 0.0
        self.reset_deadline_t = 0.0
        self.reset_retry_count = 0

        self.timer = self.create_timer(0.02, self.loop)
        self.get_logger().info("Environment node ready. Waiting for /scan, /odom, /tb3_drl/goal...")

    def on_scan(self, msg: LaserScan):
        self.scan = msg
        self.scan_seq += 1

    def on_odom(self, msg: Odometry):
        self.odom = msg
        self.odom_seq += 1

    def on_goal(self, msg: PoseStamped):
        self.goal = msg

    def on_action(self, msg: Int32):
        if self.pending_action is None and (self.reset_state == "IDLE") and not self.reset_requested:
            self.pending_action = int(msg.data)

    def on_reset_srv(self, req, res):
        self.reset_requested = True
        self.need_goal_pub.publish(Bool(data=True))
        return res

    def publish_zero(self):
        self.cmd_pub.publish(Twist())

    def action_to_twist(self, a: int) -> Twist:
        tw = Twist()

        # Base speeds (tunable)
        fwd = 0.12
        fwd_turn_soft = 0.08
        fwd_turn_hard = 0.05

        turn_soft = 0.7
        turn_hard = 1.2

        # 5 discrete actions (no pure spin)
        if a == 0:              # straight
            tw.linear.x = fwd
        elif a == 1:            # soft left
            tw.linear.x = fwd_turn_soft
            tw.angular.z = turn_soft
        elif a == 2:            # soft right
            tw.linear.x = fwd_turn_soft
            tw.angular.z = -turn_soft
        elif a == 3:            # hard left
            tw.linear.x = fwd_turn_hard
            tw.angular.z = turn_hard
        elif a == 4:            # hard right
            tw.linear.x = fwd_turn_hard
            tw.angular.z = -turn_hard

        return tw

    def get_lidar_sectors(self):
        if self.scan is None or not self.scan.ranges:
            return [1.0] * self.n_sectors, self.max_range

        ranges = list(self.scan.ranges)
        n = len(ranges)
        sec = self.n_sectors
        step = max(1, n // sec)

        mins = []
        global_min = self.max_range

        for i in range(sec):
            s = i * step
            e = n if i == sec - 1 else (i + 1) * step
            chunk = ranges[s:e]
            vals = [x for x in chunk if (x is not None and x > 0.01 and x < self.max_range)]
            m = min(vals) if vals else self.max_range
            global_min = min(global_min, m)
            mins.append(max(0.0, min(1.0, m / self.max_range)))

        return mins, global_min

    def get_goal_features(self):
        if self.odom is None or self.goal is None:
            return 1.0, 0.0, None

        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        gx = self.goal.pose.position.x
        gy = self.goal.pose.position.y

        dx = gx - rx
        dy = gy - ry
        dist = math.sqrt(dx * dx + dy * dy)

        ang = math.atan2(dy, dx)
        rel = wrap_pi(ang - yaw)

        dist_n = max(0.0, min(1.0, dist / self.goal_dist_norm))
        ang_n = max(-1.0, min(1.0, rel / math.pi))
        return dist_n, ang_n, dist

    def build_obs(self):
        lidar, min_raw = self.get_lidar_sectors()
        dist_n, ang_n, dist_raw = self.get_goal_features()
        obs = lidar + [dist_n, ang_n]
        return obs, min_raw, dist_raw

    def publish_reset_obs(self):
        obs, _, dist_raw = self.build_obs()
        self.prev_goal_dist = dist_raw
        self.dist_hist.clear()
        if dist_raw is not None:
            self.dist_hist.append(dist_raw)
        self.prev_action = None

        msg = Float32MultiArray()
        msg.data = [float(self.episode_id)] + [float(x) for x in obs]
        self.reset_obs_pub.publish(msg)

    def publish_step(self, reward, done, success):
        obs, _, dist_raw = self.build_obs()
        msg = Float32MultiArray()
        msg.data = [
            float(self.episode_id),
            float(self.step_id),
            float(reward),
            1.0 if done else 0.0,
            1.0 if success else 0.0,
        ] + [float(x) for x in obs]
        self.step_pub.publish(msg)
        self.prev_goal_dist = dist_raw
        if dist_raw is not None:
            self.dist_hist.append(dist_raw)

    def _start_reset(self):
        self.step_active = False
        self.pending_action = None
        self.publish_zero()

        self.reset_retry_count += 1
        if self.reset_retry_count > self.reset_retry_limit:
            self.get_logger().error("Reset retry limit exceeded. Check spawn/goal/world.")
            self.reset_state = "IDLE"
            self.reset_retry_count = 0
            return

        self.reset_state = "WAIT_SERVICE"

    def loop(self):
        now = time.time()

        # RESET pipeline
        if self.reset_requested and self.reset_state == "IDLE":
            self.reset_requested = False
            self.reset_retry_count = 0
            self._start_reset()

        if self.reset_state == "WAIT_SERVICE":
            if self.reset_client.wait_for_service(timeout_sec=0.0):
                self.reset_future = self.reset_client.call_async(Empty.Request())
                self.reset_state = "CALLING"
            return

        if self.reset_state == "CALLING":
            if self.reset_future is not None and self.reset_future.done():
                self.reset_t0 = now
                self.reset_deadline_t = now + self.reset_msg_timeout
                self.reset_scan_seq = self.scan_seq
                self.reset_odom_seq = self.odom_seq
                self.reset_state = "SETTLE"
            return

        if self.reset_state == "SETTLE":
            if (now - self.reset_t0) < self.settle_sec:
                return

            if self.scan is None or self.odom is None or self.goal is None:
                if now > self.reset_deadline_t:
                    self.get_logger().warn("Reset timeout waiting for scan/odom/goal. Retrying reset.")
                    self._start_reset()
                return

            if (self.scan_seq <= self.reset_scan_seq) or (self.odom_seq <= self.reset_odom_seq):
                if now > self.reset_deadline_t:
                    self.get_logger().warn("No fresh scan/odom received after reset. Retrying reset.")
                    self._start_reset()
                return

            _, min_raw, dist_raw = self.build_obs()
            start_collision = (min_raw is not None and min_raw < (self.collision_dist + self.reset_collision_margin))
            start_goal = (dist_raw is not None and dist_raw < (self.goal_radius + self.reset_goal_margin))

            if start_collision:
                self.get_logger().warn(f"Invalid reset: start collision min_raw={min_raw:.3f}. Retrying reset.")
                self._start_reset()
                return

            if start_goal:
                self.get_logger().warn(f"Invalid reset: goal too close dist={dist_raw:.3f}. Requesting new goal.")
                self.need_goal_pub.publish(Bool(data=True))
                return

            self.episode_id += 1
            self.step_id = 0
            self.publish_reset_obs()
            self.reset_state = "IDLE"
            return

        # STEP pipeline
        if self.step_active:
            if (now - self.step_start_t) >= self.dt:
                self.publish_zero()

                _, min_raw, dist_raw = self.build_obs()
                collision = (min_raw is not None and min_raw < self.collision_dist)
                success = (dist_raw is not None and dist_raw < self.goal_radius)

                # default
                done = False
                reward = -self.time_penalty  # IMPORTANT: time penalty (no positive living reward)

                # progress reward
                if self.prev_goal_dist is not None and dist_raw is not None:
                    reward += (self.prev_goal_dist - dist_raw) * self.progress_gain

                # proximity penalty (teach safety BEFORE collision)
                if min_raw is not None and min_raw < self.safe_dist:
                    reward -= (self.safe_dist - min_raw) * self.safe_gain

                # turning + jerk penalties (reduce oscillations)
                if self.last_action in (3, 4):
                    reward -= self.turn_penalty
                if self.prev_action is not None and self.last_action != self.prev_action:
                    reward -= self.jerk_penalty

                # Anti-spin shaping (prevents turn-in-place local optimum)
                tw_cmd = self.action_to_twist(self.last_action)
                reward -= 0.03 * abs(tw_cmd.angular.z)
                if abs(tw_cmd.angular.z) > 0.8 and tw_cmd.linear.x < 0.05:
                    reward -= 0.05
                self.prev_action = self.last_action

                # stuck detection
                stuck = False
                if len(self.dist_hist) == self.dist_hist.maxlen:
                    if (max(self.dist_hist) - min(self.dist_hist)) < self.stuck_eps:
                        stuck = True

                # timeout detection (environment-side)
                timeout = (self.step_id >= self.max_steps_env)

                # terminal overrides
                if collision:
                    done = True
                    reward = -100.0
                elif success:
                    done = True
                    reward = 200.0
                elif stuck:
                    done = True
                    reward = -self.stuck_penalty
                elif timeout:
                    done = True
                    reward = -self.timeout_penalty

                self.publish_step(reward, done, success)

                if success:
                    self.need_goal_pub.publish(Bool(data=True))

                self.step_active = False
                self.pending_action = None
            return

        if self.pending_action is not None:
            if self.scan is None or self.odom is None or self.goal is None:
                return
            self.last_action = int(self.pending_action)
            self.step_id += 1
            self.cmd_pub.publish(self.action_to_twist(self.last_action))
            self.step_start_t = now
            self.step_active = True


def main():
    rclpy.init()
    node = TB3DrlEnv()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
