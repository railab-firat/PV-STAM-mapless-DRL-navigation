#!/usr/bin/env python3
"""
environment_ppo.py  Phase 3  v9  (clean rewrite)
=====================================================================
Velocity-aware observation for SAC training.

Observation (54 dims, no frame stacking):
  scan           (24) — normalised LiDAR ranges [0, 1]
  scan_velocity  (24) — (current − previous) scan, clipped [−1, 1]
  goal_dist       (1) — normalised distance to goal (d/8.0, capped 1.0)
  goal_angle      (1) — normalised heading error (angle/π)
  own_lin_vel     (1) — actual linear velocity from odom (v/0.26)
  own_ang_vel     (1) — actual angular velocity from odom (ω/1.82)
  min_scan        (1) — closest obstacle reading
  min_approach    (1) — fastest approaching direction

DZ_MODE controls observation preprocessing ONLY (not rewards):
  baseline    — raw observations
  static_dz   — amplify close LiDAR readings
  velocity_dz — static_dz + amplify approaching obstacle velocities
  stam        — raw observations (learned attention in the network)
"""
import math, os, time, rclpy, random
import numpy as np
from collections import deque
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool, Float32MultiArray
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

RESET_OBS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "phase3_ppo.yaml")
try:
    import yaml
    with open(_CFG_PATH) as f:
        _CFG = yaml.safe_load(f) or {}
except Exception:
    _CFG = {}


def _cfg_get(path, default, legacy_key=None):
    cur = _CFG
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            cur = None
            break
        cur = cur[key]
    if cur is not None:
        return cur
    if legacy_key is not None:
        return _CFG.get(legacy_key, default)
    return default

# ── Seeding ───────────────────────────────────────────────────────────────────
SEED = int(os.environ.get("SEED", _CFG.get("seed", 42)))
np.random.seed(SEED); random.seed(SEED)

# ── Real-robot fixes gate ────────────────────────────────────────────────────
# Set REALFIX=true to enable 4 real-hardware fixes (spin smoothness, action
# smoothness, LiDAR noise augmentation, 5Hz sleep). Use for s777 realfix runs.
# Leave unset (default false) for all existing s42/s123 runs — they behave
# exactly as before so the paper multi-seed table stays consistent.
REALFIX = os.environ.get("REALFIX", "false").lower() == "true"

# ── Control-loop wall-clock sleep override ───────────────────────────────────
# The per-step control loop waits `time.sleep(STEP_SLEEP_S)` of WALL-CLOCK time
# between publishing an action and reading the next observation. Because this is
# wall-clock (not sim-time), the simulated distance travelled per RL step equals
#   travel = robot_speed * STEP_SLEEP_S * effective_real_time_factor
# Baselines (v8/v11) trained at RTF=1.0 with STEP_SLEEP_S=0.15 → 0.15 s sim/step.
# When training in a fast (free-run) world at RTF≈N, set STEP_SLEEP_S=0.15/N so
# each RL step still advances 0.15 s of SIM time → the MDP is IDENTICAL to the
# baselines, just N× faster in wall-clock. Example: fast world ≈6× → 0.15/6=0.025.
#   STEP_SLEEP_S=0.025 ros2 launch tb3_drl_nav train_sac_lstm.launch.py ...
# Leave unset to preserve the exact baseline behaviour (0.20 if REALFIX else 0.15).
_STEP_SLEEP_ENV = os.environ.get("STEP_SLEEP_S", "").strip()
STEP_SLEEP_S = float(_STEP_SLEEP_ENV) if _STEP_SLEEP_ENV else (0.20 if REALFIX else 0.15)

# ── Danger Zone observation parameters ───────────────────────────────────
DZ_RADIUS_NORM = float(_cfg_get(("environment", "danger_zone", "radius_norm"), 0.20))
DZ_AMPLIFY     = float(_cfg_get(("environment", "danger_zone", "amplify"), 3.0))
DZ_VEL_AMPLIFY = float(_cfg_get(("environment", "danger_zone", "velocity_amplify"), 2.0))

# ── Reward constants ─────────────────────────────────────────────────────
REWARD_COLLISION = float(_cfg_get(("environment", "rewards", "collision"), -100.0))
REWARD_GOAL      = float(_cfg_get(("environment", "rewards", "goal"), 200.0))
REWARD_STUCK     = float(_cfg_get(("environment", "rewards", "stuck"), -60.0))
REWARD_TIMEOUT   = float(_cfg_get(("environment", "rewards", "timeout"), -80.0))
STEP_PENALTY     = float(_cfg_get(("environment", "rewards", "step_penalty"), -0.02))
PROGRESS_SCALE   = float(_cfg_get(("environment", "rewards", "progress_scale"), 6.0))
HEADING_BONUS    = float(_cfg_get(("environment", "rewards", "heading_bonus"), 0.25))
SPIN_PENALTY_SCALE = float(_cfg_get(("environment", "rewards", "spin_penalty_scale"), 0.30))
CENTER_LOITER_PENALTY = float(_cfg_get(("environment", "rewards", "center_loiter_penalty"), 0.25))
PROXIMITY_DIST   = float(_cfg_get(("environment", "proximity", "baseline_dist"), 0.50))
PROXIMITY_SCALE  = float(_cfg_get(("environment", "proximity", "baseline_scale"), 1.2))
DZ_PROX_DIST     = float(_cfg_get(("environment", "proximity", "danger_zone_dist"), 0.70))
DZ_PROX_SCALE    = float(_cfg_get(("environment", "proximity", "danger_zone_scale"), 1.5))
REVERSE_ENABLED  = bool(_cfg_get(("reverse_motion", "enabled"), False))
REVERSE_ESCAPE_BONUS = float(_cfg_get(("reverse_motion", "reward_escape_bonus"), 0.0))
REVERSE_UNLOCK_SCAN_DIST = float(_cfg_get(("reverse_motion", "unlock_scan_dist"), 0.32))
GOAL_RADIUS      = float(_cfg_get(("environment", "termination", "goal_radius"), 0.45))
COLLISION_DIST   = float(_cfg_get(("environment", "termination", "collision_dist"), 0.20))
MAX_STEPS        = int(_cfg_get(("environment", "termination", "max_steps"), 500))
STUCK_WINDOW     = int(_cfg_get(("environment", "termination", "stuck_window"), 60))
STUCK_THRESH     = float(_cfg_get(("environment", "termination", "stuck_threshold"), 0.10))
STALL_PROGRESS   = float(_cfg_get(("environment", "termination", "stall_progress"), 0.12))
NO_PROGRESS_WINDOW = int(_cfg_get(("environment", "termination", "no_progress_window"), 100))
NO_PROGRESS_MIN_STEPS = int(_cfg_get(("environment", "termination", "no_progress_min_steps"), 180))
NO_PROGRESS_THRESHOLD = float(_cfg_get(("environment", "termination", "no_progress_threshold"), 0.20))
CENTER_LOITER_RADIUS = float(_cfg_get(("environment", "termination", "center_loiter_radius"), 1.8))
FAR_GOAL_MIN_DIST = float(_cfg_get(("environment", "termination", "far_goal_min_dist"), 3.5))
CENTER_LOITER_MIN_STEPS = int(_cfg_get(("environment", "termination", "center_loiter_min_steps"), 90))
CENTER_LOITER_WINDOW = int(_cfg_get(("environment", "termination", "center_loiter_window"), 50))
ENABLE_NO_PROGRESS_TERMINATION = bool(_cfg_get(("environment", "termination", "enable_no_progress_termination"), False))
ENABLE_CENTER_LOITER_TERMINATION = bool(_cfg_get(("environment", "termination", "enable_center_loiter_termination"), False))


class EnvironmentPPO(Node):
    def __init__(self):
        super().__init__("tb3_drl_env_ppo")
        self.declare_parameter("start_x", -2.0)
        self.declare_parameter("start_y", -0.5)
        self._prev_lidar = None
        self._dz_mode = os.environ.get("DZ_MODE", "velocity_dz")
        self._pos_hist = deque(maxlen=STUCK_WINDOW)  # track actual (x,y) positions
        self._dist_hist = deque(maxlen=max(STUCK_WINDOW, NO_PROGRESS_WINDOW))  # track goal-distance progress

        # Publishers
        self._cmd_pub  = self.create_publisher(Twist, "/cmd_vel", 10)
        self._obs_pub  = self.create_publisher(Float32MultiArray, "/tb3_drl/reset_obs", RESET_OBS_QOS)
        self._step_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/step_result", 10)
        self._goal_pub = self.create_publisher(Bool, "/tb3_drl/need_goal", 10)

        # QoS must match publisher durability (TRANSIENT_LOCAL) for /tb3_drl/goal
        _GOAL_QOS = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # Subscribers
        self.create_subscription(LaserScan, "/scan", self._cb_scan, 10)
        self.create_subscription(Odometry, "/odom", self._cb_odom, 10)
        self.create_subscription(String, "/tb3_drl/goal", self._cb_goal, _GOAL_QOS)
        self.create_subscription(Float32MultiArray, "/tb3_drl/action_continuous", self._cb_action, 10)

        # Service clients
        self._reset_cli     = self.create_client(Empty, "/reset_world")
        self._pause_cli     = self.create_client(Empty, "/pause_physics")
        self._unpause_cli   = self.create_client(Empty, "/unpause_physics")
        self._set_state_cli = self.create_client(SetEntityState, "/set_entity_state")

        # Initial Home position (will be refreshed from parameters on reset)
        self._home_x = self.get_parameter("start_x").value
        self._home_y = self.get_parameter("start_y").value

        # State
        self._scan = self._odom = self._goal_xy = None
        self._act_lin = None
        self._act_ang = None
        self._step_id = 0
        self._want_reset = True
        self._rs = "IDLE"
        self._prev_dist = None
        # [REALFIX] previous action for smoothness penalty (Fix 2)
        self._prev_act_lin = 0.0
        self._prev_act_ang = 0.0

        self.get_logger().info(f"[ENV] DZ_MODE={self._dz_mode}  REALFIX={REALFIX}")
        self.create_timer(0.02, self._loop)

    # ── callbacks ─────────────────────────────────────────────────────────
    def _cb_scan(self, msg):
        self._scan = msg

    def _cb_odom(self, msg):
        self._odom = msg

    def _cb_goal(self, msg):
        try:
            x, y = msg.data.split(",")
            x, y = float(x), float(y)
            if math.isfinite(x) and math.isfinite(y):
                self._goal_xy = (x, y)
        except Exception:
            pass

    def _cb_action(self, msg):
        if self._rs == "IDLE" and not self._want_reset:
            self._act_lin = float(msg.data[0])
            self._act_ang = float(msg.data[1])

    # ── observation (54 dims) ─────────────────────────────────────────────
    def _get_obs(self):
        # 1. LiDAR scan → 24 normalised values via min-pooling
        if not self._scan:
            lidar = [1.0] * 24
        else:
            raw = list(self._scan.ranges)
            step = max(1, len(raw) // 24)
            lidar = []
            for i in range(24):
                v = [x for x in raw[i * step:(i + 1) * step] if 0.01 < x < 3.5]
                lidar.append((min(v) if v else 3.5) / 3.5)

        # 2. Scan velocity = frame difference, clipped to [-1, 1]
        if self._prev_lidar is not None:
            scan_vel = [max(-1.0, min(1.0, c - p))
                        for c, p in zip(lidar, self._prev_lidar)]
        else:
            scan_vel = [0.0] * 24
        self._prev_lidar = lidar[:]

        # [REALFIX Fix 3] LiDAR noise augmentation — simulates real LDS-02
        # ±2-3cm noise so the model trains robustly for real hardware.
        # Only applied when REALFIX=true (never affects s42/s123 runs).
        if REALFIX:
            noise = np.random.normal(0, 0.015, size=len(lidar))
            lidar = [max(0.0, min(1.0, v + n)) for v, n in zip(lidar, noise)]
            scan_vel_noise = np.random.normal(0, 0.01, size=len(scan_vel))
            scan_vel = list(np.clip(np.array(scan_vel) + scan_vel_noise, -1.0, 1.0))

        # 3. DZ observation preprocessing (affects observation only, NOT reward)
        if self._dz_mode in ("static_dz", "velocity_dz"):
            for i in range(24):
                if lidar[i] < DZ_RADIUS_NORM:
                    lidar[i] = lidar[i] / DZ_AMPLIFY
        if self._dz_mode == "velocity_dz":
            for i in range(24):
                if lidar[i] < DZ_RADIUS_NORM and scan_vel[i] < 0:
                    scan_vel[i] = max(-1.0, scan_vel[i] * DZ_VEL_AMPLIFY)

        # 4. Navigation state
        dn, an, d = 1.0, 0.0, 999.0
        if self._odom and self._goal_xy:
            p = self._odom.pose.pose.position
            d = math.hypot(self._goal_xy[0] - p.x, self._goal_xy[1] - p.y)
            dn = min(1.0, d / 8.0)
            q = self._odom.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            an = math.atan2(self._goal_xy[1] - p.y,
                            self._goal_xy[0] - p.x) - yaw
            an = math.atan2(math.sin(an), math.cos(an)) / math.pi

        # 5. Summary signals
        min_scan = min(lidar)
        min_approach = min(scan_vel)

        # 6. Slots 50-51: either odometry velocities (original) or previous commanded
        #    action (REALFIX). prev_action is deterministic and lag-free; odom velocity
        #    lags ~1 step on real hardware and picks up wheel-slip noise.
        if REALFIX:
            # [REALFIX Change 1] prev_action in obs — matches tomasvr reference design.
            # _prev_act_lin/_prev_act_ang are already tracked for the smoothness penalty.
            nav_vel_lin = self._prev_act_lin / 0.26
            nav_vel_ang = self._prev_act_ang / 1.82
        else:
            if self._odom:
                nav_vel_lin = self._odom.twist.twist.linear.x / 0.26
                nav_vel_ang = self._odom.twist.twist.angular.z / 1.82
            else:
                nav_vel_lin, nav_vel_ang = 0.0, 0.0

        obs = (lidar                                                     # 24
               + scan_vel                                                # 24
               + [dn, an, nav_vel_lin, nav_vel_ang]                      #  4
               + [min_scan, min_approach])                               #  2 → 54

        # Return raw min distance in metres for reward computation
        raw_min_m = min(self._prev_lidar) * 3.5 if self._prev_lidar else 3.5
        return obs, raw_min_m, d, an

    # ── reward ────────────────────────────────────────────────────────────
    def _compute_reward(self, min_dist_m, d, an):
        """Compute step reward. Same for ALL DZ modes — ablation is observation-only."""
        rew = STEP_PENALTY

        # Progress toward goal
        if self._prev_dist is not None and d < 900:
            delta = max(-0.3, min(0.3, self._prev_dist - d))
            rew += delta * PROGRESS_SCALE

        # Heading bonus — only when moving forward (kills spin exploit)
        fwd = max(0.0, self._act_lin) / 0.26
        rew += HEADING_BONUS * (1.0 - abs(an)) * fwd

        # Optional reverse-escape reward for dedicated runs that unlock back-out behavior.
        if (REVERSE_ENABLED and self._act_lin < -0.01
                and min_dist_m < REVERSE_UNLOCK_SCAN_DIST):
            closeness = max(0.0, REVERSE_UNLOCK_SCAN_DIST - min_dist_m) / max(REVERSE_UNLOCK_SCAN_DIST, 1e-6)
            rew += REVERSE_ESCAPE_BONUS * closeness

        # Anti-spin penalty
        if REALFIX:
            # [REALFIX Fix 1] Continuous spin penalty + forward progress bonus.
            # Replaces hard-threshold version with smoother proportional shaping.
            lin_vel_norm = max(0.0, self._act_lin) / 0.26
            ang_vel_norm = abs(self._act_ang) / 1.82
            rew -= 0.08 * ang_vel_norm * (1.0 - lin_vel_norm)   # spin penalty
            if abs(an) < 0.35:                                    # heading within ~20°
                rew += 0.05 * lin_vel_norm                        # forward progress bonus
        else:
            if abs(self._act_lin) < 0.05:
                rew -= SPIN_PENALTY_SCALE * abs(self._act_ang) / 1.82

        # [REALFIX Fix 2] Action smoothness penalty — penalises direction reversals
        # each step to reduce jittery left-right flipping on real hardware.
        if REALFIX:
            smooth_penalty = -0.04 * (
                abs(self._act_lin - self._prev_act_lin) / 0.26
                + abs(self._act_ang - self._prev_act_ang) / 1.82
            )
            rew += smooth_penalty

        # [REALFIX Change 3] Linear velocity encouragement — quadratic penalty for
        # driving below max forward speed. Pushes robot to drive fast rather than
        # creeping. At act_lin=0: penalty≈−0.020. At act_lin=0.26: penalty=0.
        if REALFIX:
            rew += -0.3 * ((0.26 - max(0.0, self._act_lin)) ** 2)

        # Anti-loiter penalty for far-goal episodes that keep hanging around the center hub.
        if d > FAR_GOAL_MIN_DIST and self._is_near_center():
            rew -= CENTER_LOITER_PENALTY

        # Proximity warning
        if self._dz_mode == "baseline":
            if min_dist_m < PROXIMITY_DIST:
                rew -= PROXIMITY_SCALE * (PROXIMITY_DIST - min_dist_m) / 0.30
        else:
            if min_dist_m < DZ_PROX_DIST:
                rew -= DZ_PROX_SCALE * (DZ_PROX_DIST - min_dist_m) / 0.50

        # Terminal conditions
        done, info = False, 0
        if min_dist_m < COLLISION_DIST:
            if self._step_id < 30:
                rew = 0.0
                info = 0        # spawn artifact — reset without counting episode
                done = True
            else:
                rew = REWARD_COLLISION; info = 2; done = True
        elif d < GOAL_RADIUS:
            rew = REWARD_GOAL; info = 1; done = True
        elif (len(self._pos_hist) == STUCK_WINDOW
              and len(self._dist_hist) == STUCK_WINDOW
              and self._check_stuck()
              and self._check_progress_stall()):
            rew = REWARD_STUCK; info = 8; done = True
        elif (ENABLE_CENTER_LOITER_TERMINATION
              and self._step_id >= CENTER_LOITER_MIN_STEPS
              and self._check_center_loiter(d)):
            rew = REWARD_STUCK; info = 8; done = True
        elif (ENABLE_NO_PROGRESS_TERMINATION
              and self._step_id >= NO_PROGRESS_MIN_STEPS
              and self._check_no_progress()):
            rew = REWARD_STUCK; info = 8; done = True
        elif self._step_id >= MAX_STEPS:
            rew = REWARD_TIMEOUT; info = 4; done = True

        return rew, done, info

    def _check_stuck(self):
        """Check if robot has moved less than STUCK_THRESH over position history."""
        if len(self._pos_hist) < STUCK_WINDOW:
            return False
        x0, y0 = self._pos_hist[0]
        max_dist = 0.0
        for x, y in self._pos_hist:
            d = math.hypot(x - x0, y - y0)
            max_dist = max(max_dist, d)
        return max_dist < STUCK_THRESH

    def _check_progress_stall(self):
        """Check if goal distance has barely improved over the stuck window."""
        if len(self._dist_hist) < STUCK_WINDOW:
            return False
        start_dist = self._dist_hist[0]
        best_dist = min(self._dist_hist)
        return (start_dist - best_dist) < STALL_PROGRESS

    def _check_no_progress(self):
        """Catch long wandering episodes that move but fail to approach the goal enough."""
        if len(self._dist_hist) < NO_PROGRESS_WINDOW:
            return False
        recent = list(self._dist_hist)[-NO_PROGRESS_WINDOW:]
        start_dist = recent[0]
        best_dist = min(recent)
        return (start_dist - best_dist) < NO_PROGRESS_THRESHOLD

    def _is_near_center(self):
        if not self._odom:
            return False
        p = self._odom.pose.pose.position
        return math.hypot(p.x, p.y) < CENTER_LOITER_RADIUS

    def _check_center_loiter(self, goal_dist):
        """Detect far-goal episodes that keep circulating in the center hub."""
        if goal_dist < FAR_GOAL_MIN_DIST:
            return False
        if len(self._pos_hist) < CENTER_LOITER_WINDOW:
            return False
        recent = list(self._pos_hist)[-CENTER_LOITER_WINDOW:]
        inside = sum(1 for x, y in recent if math.hypot(x, y) < CENTER_LOITER_RADIUS)
        if inside < int(0.8 * CENTER_LOITER_WINDOW):
            return False
        if len(self._dist_hist) < CENTER_LOITER_WINDOW:
            return False
        recent_dist = list(self._dist_hist)[-CENTER_LOITER_WINDOW:]
        return (recent_dist[0] - min(recent_dist)) < NO_PROGRESS_THRESHOLD

    # ── main loop ─────────────────────────────────────────────────────────
    def _loop(self):
        # Reset sequence
        if self._want_reset and self._rs == "IDLE":
            if not self._reset_cli.service_is_ready():
                return
            self._rs = "WAIT"
            self._want_reset = False

            # Stop robot
            self._cmd_pub.publish(Twist())
            time.sleep(0.05)

            # Pause physics during reset
            if self._pause_cli.service_is_ready():
                self._pause_cli.call_async(Empty.Request())
            time.sleep(0.1)

            self._scan = None
            self._prev_lidar = None
            self._reset_cli.call_async(Empty.Request())

            # Reposition robot to Home and zero velocities
            if self._set_state_cli.service_is_ready():
                # Refresh starting coordinates from parameters
                hx = self.get_parameter("start_x").value
                hy = self.get_parameter("start_y").value
                s = EntityState()
                s.name = "waffle_pi"
                s.pose.position.x = hx
                s.pose.position.y = hy
                s.pose.position.z = 0.015
                s.pose.orientation.x = 0.0
                s.pose.orientation.y = 0.0
                s.pose.orientation.z = 0.0
                s.pose.orientation.w = 1.0
                s.twist.linear.x = 0.0
                s.twist.linear.y = 0.0
                s.twist.linear.z = 0.0
                s.twist.angular.x = 0.0
                s.twist.angular.y = 0.0
                s.twist.angular.z = 0.0
                req = SetEntityState.Request()
                req.state = s
                self._set_state_cli.call_async(req)

            time.sleep(0.7)

            # Unpause physics
            if self._unpause_cli.service_is_ready():
                self._unpause_cli.call_async(Empty.Request())
            time.sleep(0.5)

            self._pos_hist.clear()
            self._dist_hist.clear()
            self._act_lin = None
            self._act_ang = None
            self._step_id = 0
            self._rs = "IDLE"
            # [REALFIX Fix 2] Reset smoothness tracking at episode start
            self._prev_act_lin = 0.0
            self._prev_act_ang = 0.0
            obs, _, d, _ = self._get_obs()
            self._prev_dist = d
            msg = Float32MultiArray()
            msg.data = obs
            self._obs_pub.publish(msg)

        # Step execution
        if self._rs == "IDLE" and not self._want_reset and self._act_lin is not None:
            tw = Twist()
            tw.linear.x = self._act_lin
            tw.angular.z = self._act_ang
            self._cmd_pub.publish(tw)
            # [REALFIX Fix 4] 0.20s matches real LDS-02 at 5Hz.
            # Original 0.15s (6.67Hz) caused 25% faster training vs real robot.
            # STEP_SLEEP_S lets fast (free-run) worlds keep the SAME sim-time per
            # step as the baselines (set 0.15/RTF). Defaults to baseline behaviour.
            time.sleep(STEP_SLEEP_S)
            self._step_id += 1

            obs, min_dist_m, d, an = self._get_obs()
            # Track actual position for stuck detection
            if self._odom:
                p = self._odom.pose.pose.position
                self._pos_hist.append((p.x, p.y))
            if math.isfinite(d) and d < 900:
                self._dist_hist.append(d)

            # [REALFIX Fix 2] Store action for next-step smoothness penalty
            if REALFIX:
                self._prev_act_lin = self._act_lin
                self._prev_act_ang = self._act_ang

            rew, done, info = self._compute_reward(min_dist_m, d, an)

            msg = Float32MultiArray()
            msg.data = obs + [rew, float(done), float(info)]
            self._step_pub.publish(msg)
            self._act_lin = None
            self._act_ang = None
            self._prev_dist = d

            if done:
                self._cmd_pub.publish(Twist())
                self._want_reset = True
                self._step_id = 0
                self._goal_pub.publish(Bool(data=True))


def main():
    rclpy.init()
    rclpy.spin(EnvironmentPPO())


if __name__ == "__main__":
    main()
