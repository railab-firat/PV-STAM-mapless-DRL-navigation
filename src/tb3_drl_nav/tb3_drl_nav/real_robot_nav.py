#!/usr/bin/env python3
"""
real_robot_nav.py — Unified real-robot inference node
======================================================
  |  TurtleBot3 Burger + Jetson AGX Orin + LDS-02

Supports all 6 variants from a single node:
  variant=v8       : STAM + PER + 3-step + frame-stack(3)   [MLP]
  variant=v10      : v8 + Huber loss                         [MLP]
  variant=v11      : STAM + GRU Actor/Critic                 [GRU — carries hidden state]
  variant=baseline : Plain flat MLP, no attention, no stack  [MLP]
  variant=ppo      : PPO ActorCritic shared-MLP actor        [MLP, 54-dim obs]
  variant=dqn      : Phase-1 DQN discrete Q-network          [5 discrete actions, 26-dim obs]

Usage on Jetson:
  # v8 (champion)
  ros2 run tb3_drl_nav real_robot_nav \\
    --ros-args -p variant:=v8 -p run_id:=sac_v8_s42

  # v11 (recurrent)
  ros2 run tb3_drl_nav real_robot_nav \\
    --ros-args -p variant:=v11 -p run_id:=sac_v11_s42

  # PPO (model_dir must point to phase3 folder):
  ros2 run tb3_drl_nav real_robot_nav \\
    --ros-args -p variant:=ppo -p run_id:=phase3_v8 \\
                -p model_dir:=$HOME/tb3_drl_models/phase3

  # DQN (phase-1 weights copied as actor_latest.pt):
  ros2 run tb3_drl_nav real_robot_nav \\
    --ros-args -p variant:=dqn -p run_id:=dqn_phase1_final

  # Send a goal (x,y relative to odom origin, metres):
  ros2 topic pub /tb3_drl/real_goal std_msgs/msg/String "data: '2.0,0.5'" --once

LiDAR: LDS-02  →  topic /scan  (360 points, 0.12–3.5 m range)
PPO model file  : ~/tb3_drl_models/phase3/{run_id}/model_latest.pt
DQN model file  : ~/tb3_drl_models/sac/{run_id}/actor_latest.pt  (raw state_dict or q_state key)
"""

import os, sys, math, collections
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import SetBool
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants (must match training code exactly) ─────────────────────────────
N_SECTORS  = 24       # LiDAR bins after pooling (360 → 24)
NAV_DIM    = 6        # [dist_to_goal, head_err, lin_vel, ang_vel, scan_min, scan_mean]
RAW_OBS    = 54       # 24 scan + 24 scan_vel + 6 nav
ACT_DIM    = 2        # [lin_vel, ang_vel]
LIN_MAX    = 0.26     # m/s
ANG_MAX    = 1.82     # rad/s
SCAN_MAX   = 3.5      # m — LDS-02 max range
# Goal radius for REAL hardware. Raised 0.25 -> 0.40 m on 2026-08-15.
# Rationale (paper §"Goal-Registration Failure" + REAL_ROBOT_TEST_STATUS):
# wheel-odometry/OpenCR-IMU drift means the robot physically arrives at the goal
# while its own pose estimate still reports a larger distance, so the
# goal-reached test never fires and the trial times out. This was the DOMINANT
# real-robot failure mode (40% of SAC-R-PV-STAM trials) and the recorded travel
# distance confirmed the robot had reached the correct location — i.e. a
# measurement failure, not a navigation failure. Same value is used for every
# variant so cross-variant comparison stays fair.
GOAL_THRESH = 0.40    # m — consider goal reached (real robot)
# [SAFETY] added 2026-08-15
SCAN_TIMEOUT_S = 1.5  # stop if no /scan for this long (LDS-01 runs ~5 Hz => 0.2s)
# Trial timeout in CONTROL STEPS. The control loop runs at control_hz, which
# defaults to 30 Hz (NOT the 6.67 Hz sensing rate) — so 800 steps was only ~27 s.
# A 6 m goal takes >=23 s in a straight line at 0.26 m/s, so almost every real
# trial would have hit that limit and been logged as a timeout artefact.
# 2700 steps @30 Hz = 90 s, which comfortably covers a 6 m run with avoidance.
MAX_TRIAL_STEPS = int(os.environ.get("MAX_TRIAL_STEPS", "4500"))  # 150 s @30 Hz (9 m goal)
# [SAFETY] added 2026-08-16 — proximity stop, see _control_loop.
# 0.20 m is nav_environment.py's COLLISION_DIST, so sim and hardware share one
# collision definition. Raise it via -p collision_dist:=0.30 when people are in
# the arena; that also makes the robot stop earlier, which counts as a collision.
COLLISION_DIST_DEFAULT = 0.20
COLLISION_STREAK = 2   # consecutive scans required (single-beam noise guard)

# v8/v10 frame stack
N_FRAMES_V8 = 3
OBS_DIM_V8  = N_SECTORS * N_FRAMES_V8 + NAV_DIM   # 78

# v11 recurrent
ACTOR_HIDDEN_V11  = 256
OBS_DIM_V11       = RAW_OBS   # 54  (GRU handles temporal memory itself)

# baseline / PPO
OBS_DIM_BASE = RAW_OBS   # 54  (no frame stack)
OBS_DIM_PPO  = RAW_OBS   # 54  (same structure as baseline)

# DQN (Phase 1) — simple 26-dim obs: 24 LiDAR sectors + dist_norm + angle_norm
OBS_DIM_DQN       = 26
N_ACTIONS_DQN     = 5
GOAL_DIST_NORM_DQN = 3.0   # must match training (environment.py default)

# DQN discrete action table — (lin_vel m/s, ang_vel rad/s)
DQN_ACTIONS = [
    (0.00,  0.00),   # 0: Stop
    (0.22,  0.00),   # 1: Forward
    (0.15,  1.82),   # 2: Forward + Turn Left
    (0.15, -1.82),   # 3: Forward + Turn Right
   (-0.10,  0.00),   # 4: Backward (slow)
]


# ══════════════════════════════════════════════════════════════════════════════
#  Minimal network definitions  (must match training code exactly)
#  Only the actor forward pass is needed — no critic, no training.
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadScanAttention(nn.Module):
    """STAM — matches sac_v8.py and sac_v10.py exactly.
    CRITICAL: n_heads=2 and d_model=16 must match training config.
    The forward pass uses proper multi-head reshape+permute (not chunk).
    """
    def __init__(self, n_sectors=N_SECTORS, n_frames=N_FRAMES_V8,
                 d_model=16, n_heads=2, d_out=48):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_sectors = n_sectors
        self.n_heads   = n_heads
        self.d_k       = d_model // n_heads
        self.d_model   = d_model

        self.proj_in   = nn.Linear(n_frames, d_model)
        self.pos_enc   = nn.Parameter(torch.randn(1, n_sectors, d_model) * 0.02)
        self.W_qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out  = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.compress  = nn.Linear(n_sectors * d_model, d_out)

    def forward(self, x):
        B   = x.size(0)
        h   = self.proj_in(x) + self.pos_enc
        qkv = self.W_qkv(h).reshape(B, self.n_sectors, 3, self.n_heads, self.d_k
                          ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, self.n_sectors, self.d_model)
        return self.compress(self.proj_out(out).reshape(B, -1))



# ── Residual block (matches sac_v8.py exactly) ────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return F.relu(x + self.fc2(F.relu(self.fc1(x))))


# ── v8 / v10 Actor (identical architecture — only weights differ) ─────────────
class ActorV8(nn.Module):
    def __init__(self, hidden=256, n_sectors=N_SECTORS, n_frames=N_FRAMES_V8):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_frames  = n_frames
        self.stam    = MultiHeadScanAttention(n_sectors=n_sectors, n_frames=n_frames)
        self.ln      = nn.LayerNorm(48 + NAV_DIM)
        self.fc1     = nn.Linear(48 + NAV_DIM, hidden)
        self.fc2     = nn.Linear(hidden, hidden)
        self.res     = ResidualBlock(hidden)
        self.mu      = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Linear(hidden, ACT_DIM)

    def forward(self, obs):
        """obs: (B, OBS_DIM_V8)"""
        scan_stack = obs[:, :self.n_sectors * self.n_frames].reshape(
            -1, self.n_sectors, self.n_frames)
        nav = obs[:, self.n_sectors * self.n_frames:]
        x   = self.ln(torch.cat([self.stam(scan_stack), nav], dim=-1))
        x   = self.res(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return torch.tanh(self.mu(x))       # deterministic mean at eval time


class ActorMLPFS(nn.Module):
    """SAC-MLP-FS: same 78-dim frame-stacked input as v8, but NO attention —
    the stacked scan goes straight into the MLP trunk. Mirrors ActorFS in
    sac_mlp_fs.py (LayerNorm -> fc1 -> fc2 -> ResidualBlock -> mu)."""
    def __init__(self, hidden=256, n_sectors=N_SECTORS, n_frames=N_FRAMES_V8):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_frames  = n_frames
        trunk_in     = n_sectors * n_frames + NAV_DIM      # 24*3 + 6 = 78
        self.ln      = nn.LayerNorm(trunk_in)
        self.fc1     = nn.Linear(trunk_in, hidden)
        self.fc2     = nn.Linear(hidden, hidden)
        self.res     = ResidualBlock(hidden)
        self.mu      = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Linear(hidden, ACT_DIM)

    def forward(self, obs):
        """obs: (B, 78) — flat frame-stacked scan + nav, no reshape needed."""
        x = F.relu(self.fc1(self.ln(obs)))
        x = self.res(F.relu(self.fc2(x)))
        return torch.tanh(self.mu(x))       # deterministic mean at eval time


# ── v11 Recurrent Actor ────────────────────────────────────────────────────────
class MultiHeadScanAttentionV11(nn.Module):
    """STAM v2 — 2-frame input (scan, scan_vel) used in v11.
    CRITICAL: Must match sac_v11.py exactly (n_heads=2, d_model=16).
    """
    def __init__(self, n_sectors=N_SECTORS, n_frames=2,
                 d_model=16, n_heads=2, d_out=48):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_sectors = n_sectors
        self.n_heads   = n_heads
        self.d_k       = d_model // n_heads
        self.d_model   = d_model

        self.proj_in   = nn.Linear(n_frames, d_model)
        self.pos_enc   = nn.Parameter(torch.randn(1, n_sectors, d_model) * 0.02)
        self.W_qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out  = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.compress  = nn.Linear(n_sectors * d_model, d_out)

    def forward(self, x):
        B   = x.size(0)
        h   = self.proj_in(x) + self.pos_enc
        qkv = self.W_qkv(h).reshape(B, self.n_sectors, 3, self.n_heads, self.d_k
                          ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, self.n_sectors, self.d_model)
        return self.compress(self.proj_out(out).reshape(B, -1))



class ActorV11(nn.Module):
    """Recurrent actor — matches sac_v11.py RecurrentActor exactly."""
    def __init__(self, hidden=ACTOR_HIDDEN_V11):
        super().__init__()
        self.stam    = MultiHeadScanAttentionV11()
        self.ln      = nn.LayerNorm(48 + NAV_DIM)
        self.fc      = nn.Sequential(nn.Linear(48 + NAV_DIM, hidden), nn.ReLU())
        self.gru     = nn.GRU(hidden, hidden, batch_first=True)
        self.mu      = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Linear(hidden, ACT_DIM)

    def forward(self, obs, h=None):
        """obs: (B, T, OBS_DIM_V11)  h: (1, B, hidden) or None"""
        B, T, _ = obs.shape
        flat     = obs.reshape(B * T, -1)
        scan     = flat[:, :N_SECTORS]
        scan_vel = flat[:, N_SECTORS:2 * N_SECTORS]
        nav      = flat[:, 2 * N_SECTORS:]
        stam_in  = torch.stack([scan, scan_vel], dim=-1)
        x = self.fc(self.ln(torch.cat([self.stam(stam_in), nav], dim=-1)))
        x = x.reshape(B, T, ACTOR_HIDDEN_V11)
        gru_out, h_new = self.gru(x, h)
        return torch.tanh(self.mu(gru_out)), h_new   # deterministic mean


# ── Baseline Actor (plain flat MLP, no attention) ────────────────────────────
class ActorBaseline(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.ln  = nn.LayerNorm(OBS_DIM_BASE)
        self.fc1 = nn.Linear(OBS_DIM_BASE, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu  = nn.Linear(hidden, ACT_DIM)

    def forward(self, obs):
        x = F.relu(self.fc1(self.ln(obs)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.mu(x))


# ── PPO Actor (matches train_agent_ppo.py ActorCritic shared + actor_head) ───
class ActorPPO(nn.Module):
    """Deterministic mean of PPO stochastic policy (tanh-squashed).
    Loads from a full ActorCritic state_dict with strict=False — the
    critic_head and log_std keys are simply ignored at inference time.
    Obs: 54-dim  (24 scan + 24 scan_vel + 6 nav) — same as baseline/v11.
    """
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(OBS_DIM_PPO),
            nn.Linear(OBS_DIM_PPO, 256), nn.Tanh(),
            nn.Linear(256, 128),         nn.Tanh(),
        )
        self.actor_head = nn.Linear(128, ACT_DIM)
        self.log_std    = nn.Parameter(torch.zeros(ACT_DIM))  # needed for key match

    def forward(self, obs):
        """Return tanh-squashed deterministic mean action."""
        return torch.tanh(self.actor_head(self.shared(obs)))


# ── DQN Q-Network (Phase 1 — matches train_agent_dqn.py QNet exactly) ────────
class QNetDQN(nn.Module):
    """3-layer MLP discrete Q-network.
    Input: 26-dim (24 LiDAR sectors + dist_norm + angle_norm).
    Output: Q-values for 5 discrete actions.
    """
    def __init__(self, obs_dim=OBS_DIM_DQN, n_actions=N_ACTIONS_DQN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 Node
# ══════════════════════════════════════════════════════════════════════════════

class RealRobotNav(Node):
    def __init__(self):
        super().__init__('real_robot_nav')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('variant', 'v8')
        self.declare_parameter('run_id',  'sac_v8_s42')
        self.declare_parameter('model_dir',
            os.path.expanduser('~/tb3_drl_models/sac'))
        self.declare_parameter('control_hz', 30.0)
        self.declare_parameter('goal_thresh', GOAL_THRESH)
        self.declare_parameter('collision_dist', COLLISION_DIST_DEFAULT)
        # realfix=true: obs slots 50-51 are prev_commanded_action (not odom vel).
        # Must match the REALFIX env var used during training.
        # Use for models trained with REALFIX=true (e.g. sac_v8_s777, sac_v11_s777).
        # Leave false for all s42 / s123 models.
        self.declare_parameter('realfix', False)

        variant   = self.get_parameter('variant').value
        run_id    = self.get_parameter('run_id').value
        model_dir = self.get_parameter('model_dir').value
        hz        = self.get_parameter('control_hz').value
        self._goal_thresh = self.get_parameter('goal_thresh').value
        self._collision_dist = float(self.get_parameter('collision_dist').value)
        self._realfix     = bool(self.get_parameter('realfix').value)

        self.get_logger().info(f"Variant: {variant}  run_id: {run_id}")

        # ── GPU ───────────────────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Device: {self.device}")

        # ── Load actor weights ────────────────────────────────────────────────
        # PPO saves as model_latest.pt; all other variants use actor_latest.pt
        _primary = 'model_latest.pt' if variant.lower() == 'ppo' else 'actor_latest.pt'
        _fallback = 'actor_latest.pt' if _primary == 'model_latest.pt' else 'model_latest.pt'
        model_path = os.path.join(model_dir, run_id, _primary)
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, run_id, _fallback)
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found in {os.path.join(model_dir, run_id)}/")
            raise FileNotFoundError(model_path)

        self._variant = variant.lower()
        self._actor, self._load_fn = self._build_actor(self._variant)
        self._load_fn(model_path)
        self._actor.eval()
        self.get_logger().info(f"Loaded: {model_path}")

        # ── State ─────────────────────────────────────────────────────────────
        self._scan: np.ndarray | None = None
        self._prev_scan: np.ndarray | None = None    # for scan velocity (v8/v11)
        self._scan_history = collections.deque(maxlen=N_FRAMES_V8)
        self._pos    = np.zeros(2, dtype=np.float32)
        self._yaw    = 0.0
        self._vel    = np.zeros(2, dtype=np.float32)  # [lin, ang]
        self._goal   = np.array([2.0, 0.0], dtype=np.float32)
        self._h_v11  = None   # GRU hidden state for v11 (reset on new goal)
        self._goal_active = False
        self._trial_step  = 0
        self._trial_results: list[dict] = []
        # ── [SAFETY] added 2026-08-15 ──────────────────────────────────────
        self._estop        = False   # /tb3_drl/estop latch
        self._last_scan_t  = None    # set by _on_scan; drives the stale-scan watchdog
        self._stale_warned = False
        self._collision_streak = 0   # consecutive too-close scans (proximity stop)
        # [REALFIX] Previous commanded action for obs slots 50-51.
        # Normalised: lin ∈ [0,1], ang ∈ [-1,1] (same scale as training env).
        self._prev_act = np.zeros(2, dtype=np.float32)

        # ── Subscribers ───────────────────────────────────────────────────────
        qos_sensor = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, qos_sensor)
        self.create_subscription(Odometry,  '/odom', self._on_odom, 10)
        # Goal can come from the simulator topic OR the simpler real_goal topic
        self.create_subscription(String, '/tb3_drl/real_goal', self._on_goal, 10)
        self.create_subscription(String, '/tb3_drl/goal',      self._on_goal, 10)
        # [SAFETY] emergency stop — documented in REAL_ROBOT_TEST_STATUS but was
        # never implemented in code. Publish true to halt, false to re-arm:
        #   ros2 topic pub /tb3_drl/estop std_msgs/msg/Bool "data: true" --once
        self.create_subscription(Bool, '/tb3_drl/estop', self._on_estop, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self._cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',           10)
        self._status_pub = self.create_publisher(String, '/tb3_drl/nav_status', 10)

        # ── Motor torque ──────────────────────────────────────────────────────
        # The OpenCR comes up with torque DISABLED after every boot/power-cycle.
        # Nothing used to enable it, so the robot silently refused to move while
        # every other signal looked healthy (battery voltage fine, /cmd_vel
        # flowing, nodes up) — it had to be enabled by hand each session.
        # Enable it here, asynchronously so a missing service never blocks start-up.
        self._motor_cli = self.create_client(SetBool, '/motor_power')
        self.create_timer(2.0, self._ensure_torque)   # retries until it succeeds
        self._torque_ok = False
        try:
            from turtlebot3_msgs.msg import SensorState
            self.create_subscription(SensorState, '/sensor_state',
                                     self._on_sensor_state, 10)
        except Exception as e:                          # noqa: BLE001
            self.get_logger().warn(f"cannot watch torque state: {e}")

        # ── Control timer ─────────────────────────────────────────────────────
        self.create_timer(1.0 / hz, self._control_loop)
        self.get_logger().info(
            f"RealRobotNav ready  variant={variant}  goal_thresh={self._goal_thresh}m  "
            f"collision_stop={self._collision_dist}m  max_steps={MAX_TRIAL_STEPS}")

    def _on_sensor_state(self, msg):
        """Watch the real torque flag reported by the OpenCR.

        [FIX 2026-08-17] Torque was enabled once at start-up and then assumed to
        stay on. It does not: swapping the battery power-cycles the OpenCR, which
        clears torque back to false while this node keeps running and believes it
        is still enabled. The robot then silently refuses to move. Re-arm instead.
        """
        if self._torque_ok and not msg.torque:
            self.get_logger().warn("motor torque dropped (battery swap / OpenCR reset) — re-enabling")
            self._torque_ok = False        # let _ensure_torque try again

    def _ensure_torque(self):
        """Enable motor torque, and keep it enabled."""
        if self._torque_ok:
            return
        if not self._motor_cli.service_is_ready():
            return                      # bringup not up yet — try again next tick
        req = SetBool.Request(); req.data = True
        fut = self._motor_cli.call_async(req)

        def _done(f):
            try:
                if f.result() is not None and f.result().success:
                    self._torque_ok = True
                    self.get_logger().info("✓ Motor torque ENABLED")
                else:
                    self.get_logger().warn("motor_power call returned failure; will retry")
            except Exception as e:                      # noqa: BLE001
                self.get_logger().warn(f"motor_power call failed: {e}; will retry")
        fut.add_done_callback(_done)

    # ── Network factory ───────────────────────────────────────────────────────

    def _build_actor(self, variant: str):
        """Return (actor_module, load_fn) for the requested variant."""
        if variant in ('v8', 'v10'):
            actor = ActorV8().to(self.device)

            def load(path):
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                # actor_latest.pt may be a full state_dict or just actor weights
                if isinstance(ckpt, dict) and 'actor' in ckpt:
                    actor.load_state_dict(ckpt['actor'])
                else:
                    actor.load_state_dict(ckpt)
            return actor, load

        elif variant == 'v11':
            actor = ActorV11().to(self.device)

            def load(path):
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(ckpt, dict) and 'actor' in ckpt:
                    actor.load_state_dict(ckpt['actor'])
                else:
                    actor.load_state_dict(ckpt)
            return actor, load

        elif variant in ('mlp_fs', 'mlpfs'):
            actor = ActorMLPFS().to(self.device)

            def load(path):
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(ckpt, dict) and 'actor' in ckpt:
                    actor.load_state_dict(ckpt['actor'])
                else:
                    actor.load_state_dict(ckpt)
            return actor, load

        elif variant == 'baseline':
            actor = ActorBaseline().to(self.device)

            def load(path):
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(ckpt, dict) and 'actor' in ckpt:
                    actor.load_state_dict(ckpt['actor'], strict=False)
                else:
                    actor.load_state_dict(ckpt, strict=False)
            return actor, load

        elif variant == 'ppo':
            actor = ActorPPO().to(self.device)

            def load(path):
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                # PPO checkpoint may be a full ckpt dict (key 'model') or raw state_dict
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    sd = ckpt['model']
                elif isinstance(ckpt, dict) and 'actor' in ckpt:
                    sd = ckpt['actor']
                else:
                    sd = ckpt
                # strict=False: silently ignores critic_head and extra keys
                actor.load_state_dict(sd, strict=False)
            return actor, load

        elif variant == 'dqn':
            actor = QNetDQN().to(self.device)

            def load(path):
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(ckpt, dict) and 'q_state' in ckpt:
                    actor.load_state_dict(ckpt['q_state'])
                elif isinstance(ckpt, dict) and 'qt_state' in ckpt:
                    actor.load_state_dict(ckpt['qt_state'])
                elif isinstance(ckpt, dict) and ('net.0.weight' in ckpt
                                                  or 'net.0.weight' in list(ckpt.keys())[:3]):
                    actor.load_state_dict(ckpt)
                else:
                    # Try raw state_dict directly
                    actor.load_state_dict(ckpt)
            return actor, load

        else:
            raise ValueError(
                f"Unknown variant: {variant!r}. "
                f"Use v8, v10, v11, baseline, ppo, or dqn.")

    # ── ROS Callbacks ─────────────────────────────────────────────────────────

    def _on_scan(self, msg: LaserScan):
        """Pool 360 LiDAR readings → 24 bins, clip to [0.12, 3.5].

        [FIX 2026-08-15, reapplied 2026-08-16] 'No return' handling.
        In Gazebo a beam that hits nothing returns +inf, which the isfinite()
        line below maps to SCAN_MAX correctly. On the PHYSICAL LDS the same case
        returns **exactly 0.0**, which IS finite, so it survived that line and
        np.clip then turned it into 0.12 m — the minimum range, i.e. "an obstacle
        is touching the robot". Sector pooling takes the MIN, so one bad beam
        poisoned its whole sector.

        Measured on hardware: 45/360 beams read 0.0, producing phantom obstacles
        in 9 of 24 sectors and pinning min_scan at 0.12 m instead of 0.42 m.
        Without this fix the new proximity stop fires instantly and every trial
        is logged as an immediate COLLISION (observed 2026-08-16).

        The ROS LaserScan spec says values outside [range_min, range_max] must be
        DISCARDED, not clamped — so 0.0 and >range_max both become SCAN_MAX.
        """
        raw = np.array(msg.ranges, dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, SCAN_MAX)
        lo = float(getattr(msg, "range_min", 0.12)) or 0.12
        hi = float(getattr(msg, "range_max", SCAN_MAX)) or SCAN_MAX
        raw = np.where((raw < lo) | (raw > hi), SCAN_MAX, raw)   # no-return = free
        raw = np.clip(raw, 0.12, SCAN_MAX)
        n   = len(raw)
        # Reshape to 24 equal sectors, take min (most conservative)
        bins_per_sector = n // N_SECTORS
        pooled = raw[:bins_per_sector * N_SECTORS].reshape(N_SECTORS, -1).min(axis=1)
        self._prev_scan = self._scan
        self._scan = pooled
        self._last_scan_t = self.get_clock().now()   # [SAFETY] watchdog timestamp

    def _on_odom(self, msg: Odometry):
        self._pos[0] = msg.pose.pose.position.x
        self._pos[1] = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        siny  = 2.0 * (q.w * q.z + q.x * q.y)
        cosy  = 1.0 - 2.0 * (q.y**2 + q.z**2)
        self._yaw    = math.atan2(siny, cosy)
        self._vel[0] = msg.twist.twist.linear.x
        self._vel[1] = msg.twist.twist.angular.z

    def _on_goal(self, msg: String):
        try:
            parts = msg.data.strip().split(',')
            gx, gy = float(parts[0]), float(parts[1])
            self._goal = np.array([gx, gy], dtype=np.float32)
            # Reset GRU hidden state on new goal (new episode)
            self._h_v11 = None
            self._scan_history.clear()
            self._trial_step = 0
            self._prev_act   = np.zeros(2, dtype=np.float32)  # [REALFIX] reset
            self._goal_active = True
            self.get_logger().info(f"New goal: ({gx:.2f}, {gy:.2f})")
        except Exception as e:
            self.get_logger().error(f"Bad goal message: {msg.data!r} — {e}")

    # ── Observation builders ───────────────────────────────────────────────────

    def _nav_state(self) -> np.ndarray:
        """Compute 6-dim navigation state matching training environment.
        Slots 50-51 (indices 2-3 here):
          realfix=False (default) : odom velocity — matches s42/s123 models
          realfix=True            : prev commanded action — matches s777 realfix models
        """
        dx   = self._goal[0] - self._pos[0]
        dy   = self._goal[1] - self._pos[1]
        dist = float(np.hypot(dx, dy))
        angle_to_goal = math.atan2(dy, dx)
        head_err = math.atan2(math.sin(angle_to_goal - self._yaw),
                              math.cos(angle_to_goal - self._yaw))
        scan_min  = float(self._scan.min()) / SCAN_MAX
        scan_mean = float(self._scan.mean()) / SCAN_MAX
        if self._realfix:
            # [REALFIX Change 1] use prev commanded action (deterministic, lag-free)
            vel_lin = float(self._prev_act[0])   # already normalised [0,1]
            vel_ang = float(self._prev_act[1])   # already normalised [-1,1]
        else:
            vel_lin = self._vel[0] / LIN_MAX
            vel_ang = self._vel[1] / ANG_MAX
        return np.array([dist / 5.0,    # normalised distance
                         head_err / math.pi,
                         vel_lin,
                         vel_ang,
                         scan_min,
                         scan_mean], dtype=np.float32)

    def _scan_vel(self) -> np.ndarray:
        """Approximate scan velocity (change per step)."""
        if self._prev_scan is None:
            return np.zeros(N_SECTORS, dtype=np.float32)
        return np.clip((self._scan - self._prev_scan) / SCAN_MAX, -1.0, 1.0
                       ).astype(np.float32)

    def _build_obs_v8(self) -> np.ndarray:
        """OBS_DIM_V8=78: 3×24 frame-stacked scans + 6 nav."""
        norm_scan = self._scan / SCAN_MAX
        self._scan_history.append(norm_scan)
        while len(self._scan_history) < N_FRAMES_V8:
            self._scan_history.append(norm_scan)
        return np.concatenate(list(self._scan_history) + [self._nav_state()])

    def _build_obs_raw(self) -> np.ndarray:
        """OBS_DIM_BASE=54: 24 scan + 24 scan_vel + 6 nav (for v11, baseline, ppo)."""
        norm_scan = self._scan / SCAN_MAX
        scan_vel  = self._scan_vel()
        return np.concatenate([norm_scan, scan_vel, self._nav_state()])

    def _build_obs_dqn(self) -> np.ndarray:
        """OBS_DIM_DQN=26: 24 LiDAR sectors + dist_norm + angle_norm.
        Matches environment.py build_obs() used during Phase-1 DQN training.
        """
        norm_scan = np.clip(self._scan / SCAN_MAX, 0.0, 1.0)
        dx = self._goal[0] - self._pos[0]
        dy = self._goal[1] - self._pos[1]
        dist = float(np.hypot(dx, dy))
        angle_to_goal = math.atan2(dy, dx)
        rel_angle = math.atan2(math.sin(angle_to_goal - self._yaw),
                               math.cos(angle_to_goal - self._yaw))
        dist_n = max(0.0, min(1.0, dist / GOAL_DIST_NORM_DQN))
        ang_n  = max(-1.0, min(1.0, rel_angle / math.pi))
        return np.concatenate([norm_scan, [dist_n, ang_n]]).astype(np.float32)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _select_action(self) -> np.ndarray:
        with torch.no_grad():
            if self._variant in ('v8', 'v10', 'mlp_fs', 'mlpfs'):
                obs = self._build_obs_v8()   # 78-dim frame-stacked; same for mlp_fs
                t   = torch.tensor(obs, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                act = self._actor(t)
                return act[0].cpu().numpy()

            elif self._variant == 'v11':
                obs = self._build_obs_raw()
                # (1, 1, OBS_DIM) — single step, batch=1, seq=1
                t   = torch.tensor(obs, dtype=torch.float32,
                                   device=self.device).unsqueeze(0).unsqueeze(0)
                act_t, self._h_v11 = self._actor(t, self._h_v11)
                return act_t[0, 0].cpu().numpy()

            elif self._variant == 'baseline':
                obs = self._build_obs_raw()
                t   = torch.tensor(obs, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                act = self._actor(t)
                return act[0].cpu().numpy()

            elif self._variant == 'ppo':
                # Same 54-dim obs as baseline; tanh-squashed continuous action
                obs = self._build_obs_raw()
                t   = torch.tensor(obs, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                act = self._actor(t)
                return act[0].cpu().numpy()

            elif self._variant == 'dqn':
                # 26-dim obs; greedy argmax over Q-values → (lin, ang) physical vels
                obs = self._build_obs_dqn()
                t   = torch.tensor(obs, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                q   = self._actor(t)
                a   = int(q.argmax(dim=1).item())
                lin, ang = DQN_ACTIONS[a]
                self.get_logger().debug(f"DQN action={a} lin={lin:.2f} ang={ang:.2f}")
                # Return actual physical velocities (not normalised)
                return np.array([lin, ang], dtype=np.float32)

    # ── Control loop ──────────────────────────────────────────────────────────

    def _on_estop(self, msg: Bool):
        """[SAFETY] Emergency stop. true = halt immediately and cancel the goal."""
        if bool(msg.data):
            if not self._estop:
                self.get_logger().warn("*** ESTOP ENGAGED — robot halted, goal cancelled ***")
            self._estop = True
            self._goal_active = False
            self._cmd_pub.publish(Twist())
        else:
            if self._estop:
                self.get_logger().info("ESTOP released — send a new goal to resume.")
            self._estop = False

    def _control_loop(self):
        # [SAFETY] estop wins over everything else
        if self._estop:
            self._cmd_pub.publish(Twist())
            return

        if self._scan is None:
            return   # wait for first scan

        # [SAFETY] stale-scan watchdog. Previously only the FIRST scan was
        # checked: if the LiDAR died mid-trial (unplugged, driver crash, USB
        # glitch) self._scan kept its last value forever and the robot carried
        # on driving at full speed on blind, frozen data. Now we stop.
        if self._last_scan_t is not None:
            age = (self.get_clock().now() - self._last_scan_t).nanoseconds / 1e9
            if age > SCAN_TIMEOUT_S:
                if not self._stale_warned:
                    self.get_logger().error(
                        f"*** LiDAR SILENT for {age:.1f}s — STOPPING (was the robot "
                        f"unplugged / driver crashed?) ***")
                    self._stale_warned = True
                self._goal_active = False
                self._cmd_pub.publish(Twist())
                return
            elif self._stale_warned:
                self.get_logger().info("LiDAR recovered.")
                self._stale_warned = False

        if not self._goal_active:
            # Idle — publish zero velocity and wait for a goal
            self._cmd_pub.publish(Twist())
            return

        dist_to_goal = float(np.hypot(self._goal[0] - self._pos[0],
                                      self._goal[1] - self._pos[1]))

        if dist_to_goal < self._goal_thresh:
            self.get_logger().info(
                f"GOAL REACHED in {self._trial_step} steps  dist={dist_to_goal:.2f}m")
            self._goal_active = False
            self._cmd_pub.publish(Twist())     # stop
            status = String()
            status.data = f"REACHED,{self._trial_step},{dist_to_goal:.3f}"
            self._status_pub.publish(status)
            return

        # [SAFETY] trial timeout — without this the robot drives indefinitely
        # whenever the goal is never registered (the documented OpenCR/IMU
        # goal-registration failure caused exactly this, 40% of v11 trials).
        if self._trial_step >= MAX_TRIAL_STEPS:
            self.get_logger().warn(
                f"TIMEOUT after {self._trial_step} steps (dist={dist_to_goal:.2f}m) — stopping.")
            self._goal_active = False
            self._cmd_pub.publish(Twist())
            status = String()
            status.data = f"TIMEOUT,{self._trial_step},{dist_to_goal:.3f}"
            self._status_pub.publish(status)
            return

        # ── [SAFETY] proximity / collision stop ───────────────────────────────
        # Added 2026-08-16. THIS DID NOT EXIST and the robot drove into walls:
        # measured closest approach was 0.12 m (the LiDAR's minimum range, i.e.
        # touching) for the baseline policy and 0.23 m for v8. In Gazebo,
        # nav_environment.py ends the episode when min range < COLLISION_DIST
        # (0.20 m) — on real hardware nothing did, so the policy just kept going.
        #
        # Matching the simulator's threshold keeps the collision DEFINITION
        # identical between sim and hardware, so trial outcomes stay comparable.
        # Uses the 2nd-smallest beam because single spurious beams occur: a lone
        # 0.24 m reading was measured while the true minimum over 19 consecutive
        # scans was 0.65 m.
        near = float(np.partition(self._scan, 1)[1])
        if near < self._collision_dist:
            self._collision_streak += 1
        else:
            self._collision_streak = 0
        if self._collision_streak >= COLLISION_STREAK:
            self.get_logger().error(
                f"*** COLLISION STOP — obstacle {near:.2f} m (<{self._collision_dist:.2f} m) "
                f"for {COLLISION_STREAK} scans, step {self._trial_step} ***")
            self._goal_active = False
            self._collision_streak = 0
            self._cmd_pub.publish(Twist())
            status = String()
            status.data = f"COLLISION,{self._trial_step},{dist_to_goal:.3f},{near:.3f}"
            self._status_pub.publish(status)
            return

        act = self._select_action()
        cmd = Twist()
        if self._variant == 'dqn':
            # act already contains physical velocities from DQN_ACTIONS table
            cmd.linear.x  = float(act[0])
            cmd.angular.z = float(act[1])
        else:
            # SAC / PPO: tanh-normalised output in (-1, 1) → scale to physical vels
            cmd.linear.x  = float(np.clip(act[0] * LIN_MAX, 0.0, LIN_MAX))   # forward only
            cmd.angular.z = float(np.clip(act[1] * ANG_MAX, -ANG_MAX, ANG_MAX))
        self._cmd_pub.publish(cmd)
        # [REALFIX] store normalised prev_act for next obs build
        if self._realfix and self._variant != 'dqn':
            self._prev_act[0] = float(np.clip(act[0], 0.0, 1.0))   # tanh output [0,1] lin
            self._prev_act[1] = float(np.clip(act[1], -1.0, 1.0))  # tanh output [-1,1] ang
        self._trial_step += 1


def main(args=None):
    rclpy.init(args=args)
    node = RealRobotNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd_pub.publish(Twist())   # stop robot before exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
