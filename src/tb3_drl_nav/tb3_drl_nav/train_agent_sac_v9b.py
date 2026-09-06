#!/usr/bin/env python3
"""
train_agent_sac_v9b.py  —  SAC with STAM v3 (Safety-Aware Temporal Attention)
=============================================================================
Upgrades over v8:
  1. Action Smoothing — penalty for jerky actions, produces real-robot-ready trajectories
  2. Safety Critic  — auxiliary collision predictor that can veto dangerous actions
  3. Safety-Triggered Reverse — robot can back away when danger is high (forward-only otherwise)
  4. Collision Prediction Head — auxiliary self-supervised task (predicts collision in N steps)
  5. Curriculum-Aware PER — boosts replay priority near phase transitions
  6. Domain Randomization — LiDAR noise + observation delay for sim-to-real robustness
  7. All v8 features (multi-head STAM, 3-frame stacking, PER, n-step, residual, scheduled entropy)

Observation: env publishes 54 dims, we stack 3 LiDAR frames + prev action:
  scan_t (24) + scan_{t-1} (24) + scan_{t-2} (24) + nav (6) + prev_act (2) = 80 dims

Usage:
  DZ_MODE=stam RUN_ID=sac_v9b_stam FRESH=true ros2 run tb3_drl_nav train_agent_sac_v9b
"""
import collections, csv, glob, math, os, random, signal, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String, Int32

RESET_OBS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

# ── config ───────────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "phase3_ppo.yaml")
try:
    import yaml
    with open(_CFG_PATH) as f:
        _C = yaml.safe_load(f) or {}
except Exception:
    _C = {}

# ── hyperparameters (inherited from v7/v8 + new v9 keys) ─────────────────────
SEED       = int(os.environ.get("SEED", _C.get("seed", 42)))
RAW_OBS    = _C.get("obs_dim", 54)
ACT_DIM    = _C.get("act_dim", 2)
LIN_MIN    = _C.get("lin_min", 0.0)
LIN_MAX    = _C.get("lin_max", 0.26)
ANG_MAX    = _C.get("ang_max", 1.82)
HIDDEN     = _C.get("sac_hidden", 256)
BUFFER_CAP = _C.get("sac_buffer", 200_000)
BATCH      = _C.get("sac_batch", 256)
GAMMA      = _C.get("gamma", 0.99)
TAU        = _C.get("sac_tau", 0.005)
LR_ACTOR   = _C.get("sac_lr_actor", 3e-4)
LR_CRITIC  = _C.get("sac_lr_critic", 3e-4)
LR_ALPHA   = _C.get("sac_lr_alpha", 3e-4)
WARMUP     = _C.get("sac_warmup", 10_000)
GRAD_STEPS = _C.get("sac_grad_steps", 1)
UPDATE_EVERY = _C.get("sac_update_every", 1)
GRAD_CLIP  = _C.get("sac_grad_clip", 1.0)
LOG_EVERY  = _C.get("sac_log_every", 10)
SAVE_EVERY = _C.get("save_every", 50)
KEEP_CKPTS = _C.get("keep_ckpts", 8)
REPLAY_SAVE = _C.get("replay_save_every", 250)
DZ_MODE    = os.environ.get("DZ_MODE", "stam")
EVAL_ONLY  = os.environ.get("EVAL_ONLY", "").lower() in ("true", "1", "yes")
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", 0))

# v8 features (kept)
N_FRAMES   = _C.get("sac_v8_frame_stack", 3)
N_STEP     = _C.get("sac_v8_nstep", 3)
STAM_HEADS = _C.get("sac_v8_stam_heads", 2)
STAM_D     = _C.get("sac_v8_stam_d_model", 16)
PER_ALPHA  = _C.get("sac_v8_per_alpha", 0.6)
PER_BETA0  = _C.get("sac_v8_per_beta_start", 0.4)
PER_BETA_FRAMES = _C.get("sac_v8_per_beta_frames", 500_000)
ENT_START  = float(_C.get("sac_v8_entropy_start", -1.0))
ENT_END    = float(_C.get("sac_v8_entropy_end", -ACT_DIM))
ENT_ANNEAL = int(_C.get("sac_v8_entropy_anneal_eps", 500))

# v9-specific parameters
ACTION_SMOOTH_COEF   = float(_C.get("sac_v9_action_smooth", 0.1))
SAFETY_THRESH_START  = float(_C.get("sac_v9_safety_thresh_start", 0.95))
SAFETY_THRESH_END    = float(_C.get("sac_v9_safety_thresh_end", 0.70))
SAFETY_THRESH_ANNEAL = int(_C.get("sac_v9_safety_thresh_anneal", 500))
SAFETY_COEF          = float(_C.get("sac_v9_safety_coef", 0.3))
COLLISION_PRED_HORIZON = int(_C.get("sac_v9_collision_horizon", 5))
AUX_LOSS_COEF        = float(_C.get("sac_v9_aux_loss_coef", 0.2))
PHASE_BOOST          = float(_C.get("sac_v9_phase_boost", 2.0))
LIDAR_NOISE_STD      = float(_C.get("sac_v9_lidar_noise", 0.02))
OBS_DELAY_PROB       = float(_C.get("sac_v9_obs_delay_prob", 0.05))
REVERSE_SPEED        = float(_C.get("sac_v9_reverse_speed", 0.10))
NEAR_MISS_DIST       = float(_C.get("sac_v9_near_miss_dist", 0.40))
NEAR_MISS_BONUS      = float(_C.get("sac_v9_near_miss_bonus", 0.3))
LR_COSINE_MIN        = float(_C.get("sac_v9_lr_min_factor", 0.1))
LR_COSINE_EPISODES   = int(_C.get("sac_v9_lr_cosine_eps", 2000))

# Observation: 24*3 scan frames + 6 nav + 2 prev_action = 80
OBS_DIM    = 24 * N_FRAMES + 6 + ACT_DIM   # 80
N_SECTORS  = 24

GOAL_CATALOG = [
    (0.0, 1.2), (0.0, -1.2), (0.0, 1.9), (0.0, -1.9),
    (1.4, 1.4), (-1.4, 1.4), (1.4, -1.4), (-1.4, -1.4),
    (0.0, 2.5), (0.0, -2.5), (2.5, 0.0), (-2.5, 0.0),
    (3.2, 1.5), (-3.2, 1.5), (3.2, -1.5), (-3.2, -1.5),
    (-2.8, 2.8), (2.8, 2.8), (-2.8, -2.8), (2.8, -2.8),
    (-3.0, 2.2), (3.0, 2.2), (-3.0, -2.2), (3.0, -2.2),
    (-0.5, 3.0), (0.5, 3.0), (0.0, -3.1), (-0.5, -3.0), (0.5, -3.0),
    (-4.4, 4.4), (4.4, 4.4), (-4.4, -4.4), (4.4, -4.4),
    (0.0, 4.4), (0.0, -4.4), (4.4, 0.0), (-4.4, 0.0),
    # Benchmark goals (5×5m ROBOTIS worlds)
    (0.7, 0.7), (-0.7, 0.7), (-0.7, -0.7), (0.7, -0.7),
    (0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0),
    (1.5, 1.5), (-1.5, 1.5), (-1.5, -1.5), (1.5, -1.5),
    (0.0, 1.8), (0.0, -1.8), (1.8, 0.0), (-1.8, 0.0),
]
GOAL_ID_BY_POS = {
    (round(x, 1), round(y, 1)): f"G{idx:02d}"
    for idx, (x, y) in enumerate(GOAL_CATALOG, start=1)
}


def _goal_label(x, y):
    return GOAL_ID_BY_POS.get((round(x, 1), round(y, 1)), "G??")


# ══════════════════════════════════════════════════════════════════════════════
#  PRIORITIZED EXPERIENCE REPLAY (from v8, enhanced with curriculum awareness)
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """Binary tree for O(log n) proportional sampling."""
    __slots__ = ("capacity", "tree", "data", "write_idx", "n_entries")

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def _leaf(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self._update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        if tree_idx > 0:
            self._propagate(tree_idx, change)

    def get(self, s):
        idx = self._leaf(s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    def update(self, tree_idx, priority):
        self._update(tree_idx, priority)


class PrioritizedReplayBuffer:
    """PER with proportional prioritization, IS weights, and curriculum-aware boosting."""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=500_000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._max_priority = 1.0
        self._min_priority = 1e-6

    def add(self, obs, act, rew, nobs, done, priority_boost=1.0):
        """Store a transition. priority_boost > 1 for phase-transition experiences."""
        transition = (
            np.asarray(obs, np.float32),
            np.asarray(act, np.float32),
            float(rew),
            np.asarray(nobs, np.float32),
            float(done),
        )
        priority = self._max_priority * priority_boost
        self.tree.add(priority, transition)

    def sample(self, n, device, total_steps):
        beta = min(1.0, self.beta_start + total_steps *
                   (1.0 - self.beta_start) / self.beta_frames)
        indices, priorities, batch = [], [], []
        segment = self.tree.total / n

        for i in range(n):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, prio, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0, self.tree.total)
                idx, prio, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(max(prio, self._min_priority))
            batch.append(data)

        probs = np.array(priorities) / self.tree.total
        weights = (len(self) * probs) ** (-beta)
        weights /= weights.max()

        o, a, r, no, d = zip(*batch)
        t = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)
        return (t(o), t(a), t(r).unsqueeze(1), t(no), t(d).unsqueeze(1),
                indices, torch.tensor(weights, dtype=torch.float32,
                                      device=device).unsqueeze(1))

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            p = (abs(td) + self._min_priority) ** self.alpha
            self._max_priority = max(self._max_priority, p)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

    def to_list(self):
        out = []
        for i in range(self.tree.n_entries):
            d = self.tree.data[i]
            if d is not None:
                out.append(d)
        return out

    @classmethod
    def from_list(cls, data, cap, alpha=0.6, beta_start=0.4, beta_frames=500_000):
        buf = cls(cap, alpha, beta_start, beta_frames)
        for item in data[-cap:]:
            buf.tree.add(buf._max_priority, tuple(item))
        return buf


# ══════════════════════════════════════════════════════════════════════════════
#  N-STEP RETURN BUFFER (unchanged from v8)
# ══════════════════════════════════════════════════════════════════════════════

class NStepBuffer:
    """Accumulates transitions and yields n-step returns."""

    def __init__(self, n=3, gamma=0.99):
        self.n = n
        self.gamma = gamma
        self.buf = collections.deque(maxlen=n)

    def add(self, obs, act, rew, nobs, done):
        self.buf.append((obs, act, rew, nobs, done))
        if done:
            results = []
            while self.buf:
                results.append(self._compute())
                self.buf.popleft()
            return results
        elif len(self.buf) == self.n:
            result = [self._compute()]
            self.buf.popleft()
            return result
        return []

    def _compute(self):
        obs0, act0 = self.buf[0][0], self.buf[0][1]
        R = 0.0
        for i in range(len(self.buf)):
            R += (self.gamma ** i) * self.buf[i][2]
            if self.buf[i][4]:
                return obs0, act0, R, self.buf[i][3], True
        return obs0, act0, R, self.buf[-1][3], False

    def reset(self):
        self.buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  COLLISION PREDICTION BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class CollisionLabelBuffer:
    """
    Tracks recent (obs, action) pairs within an episode. When the episode ends,
    we can label the last N steps as 'about to collide' (1.0) or 'safe' (0.0).
    This gives us free self-supervised labels for the collision predictor.
    """

    def __init__(self, horizon=5):
        self.horizon = horizon
        self._episode_buffer = []

    def add(self, obs, act):
        self._episode_buffer.append((obs.copy(), act.copy()))

    def flush(self, collided):
        """
        Call at episode end. Returns list of (obs, act, label) tuples.
        If the robot collided, the last `horizon` steps get label=1.0.
        All other steps get label=0.0.
        """
        labeled = []
        n = len(self._episode_buffer)
        for i, (obs, act) in enumerate(self._episode_buffer):
            if collided and i >= n - self.horizon:
                label = 1.0
            else:
                label = 0.0
            labeled.append((obs, act, label))
        self._episode_buffer = []
        return labeled

    def reset(self):
        self._episode_buffer = []


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORKS — STAM v3
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadScanAttention(nn.Module):
    """
    STAM v3 — same multi-head self-attention as v2 but with dropout for
    regularization. The attention mechanism learns which LiDAR sectors matter
    most given the current spatial-temporal context.
    """

    def __init__(self, n_sectors=N_SECTORS, n_frames=N_FRAMES,
                 d_model=STAM_D, n_heads=STAM_HEADS, d_out=48):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_sectors = n_sectors
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.d_out = d_out

        self.proj_in = nn.Linear(n_frames, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_sectors, d_model) * 0.02)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_drop = nn.Dropout(0.05)

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.compress = nn.Linear(n_sectors * d_model, d_out)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.W_qkv.weight)
        nn.init.xavier_uniform_(self.compress.weight)
        nn.init.zeros_(self.compress.bias)

    def forward(self, x):
        B = x.size(0)
        h = self.proj_in(x) + self.pos_enc
        qkv = self.W_qkv(h).reshape(B, self.n_sectors, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, self.n_sectors, self.d_model)
        out = self.proj_out(out)
        out = out.reshape(B, -1)
        return self.compress(out)

    def get_attention_weights(self, x):
        """Extract attention weights for visualization (inference only)."""
        B = x.size(0)
        h = self.proj_in(x) + self.pos_enc
        qkv = self.W_qkv(h).reshape(B, self.n_sectors, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        scale = math.sqrt(self.d_k)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        return attn  # (B, heads, 24, 24)


class ResidualBlock(nn.Module):
    """Two-layer residual block with zero-init second layer."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self._init()

    def _init(self):
        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return F.relu(x + self.fc2(F.relu(self.fc1(x))))


class CollisionPredictor(nn.Module):
    """
    Auxiliary head: given (obs, action), predict P(collision within N steps).
    This is a self-supervised binary classifier trained on free labels from
    episode outcomes. Forces the trunk to learn better obstacle representations.
    """

    def __init__(self, obs_features=48 + 6, act_dim=ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_features + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Small init so it doesn't dominate early training
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, obs_features, action):
        """Returns logit (not sigmoid) for numerical stability with BCE loss."""
        x = torch.cat([obs_features, action], dim=-1)
        return self.net(x)


class SafetyCritic(nn.Module):
    """
    Separate critic that estimates collision probability from the current state.
    Unlike the Q-critic (which estimates total return), this only asks:
    'Am I about to crash?'

    Used at inference time: if P(collision) > threshold, we slow the robot
    down and increase turning — a learned safety override.
    """

    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        return self.net(obs)


class ActorV9b(nn.Module):
    """
    Squashed Gaussian policy with STAM v3 + residual block + prev action awareness.
    Obs layout: [scan_frames(72), nav(6), prev_action(2)] = 80 dims.
    STAM processes scan frames → 48, then we concat with nav(6) + prev_act(2) = 56.
    """
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self):
        super().__init__()
        self.use_stam = (DZ_MODE == "stam")
        if self.use_stam:
            self.stam = MultiHeadScanAttention(d_out=48)
            # stam_out(48) + nav(6) + prev_action(2) = 56
            trunk_in = 48 + 6 + ACT_DIM
        else:
            trunk_in = OBS_DIM  # 80 raw

        self.ln = nn.LayerNorm(trunk_in)
        self.fc1 = nn.Linear(trunk_in, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.res = ResidualBlock(HIDDEN)
        self.mu = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(HIDDEN, ACT_DIM)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        for m in [self.mu, self.log_std]:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def _apply_stam(self, obs):
        scan_frames = obs[:, :N_SECTORS * N_FRAMES]          # (B, 72)
        nav_and_act = obs[:, N_SECTORS * N_FRAMES:]           # (B, 8) = nav(6) + prev_act(2)
        x = scan_frames.reshape(-1, N_SECTORS, N_FRAMES)      # (B, 24, 3)
        stam_out = self.stam(x)                                # (B, 48)
        return torch.cat([stam_out, nav_and_act], dim=-1)      # (B, 56)

    def get_features(self, obs):
        """Get STAM features + nav + prev_act (used by collision predictor)."""
        if self.use_stam:
            return self._apply_stam(obs)
        return obs

    def forward(self, obs):
        if self.use_stam:
            x = self._apply_stam(obs)
        else:
            x = obs
        x = F.relu(self.fc1(self.ln(x)))
        x = F.relu(self.fc2(x))
        x = self.res(x)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std.exp()

    def sample(self, obs):
        mu, std = self(obs)
        dist = Normal(mu, std)
        raw = dist.rsample()
        act = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) -
                    torch.log(1.0 - act.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return act, log_prob

    def deterministic(self, obs):
        mu, _ = self(obs)
        return torch.tanh(mu)


class CriticV9b(nn.Module):
    """Twin Q-networks with STAM v3 + residual block."""

    def __init__(self):
        super().__init__()
        self.use_stam = (DZ_MODE == "stam")
        if self.use_stam:
            self.stam = MultiHeadScanAttention(d_out=48)
            # stam_out(48) + nav(6) + prev_act(2) + current_act(2) = 58
            trunk_in = 48 + 6 + ACT_DIM + ACT_DIM
        else:
            trunk_in = OBS_DIM + ACT_DIM  # 80 + 2 = 82

        self.q1_fc1 = nn.Linear(trunk_in, HIDDEN)
        self.q1_fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.q1_res = ResidualBlock(HIDDEN)
        self.q1_out = nn.Linear(HIDDEN, 1)
        self.q2_fc1 = nn.Linear(trunk_in, HIDDEN)
        self.q2_fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.q2_res = ResidualBlock(HIDDEN)
        self.q2_out = nn.Linear(HIDDEN, 1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        for m in [self.q1_out, self.q2_out]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def _apply_stam(self, obs):
        scan_frames = obs[:, :N_SECTORS * N_FRAMES]            # (B, 72)
        nav_and_act = obs[:, N_SECTORS * N_FRAMES:]             # (B, 8)
        x = scan_frames.reshape(-1, N_SECTORS, N_FRAMES)
        stam_out = self.stam(x)
        return torch.cat([stam_out, nav_and_act], dim=-1)     # (B, 56)

    def forward(self, obs, act):
        if self.use_stam:
            x = self._apply_stam(obs)                          # (B, 56)
        else:
            x = obs                                            # (B, 80)
        x = torch.cat([x, act], dim=-1)                       # + current action
        q1 = self.q1_out(self.q1_res(F.relu(self.q1_fc2(F.relu(self.q1_fc1(x))))))
        q2 = self.q2_out(self.q2_res(F.relu(self.q2_fc2(F.relu(self.q2_fc1(x))))))
        return q1, q2

    def q_min(self, obs, act):
        q1, q2 = self(obs, act)
        return torch.min(q1, q2)


# ══════════════════════════════════════════════════════════════════════════════
#  SAC AGENT — v9
# ══════════════════════════════════════════════════════════════════════════════

class SACv9b:
    def __init__(self, device):
        self.device = device

        # Core networks (same roles as v8)
        self.actor      = ActorV9b().to(device)
        self.critic     = CriticV9b().to(device)
        self.critic_tgt = CriticV9b().to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        # New v9: auxiliary heads
        # Features = stam_out(48) + nav(6) + prev_act(2) = 56 for stam mode
        feat_dim = 48 + 6 + ACT_DIM if DZ_MODE == "stam" else OBS_DIM
        self.collision_pred = CollisionPredictor(feat_dim, ACT_DIM).to(device)
        self.safety_critic  = SafetyCritic(OBS_DIM).to(device)

        # Optimizers
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.opt_aux    = torch.optim.Adam(
            list(self.collision_pred.parameters()) +
            list(self.safety_critic.parameters()),
            lr=LR_CRITIC)

        # Temperature (same as v8)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, obs, deterministic=False):
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.deterministic(t)
            else:
                a, _ = self.actor.sample(t)
            # Safety check: if collision predictor says danger, nudge action
            danger = self.safety_critic(t).item()
        return a.squeeze(0).cpu().numpy(), danger

    def update(self, replay, target_h, total_steps, prev_actions=None):
        """One SAC gradient step with PER weights, n-step discount, and action smoothing."""
        obs, act, rew, nobs, done, indices, is_w = replay.sample(
            BATCH, self.device, total_steps)
        alpha = self.log_alpha.exp().detach()

        # ── Critic update (n-step discount) ──────────────────────────────
        with torch.no_grad():
            na, nlp = self.actor.sample(nobs)
            tq1, tq2 = self.critic_tgt(nobs, na)
            target = rew + (GAMMA ** N_STEP) * (1 - done) * (
                torch.min(tq1, tq2) - alpha * nlp)

        q1, q2 = self.critic(obs, act)
        loss_c = (is_w * (q1 - target).pow(2)).mean() + \
                 (is_w * (q2 - target).pow(2)).mean()
        self.opt_critic.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        # TD errors for priority update
        td_err = torch.max((q1 - target).abs(),
                           (q2 - target).abs()).squeeze(1).detach().cpu().numpy()
        replay.update_priorities(indices, td_err)

        # ── Actor update with action smoothing ────────────────────────────
        for p in self.critic.parameters():
            p.requires_grad = False
        new_act, new_lp = self.actor.sample(obs)
        q_pi = self.critic.q_min(obs, new_act)

        # Action smoothing: penalize large changes between consecutive actions
        # This encourages smooth, real-robot-friendly trajectories
        if prev_actions is not None:
            smooth_penalty = ACTION_SMOOTH_COEF * (new_act - prev_actions).pow(2).mean()
        else:
            smooth_penalty = 0.0

        loss_a = (alpha * new_lp - q_pi).mean() + smooth_penalty
        self.opt_actor.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
        for p in self.critic.parameters():
            p.requires_grad = True

        # ── Temperature update ────────────────────────────────────────────
        loss_t = -(self.log_alpha * (new_lp.detach() + target_h)).mean()
        self.opt_alpha.zero_grad()
        loss_t.backward()
        self.opt_alpha.step()

        # ── Soft-update target critic ─────────────────────────────────────
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - TAU).add_(TAU * p.data)

        return loss_a.item(), loss_c.item(), -new_lp.mean().item(), self.alpha

    def update_auxiliary(self, obs_batch, act_batch, labels):
        """
        Train collision predictor and safety critic on self-supervised labels.
        Called at the end of each episode with labeled data from that episode.
        """
        if len(obs_batch) == 0:
            return 0.0

        obs_t = torch.tensor(np.array(obs_batch), dtype=torch.float32,
                             device=self.device)
        act_t = torch.tensor(np.array(act_batch), dtype=torch.float32,
                             device=self.device)
        lbl_t = torch.tensor(np.array(labels), dtype=torch.float32,
                             device=self.device).unsqueeze(1)

        # Collision predictor uses STAM features (shared with actor)
        with torch.no_grad():
            features = self.actor.get_features(obs_t)
        logits = self.collision_pred(features.detach(), act_t)
        loss_pred = F.binary_cross_entropy_with_logits(logits, lbl_t)

        # Safety critic uses raw observation
        danger_prob = self.safety_critic(obs_t)
        loss_safety = F.binary_cross_entropy(danger_prob, lbl_t)

        loss_aux = AUX_LOSS_COEF * (loss_pred + loss_safety)
        self.opt_aux.zero_grad()
        loss_aux.backward()
        nn.utils.clip_grad_norm_(
            list(self.collision_pred.parameters()) +
            list(self.safety_critic.parameters()), GRAD_CLIP)
        self.opt_aux.step()

        return loss_aux.item()

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "collision_pred": self.collision_pred.state_dict(),
            "safety_critic": self.safety_critic.state_dict(),
            "opt_aux": self.opt_aux.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_alpha": self.opt_alpha.state_dict(),
        }

    def load_state_dict(self, d, device):
        self.actor.load_state_dict(d["actor"], strict=False)
        self.critic.load_state_dict(d["critic"], strict=False)
        self.critic_tgt.load_state_dict(d.get("critic_tgt", d["critic"]), strict=False)
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_critic.load_state_dict(d["opt_critic"])
        if "collision_pred" in d:
            self.collision_pred.load_state_dict(d["collision_pred"], strict=False)
        if "safety_critic" in d:
            self.safety_critic.load_state_dict(d["safety_critic"], strict=False)
        if "opt_aux" in d:
            self.opt_aux.load_state_dict(d["opt_aux"])
        if "log_alpha" in d:
            self.log_alpha = d["log_alpha"].clone().to(device).requires_grad_(True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
            if "opt_alpha" in d:
                self.opt_alpha.load_state_dict(d["opt_alpha"])


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 TRAINING NODE
# ══════════════════════════════════════════════════════════════════════════════

class TrainAgentSACv9b(Node):
    def __init__(self):
        super().__init__("train_agent_sac_v9b")
        self.declare_parameter("run_id", "sac_v9b_stam")
        self.declare_parameter("fresh", False)
        run_id = self.get_parameter("run_id").value
        self._fresh = self.get_parameter("fresh").value

        # Reproducibility
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # Device
        cuda_ok = False
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] * 10 + cap[1] >= 60:
                try:
                    torch.zeros(1).cuda()
                    cuda_ok = True
                except Exception:
                    pass
        self.device = torch.device("cuda" if cuda_ok else "cpu")

        # Agent + buffers
        self.agent = SACv9b(self.device)
        self.replay = PrioritizedReplayBuffer(
            BUFFER_CAP, PER_ALPHA, PER_BETA0, PER_BETA_FRAMES)
        self.nstep = NStepBuffer(N_STEP, GAMMA)
        self.collision_labels = CollisionLabelBuffer(COLLISION_PRED_HORIZON)

        # Frame stacking
        self._scan_history = collections.deque(maxlen=N_FRAMES)

        # Domain randomization: delayed observation buffer
        self._delayed_obs = None

        # Action smoothing: track previous action
        self._prev_tanh_act = np.zeros(ACT_DIM, dtype=np.float32)

        # Phase transition tracking for curriculum-aware PER
        self._recent_phase_change = False
        self._phase_change_countdown = 0

        # Paths
        log_dir = os.path.expanduser(_C.get("log_dir", "~/tb3_drl_logs/phase3"))
        model_dir = os.path.expanduser(
            os.path.join(_C.get("sac_model_dir", "~/tb3_drl_models/sac"), run_id))
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self._model_dir  = model_dir
        self._log_path   = os.path.join(log_dir, f"{run_id}.csv")
        self._update_log = os.path.join(log_dir, f"{run_id}_updates.csv")
        self._run_id     = run_id

        # Counters
        self._ep = 0
        self._total_steps = 0
        self._best_sr = 0.0
        self._updates = 0
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._elapsed_offset = 0.0
        self._t0 = time.time()
        self._outcomes = []
        self._steps_hist = []
        self._waiting_reset = True
        self._current_obs = None
        self._current_act = None
        self._cur_phase = 1
        self._cur_goal = "?"
        self._danger_level = 0.0
        self._aux_loss_avg = 0.0
        self._safety_interventions = 0
        self._ep_safety_interventions = 0

        # Resume or fresh
        if self._fresh:
            self.get_logger().info("[SACv9b] FRESH START — ignoring checkpoints.")
        else:
            self._resume()

        # CSV headers
        if self._ep == 0:
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode", "reward", "steps", "collision", "goal_reached",
                    "terminal_type", "sr_100", "mean_steps_100", "updates", "elapsed_s",
                    "danger_avg", "safety_interventions"])
            with open(self._update_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "update", "episode", "actor_loss", "critic_loss",
                    "entropy", "alpha", "aux_loss", "buffer_size", "elapsed_s"])

        # ROS2 pub/sub
        self._act_pub = self.create_publisher(
            Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/reset_obs", self._on_reset, RESET_OBS_QOS)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/step_result", self._on_step, 10)
        self.create_subscription(
            String, "/tb3_drl/goal", self._on_goal, 10)
        self.create_subscription(
            Int32, "/tb3_drl/curriculum_phase", self._on_phase, RESET_OBS_QOS)

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        mode_str = "EVAL (deterministic, no training)" if EVAL_ONLY else "TRAIN"
        a_params = sum(p.numel() for p in self.agent.actor.parameters())
        s_params = sum(p.numel() for p in self.agent.safety_critic.parameters())
        c_params = sum(p.numel() for p in self.agent.collision_pred.parameters())
        self.get_logger().info(
            f"[SACv9b] Ready  run={run_id}  DZ_MODE={DZ_MODE}  mode={mode_str}  "
            f"obs_dim={OBS_DIM}  n_step={N_STEP}  frames={N_FRAMES}  "
            f"stam_heads={STAM_HEADS}  actor={a_params:,}  "
            f"safety={s_params:,}  collision_pred={c_params:,}  "
            f"smooth={ACTION_SMOOTH_COEF}  "
            f"safety={SAFETY_THRESH_START}→{SAFETY_THRESH_END}/{SAFETY_THRESH_ANNEAL}eps  "
            f"reverse={REVERSE_SPEED}m/s  near_miss={NEAR_MISS_BONUS}@{NEAR_MISS_DIST}m  "
            f"lr_cosine={LR_COSINE_MIN}x/{LR_COSINE_EPISODES}eps  lidar_noise={LIDAR_NOISE_STD}  "
            f"ent={ENT_START}→{ENT_END}/{ENT_ANNEAL}eps  device={self.device}")

    # ── domain randomization ────────────────────────────────────────────
    def _add_lidar_noise(self, scan):
        """Add small Gaussian noise to LiDAR readings for sim-to-real robustness."""
        if LIDAR_NOISE_STD > 0 and not EVAL_ONLY:
            noise = np.random.normal(0, LIDAR_NOISE_STD, len(scan))
            scan = np.clip(scan + noise, 0.0, 1.0)
        return scan

    # ── frame stacking ──────────────────────────────────────────────────
    def _build_stacked_obs(self, raw_obs):
        """Convert 54-dim env obs → 80-dim stacked obs with prev action + domain randomization."""
        scan_t = np.array(raw_obs[:N_SECTORS], dtype=np.float32)

        # Domain randomization: add sensor noise during training
        scan_t = self._add_lidar_noise(scan_t)

        self._scan_history.append(scan_t)
        while len(self._scan_history) < N_FRAMES:
            self._scan_history.appendleft(scan_t.copy())

        nav = np.array(raw_obs[48:54], dtype=np.float32)

        # Append previous action so the robot knows what it just did
        # This helps avoid oscillation — if it just turned left, it knows
        prev_act = self._prev_tanh_act.copy()

        stacked = np.concatenate(
            [self._scan_history[N_FRAMES - 1 - i] for i in range(N_FRAMES)]
            + [nav, prev_act], dtype=np.float32)  # (80,)

        # Domain randomization: occasional 1-step observation delay
        if not EVAL_ONLY and random.random() < OBS_DELAY_PROB:
            if self._delayed_obs is not None:
                old = stacked.copy()
                stacked = self._delayed_obs
                self._delayed_obs = old
                return stacked
            self._delayed_obs = stacked.copy()

        return stacked  # (80,)

    # ── scheduled entropy target ─────────────────────────────────────────
    @property
    def _target_entropy(self):
        t = min(1.0, self._ep / max(1, ENT_ANNEAL))
        return ENT_START + t * (ENT_END - ENT_START)

    # ── adaptive safety threshold ────────────────────────────────────────
    @property
    def _safety_threshold(self):
        """
        Early training: threshold is high (0.95) → safety critic barely intervenes
        (because it hasn't learned yet and would give bad advice).
        Late training: threshold drops to 0.70 → safety critic actively protects.
        """
        t = min(1.0, self._ep / max(1, SAFETY_THRESH_ANNEAL))
        return SAFETY_THRESH_START + t * (SAFETY_THRESH_END - SAFETY_THRESH_START)

    # ── cosine learning rate schedule ────────────────────────────────────
    def _update_lr(self):
        """
        Cosine decay: high LR early (fast learning) → low LR late (fine-tuning).
        Matches the entropy schedule — explore fast, then refine.
        """
        t = min(1.0, self._ep / max(1, LR_COSINE_EPISODES))
        factor = LR_COSINE_MIN + 0.5 * (1.0 - LR_COSINE_MIN) * (1.0 + math.cos(math.pi * t))
        for pg in self.agent.opt_actor.param_groups:
            pg['lr'] = LR_ACTOR * factor
        for pg in self.agent.opt_critic.param_groups:
            pg['lr'] = LR_CRITIC * factor

    # ── resume ───────────────────────────────────────────────────────────
    def _resume(self):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
        if not ckpts:
            return
        try:
            ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
            self.agent.load_state_dict(ckpt["agent"], self.device)
            self._ep = int(ckpt["episode"])
            self._total_steps = int(ckpt.get("total_steps", 0))
            self._updates = int(ckpt["updates"])
            self._outcomes = list(ckpt.get("outcomes", []))
            self._steps_hist = list(ckpt.get("steps_hist", []))
            self._elapsed_offset = float(ckpt.get("elapsed_s", 0.0))
            self._best_sr = float(ckpt.get("best_sr", 0.0))
            if ckpt.get("replay"):
                self.replay = PrioritizedReplayBuffer.from_list(
                    ckpt["replay"], BUFFER_CAP, PER_ALPHA, PER_BETA0, PER_BETA_FRAMES)
            if len(self.replay) == 0:
                rb = os.path.join(self._model_dir, "replay_buffer.pt")
                if os.path.exists(rb):
                    try:
                        d = torch.load(rb, map_location="cpu", weights_only=False)
                        self.replay = PrioritizedReplayBuffer.from_list(
                            d["replay"], BUFFER_CAP,
                            PER_ALPHA, PER_BETA0, PER_BETA_FRAMES)
                    except Exception:
                        pass
            self.get_logger().info(
                f"[SACv9b] Resumed ep={self._ep}  buf={len(self.replay)}  "
                f"upd={self._updates}  elapsed={self._elapsed_offset/3600:.1f}h")
        except Exception as e:
            self.get_logger().error(f"[SACv9b] Resume failed ({e}) — fresh start.")

    # ── checkpoint ───────────────────────────────────────────────────────
    def _save_ckpt(self, include_replay=False):
        path = os.path.join(self._model_dir, f"ckpt_ep{self._ep:06d}.pt")
        elapsed = self._elapsed_offset + (time.time() - self._t0)
        torch.save({
            "agent": self.agent.state_dict(),
            "episode": self._ep,
            "total_steps": self._total_steps,
            "updates": self._updates,
            "outcomes": self._outcomes,
            "steps_hist": self._steps_hist,
            "elapsed_s": elapsed,
            "best_sr": self._best_sr,
            "replay": (self.replay.to_list()
                       if include_replay and len(self.replay) > 0 else None),
        }, path)
        torch.save(self.agent.actor.state_dict(),
                   os.path.join(self._model_dir, "actor_latest.pt"))
        for old in sorted(glob.glob(
                os.path.join(self._model_dir, "ckpt_ep*.pt")))[:-KEEP_CKPTS]:
            try:
                os.remove(old)
            except OSError:
                pass

    def _save_replay_backup(self):
        if len(self.replay) == 0:
            return
        torch.save({
            "replay": self.replay.to_list(),
            "episode": self._ep,
        }, os.path.join(self._model_dir, "replay_buffer.pt"))

    def _shutdown(self, signum, frame):
        self.get_logger().info("[SACv9b] Shutdown — saving with replay…")
        self.nstep.reset()
        self._save_ckpt(include_replay=True)
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass

    # ── ROS callbacks ────────────────────────────────────────────────────
    def _on_reset(self, msg):
        raw = np.array(msg.data[:RAW_OBS], dtype=np.float32)
        if not np.all(np.isfinite(raw)):
            return
        self._scan_history.clear()
        self._delayed_obs = None
        self.nstep.reset()
        self.collision_labels.reset()
        self._current_obs = self._build_stacked_obs(raw)
        self._current_act = None
        self._prev_tanh_act = np.zeros(ACT_DIM, dtype=np.float32)
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_safety_interventions = 0
        self._waiting_reset = False
        self._send_action(self._current_obs)

    def _on_step(self, msg):
        if self._waiting_reset or self._current_act is None:
            return
        if len(msg.data) < RAW_OBS + 3:
            return
        raw_nobs = np.array(msg.data[:RAW_OBS], dtype=np.float32)
        reward = float(msg.data[RAW_OBS])
        done   = bool(msg.data[RAW_OBS + 1])
        info   = int(msg.data[RAW_OBS + 2])

        nobs = self._build_stacked_obs(raw_nobs)
        if not (np.all(np.isfinite(nobs)) and math.isfinite(reward)):
            return

        # Near-miss bonus: reward the robot for being close to obstacles
        # without colliding. Teaches "close but safe" is good, not lucky.
        # Only applies when not done (surviving a close call mid-episode).
        if not done and not EVAL_ONLY:
            min_scan = min(raw_nobs[:N_SECTORS])  # normalised [0,1]
            min_dist_m = min_scan * 3.5            # convert to metres
            if min_dist_m < NEAR_MISS_DIST and info != 2:
                reward += NEAR_MISS_BONUS * (NEAR_MISS_DIST - min_dist_m) / NEAR_MISS_DIST

        self._ep_reward += reward
        self._ep_steps += 1
        self._total_steps += 1

        if not EVAL_ONLY:
            # Curriculum-aware PER: boost priority if we recently changed phase
            boost = PHASE_BOOST if self._phase_change_countdown > 0 else 1.0
            if self._phase_change_countdown > 0:
                self._phase_change_countdown -= 1

            # Push through n-step buffer → PER replay with boost
            transitions = self.nstep.add(
                self._current_obs, self._current_act, reward, nobs, float(done))
            for t in transitions:
                self.replay.add(*t, priority_boost=boost)

            # Track for collision prediction labels
            self.collision_labels.add(self._current_obs, self._current_act)

            # SAC gradient updates
            if (self._total_steps > WARMUP
                    and len(self.replay) >= BATCH
                    and self._total_steps % UPDATE_EVERY == 0):
                # Build prev_actions tensor from batch for smoothing
                # We use the current action as a proxy since we don't store
                # previous actions in the replay buffer
                prev_act_t = torch.tensor(
                    self._prev_tanh_act, dtype=torch.float32,
                    device=self.device).unsqueeze(0).expand(BATCH, -1)

                for _ in range(GRAD_STEPS):
                    la, lc, ent, alpha = self.agent.update(
                        self.replay, self._target_entropy,
                        self._total_steps, prev_act_t)
                    self._updates += 1
                    if self._updates % LOG_EVERY == 0:
                        elapsed = self._elapsed_offset + (time.time() - self._t0)
                        with open(self._update_log, "a", newline="") as f:
                            csv.writer(f).writerow([
                                self._updates, self._ep,
                                round(la, 6), round(lc, 6),
                                round(ent, 6), round(alpha, 6),
                                round(self._aux_loss_avg, 6),
                                len(self.replay), round(elapsed, 1)])

        self._current_obs = nobs
        if done:
            self._end_episode(info)
        else:
            self._send_action(nobs)

    def _on_phase(self, msg):
        old = self._cur_phase
        self._cur_phase = msg.data
        if msg.data > old:
            self.get_logger().info(f"[SACv9b] Phase {old} → {msg.data}")
            # Mark next 200 steps for curriculum-aware PER boost
            self._phase_change_countdown = 200
            self._recent_phase_change = True

    def _on_goal(self, msg):
        try:
            x, y = [float(v) for v in msg.data.split(",")]
            self._cur_goal = f"{_goal_label(x, y)} ({x:+.1f},{y:+.1f})"
        except Exception:
            self._cur_goal = "?"

    # ── episode end ──────────────────────────────────────────────────────
    def _end_episode(self, info):
        if info == 0:
            self._waiting_reset = True
            return

        self._ep += 1
        goal = 1 if info == 1 else 0
        collision = 1 if info == 2 else 0

        # Update cosine learning rate schedule
        if not EVAL_ONLY:
            self._update_lr()

        # Train auxiliary heads on this episode's data
        if not EVAL_ONLY and self._ep_steps > 5:
            labeled = self.collision_labels.flush(collided=bool(collision))
            if labeled:
                obs_b = [l[0] for l in labeled]
                act_b = [l[1] for l in labeled]
                lbl_b = [l[2] for l in labeled]
                aux_loss = self.agent.update_auxiliary(obs_b, act_b, lbl_b)
                # Exponential moving average for logging
                self._aux_loss_avg = 0.95 * self._aux_loss_avg + 0.05 * aux_loss
        else:
            self.collision_labels.reset()

        self._outcomes.append(goal)
        self._steps_hist.append(self._ep_steps)
        if len(self._outcomes) > 100:
            self._outcomes.pop(0)
        if len(self._steps_hist) > 100:
            self._steps_hist.pop(0)
        sr = 100.0 * sum(self._outcomes) / len(self._outcomes)
        ms = sum(self._steps_hist) / len(self._steps_hist)
        elapsed = self._elapsed_offset + (time.time() - self._t0)

        terminal_tag = "GOAL(1)" if goal else ("COLL(0)" if collision else ("STUK(0)" if info == 8 else "TIME(0)"))
        warmup_str = (f"  [WARMUP {self._total_steps}/{WARMUP}]"
                      if self._total_steps < WARMUP else "")
        safe_str = (f"  safe_int={self._ep_safety_interventions}"
                    if self._ep_safety_interventions > 0 else "")
        self.get_logger().info(
            f"Ep {self._ep:5d} | Ph{self._cur_phase} | goal={self._cur_goal} | "
            f"{terminal_tag} | "
            f"R={self._ep_reward:+8.1f} | steps={self._ep_steps:4d} | "
            f"SR={sr:5.1f}% | "
            f"H_tgt={self._target_entropy:.2f} | "
            f"S_thr={self._safety_threshold:.2f} | "
            f"danger={self._danger_level:.2f} | "
            f"buf={len(self.replay):,} | upd={self._updates}"
            f"{warmup_str}{safe_str}")

        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self._ep, round(self._ep_reward, 2), self._ep_steps,
                collision, goal, goal and "GOAL" or collision and "COLL" or terminal_tag, round(sr, 2), round(ms, 1),
                self._updates, round(elapsed, 1),
                round(self._danger_level, 3),
                self._ep_safety_interventions])

        if not EVAL_ONLY:
            if self._ep % SAVE_EVERY == 0:
                self._save_ckpt()
            if self._ep % REPLAY_SAVE == 0:
                self._save_replay_backup()

            if sr > self._best_sr and self._ep >= 100:
                self._best_sr = sr
                best = os.path.join(self._model_dir, "ckpt_best_sr.pt")
                torch.save({
                    "agent": self.agent.state_dict(),
                    "episode": self._ep, "best_sr": self._best_sr,
                    "total_steps": self._total_steps,
                    "updates": self._updates,
                    "outcomes": self._outcomes,
                    "steps_hist": self._steps_hist,
                    "elapsed_s": elapsed, "replay": None,
                }, best)
                torch.save(self.agent.actor.state_dict(),
                           os.path.join(self._model_dir, "actor_best_sr.pt"))
                self.get_logger().info(
                    f"[SACv9b] *** NEW BEST SR={sr:.1f}% at ep={self._ep} ***")

        if MAX_EPISODES > 0 and self._ep >= MAX_EPISODES:
            self.get_logger().info(
                f"[SACv9b] MAX_EPISODES={MAX_EPISODES} reached — SR={sr:.1f}%")
            self._save_ckpt()
            rclpy.shutdown()
            return

        self._waiting_reset = True

    # ── action with safety override + reverse ───────────────────────────
    def _send_action(self, obs):
        if EVAL_ONLY:
            tanh_act, danger = self.agent.select_action(obs, deterministic=True)
        elif self._total_steps < WARMUP:
            tanh_act = np.random.uniform(-1, 1, ACT_DIM).astype(np.float32)
            danger = 0.0
        else:
            tanh_act, danger = self.agent.select_action(obs)

        self._danger_level = danger

        # Safety-triggered response: when the safety critic detects high
        # collision probability, we intervene in two stages:
        #
        #   danger > 0.7 (SAFETY_THRESHOLD):
        #       Slow down — scale forward speed by (1 - danger)
        #       Unlock reverse — remap action so robot CAN back away
        #       The agent's tanh_act[0] < 0 now means actual reverse motion
        #
        #   danger <= 0.7:
        #       Normal forward-only mapping [0, LIN_MAX]
        #       Robot cannot go backwards — simpler action space for learning
        #
        # This gives the robot an escape ability it only uses when cornered,
        # without making the normal action space harder to learn.

        if danger > self._safety_threshold and self._total_steps > WARMUP:
            # Unlock reverse: remap full tanh range [-1, 1] → [-REVERSE_SPEED, LIN_MAX]
            # tanh < ~-0.28 = actual reverse, tanh > ~-0.28 = forward (but capped slower)
            # The agent learns: "when I'm in danger, negative tanh = back away"
            lin = float(-REVERSE_SPEED + (tanh_act[0] + 1.0) / 2.0
                        * (LIN_MAX + REVERSE_SPEED))

            # Bias toward caution: cap forward speed based on danger level
            # danger=0.7 → cap=0.20, danger=0.85 → cap=0.14, danger=1.0 → cap=0.08
            max_safe_lin = LIN_MAX * max(0.3, 1.5 - 1.5 * danger)
            if lin > 0:
                lin = min(lin, max_safe_lin)

            self._safety_interventions += 1
            self._ep_safety_interventions += 1
        else:
            # Normal mode: forward only [0, LIN_MAX]
            lin = float(LIN_MIN + (tanh_act[0] + 1.0) / 2.0 * (LIN_MAX - LIN_MIN))

        ang = float(tanh_act[1] * ANG_MAX)

        self._prev_tanh_act = tanh_act.copy()
        self._current_act = tanh_act

        out = Float32MultiArray()
        out.data = [lin, ang]
        self._act_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrainAgentSACv9b()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
