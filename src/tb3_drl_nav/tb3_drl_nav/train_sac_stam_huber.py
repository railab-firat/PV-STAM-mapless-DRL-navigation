#!/usr/bin/env python3
"""
train_agent_sac_v10.py  —  SAC STAM v2 + wider critic + Huber loss
===================================================================
What changed from v8 (two targeted changes only):

  Root cause from v8 analysis:
    - Long episodes (>200 steps): only 13% SR — robot fails on far goals
    - Reward range: collision −89 to −335 → huge MSE gradient variance
    - Critic oscillated (loss 96 at ep819, only 33 at ep1439) = capacity bottleneck

  Change 1 — Wider critic: 256 → CRITIC_HIDDEN=384 (both layers)
    Critic needs to model long 200+ step trajectories through 15 obstacles.
    Actor stays at HIDDEN=256 — actor was not the bottleneck.

  Change 2 — Huber loss on critic (delta=10) instead of MSE
    MSE squares large collision errors (−335²=112K vs −89²=7.9K = 14× variance).
    Huber clips errors above delta=10, giving stable gradients on terminal states.

Everything else identical to v8: same obs (78 dims), same STAM, same PER,
same n-step, same reward, same entropy schedule, same actor.

Usage:
  DZ_MODE=stam RUN_ID=sac_v10_stam FRESH=true ros2 run tb3_drl_nav train_agent_sac_v10
"""
import collections, csv, glob, math, os, random, signal, sys, time
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


def _cfg_get(path, default, legacy_key=None):
    cur = _C
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            cur = None
            break
        cur = cur[key]
    if cur is not None:
        return cur
    if legacy_key is not None:
        return _C.get(legacy_key, default)
    return default

# ── hyperparameters (v7 keys reused + new v8 keys) ──────────────────────────
SEED       = int(os.environ.get("SEED", _C.get("seed", 42)))
RAW_OBS    = _C.get("obs_dim", 54)        # env observation dim
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
RUN_NOTE   = os.environ.get("RUN_NOTE", "").strip()
RESET_SETTLE_S = float(_cfg_get(("trainer_runtime", "terminal_filters", "reset_settle_s"), 0.10))
MIN_STUCK_STEPS = int(_cfg_get(("trainer_runtime", "terminal_filters", "min_stuck_steps"), 60))
MIN_TIMEOUT_STEPS = int(_cfg_get(("trainer_runtime", "terminal_filters", "min_timeout_steps"), 500))
REVERSE_ENABLED = bool(_cfg_get(("reverse_motion", "enabled"), False))
REVERSE_SPEED = float(_cfg_get(("reverse_motion", "speed"), 0.10))
REVERSE_UNLOCK_SCAN_DIST = float(_cfg_get(("reverse_motion", "unlock_scan_dist"), 0.32))
REVERSE_UNLOCK_APPROACH = float(_cfg_get(("reverse_motion", "unlock_approach"), -0.03))
REVERSE_CAUTION_FORWARD_FACTOR = float(_cfg_get(("reverse_motion", "caution_forward_factor"), 0.55))

# v8 features (unchanged)
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

# v10 changes
CRITIC_HIDDEN = _C.get("sac_v10_critic_hidden", 384)   # wider than actor (256)
HUBER_DELTA   = _C.get("sac_v10_huber_delta", 10.0)    # Huber loss clip threshold

OBS_DIM    = 24 * N_FRAMES + 6   # stacked obs: 78 (unchanged)
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
#  PRIORITIZED EXPERIENCE REPLAY
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
        while idx < self.capacity - 1:        # internal node
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
        """Return (tree_idx, priority, data) for a cumulative sum query."""
        idx = self._leaf(s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    def update(self, tree_idx, priority):
        self._update(tree_idx, priority)


class PrioritizedReplayBuffer:
    """PER with proportional prioritization and importance-sampling weights."""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=500_000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._max_priority = 1.0
        self._min_priority = 1e-6

    def add(self, obs, act, rew, nobs, done):
        transition = (
            np.asarray(obs, np.float32),
            np.asarray(act, np.float32),
            float(rew),
            np.asarray(nobs, np.float32),
            float(done),
        )
        self.tree.add(self._max_priority, transition)

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
            if data is None:                    # safety: shouldn't happen
                s = random.uniform(0, self.tree.total)
                idx, prio, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(max(prio, self._min_priority))
            batch.append(data)

        # Importance-sampling weights
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
        """Serialize for checkpoint (transitions only; priorities rebuilt)."""
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
#  N-STEP RETURN BUFFER
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
            # Flush all remaining partial n-step transitions
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
            if self.buf[i][4]:          # episode ended at step i
                return obs0, act0, R, self.buf[i][3], True
        # Not terminal — bootstrap from last transition
        return obs0, act0, R, self.buf[-1][3], False

    def reset(self):
        self.buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORKS — STAM v2
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadScanAttention(nn.Module):
    """
    STAM v2 — Multi-head self-attention over 24 LiDAR sectors.

    Each sector has N_FRAMES temporal channels (scan_t, scan_{t-1}, scan_{t-2}).
    Self-attention lets sectors attend to each other, learning patterns like
    "obstacle left AND right → go straight" that per-sector MLPs cannot.

    Architecture:
      Input (B, 24, 3)
      → Linear projection (B, 24, d_model=16) + positional encoding
      → Multi-head self-attention (2 heads, d_k=8)
      → Output projection (B, 24, d_model) → flatten → Linear → (B, 48)
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

        # Input projection: per-sector temporal features → d_model
        self.proj_in = nn.Linear(n_frames, d_model)

        # Learnable positional encoding (angular position of each sector)
        self.pos_enc = nn.Parameter(torch.randn(1, n_sectors, d_model) * 0.02)

        # QKV projections (fused for efficiency)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # Final compression: flatten (B, 24, d_model) → (B, d_out)
        self.compress = nn.Linear(n_sectors * d_model, d_out)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.W_qkv.weight)
        nn.init.xavier_uniform_(self.compress.weight)
        nn.init.zeros_(self.compress.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 24, N_FRAMES) — temporal scan data per sector.
        Returns:
            (B, d_out) — compressed attention-weighted representation.
        """
        B = x.size(0)

        # Project to d_model and add positional encoding
        h = self.proj_in(x) + self.pos_enc                     # (B, 24, d_model)

        # Compute Q, K, V
        qkv = self.W_qkv(h).reshape(B, self.n_sectors, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)                      # (3, B, heads, 24, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale    # (B, heads, 24, 24)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                             # (B, heads, 24, d_k)

        # Concatenate heads
        out = out.transpose(1, 2).reshape(B, self.n_sectors, self.d_model)  # (B, 24, d_model)
        out = self.proj_out(out)                                # (B, 24, d_model)

        # Flatten and compress
        out = out.reshape(B, -1)                                # (B, 24*d_model)
        return self.compress(out)                               # (B, d_out)


class ResidualBlock(nn.Module):
    """Two-layer residual block with pre-activation."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self._init()

    def _init(self):
        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        # Zero-init second layer so residual starts as identity
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return F.relu(x + self.fc2(F.relu(self.fc1(x))))


class ActorV10(nn.Module):
    """Squashed Gaussian policy with STAM v2 + residual block."""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self):
        super().__init__()
        # STAM v2 (only for stam mode; others get identity)
        self.use_stam = (DZ_MODE == "stam")
        if self.use_stam:
            self.stam = MultiHeadScanAttention(d_out=48)
            trunk_in = 48 + 6  # stam_out + nav
        else:
            trunk_in = OBS_DIM  # 78 raw

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
        """Split obs into scan frames + nav, apply STAM to scan frames."""
        scan_frames = obs[:, :N_SECTORS * N_FRAMES]             # (B, 72)
        nav = obs[:, N_SECTORS * N_FRAMES:]                     # (B, 6)
        x = scan_frames.reshape(-1, N_SECTORS, N_FRAMES)        # (B, 24, 3)
        stam_out = self.stam(x)                                 # (B, 48)
        return torch.cat([stam_out, nav], dim=-1)               # (B, 54)

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


class CriticV10(nn.Module):
    """Twin Q-networks — wider hidden (384 vs actor's 256) for better value estimation."""

    def __init__(self):
        super().__init__()
        self.use_stam = (DZ_MODE == "stam")
        if self.use_stam:
            self.stam = MultiHeadScanAttention(d_out=48)
            trunk_in = 48 + 6 + ACT_DIM   # stam_out + nav + action
        else:
            trunk_in = OBS_DIM + ACT_DIM

        # Q1 — wider (CRITIC_HIDDEN=384)
        self.q1_fc1 = nn.Linear(trunk_in, CRITIC_HIDDEN)
        self.q1_fc2 = nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
        self.q1_res = ResidualBlock(CRITIC_HIDDEN)
        self.q1_out = nn.Linear(CRITIC_HIDDEN, 1)
        # Q2 — wider (CRITIC_HIDDEN=384)
        self.q2_fc1 = nn.Linear(trunk_in, CRITIC_HIDDEN)
        self.q2_fc2 = nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
        self.q2_res = ResidualBlock(CRITIC_HIDDEN)
        self.q2_out = nn.Linear(CRITIC_HIDDEN, 1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        for m in [self.q1_out, self.q2_out]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def _apply_stam(self, obs):
        scan_frames = obs[:, :N_SECTORS * N_FRAMES]
        nav = obs[:, N_SECTORS * N_FRAMES:]
        x = scan_frames.reshape(-1, N_SECTORS, N_FRAMES)
        stam_out = self.stam(x)
        return torch.cat([stam_out, nav], dim=-1)

    def forward(self, obs, act):
        if self.use_stam:
            x = self._apply_stam(obs)
        else:
            x = obs
        x = torch.cat([x, act], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_res(q1)
        q1 = self.q1_out(q1)
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_res(q2)
        q2 = self.q2_out(q2)
        return q1, q2

    def q_min(self, obs, act):
        q1, q2 = self(obs, act)
        return torch.min(q1, q2)


# ══════════════════════════════════════════════════════════════════════════════
#  SAC AGENT — v8
# ══════════════════════════════════════════════════════════════════════════════

class SACv10:
    def __init__(self, device):
        self.device = device
        self.actor      = ActorV10().to(device)
        self.critic     = CriticV10().to(device)
        self.critic_tgt = CriticV10().to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Auto-entropy: log_alpha unconstrained (proven: no clamp)
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
        return a.squeeze(0).cpu().numpy()

    def update(self, replay, target_h, total_steps):
        """One SAC gradient step with PER weights and n-step discount."""
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

        # IS-weighted Huber loss (delta=10) — clips large collision reward gradients
        # v8 used MSE; collision rewards ranged −89 to −335 causing gradient spikes
        loss_c = (is_w * F.huber_loss(q1, target, reduction='none', delta=HUBER_DELTA)).mean() + \
                 (is_w * F.huber_loss(q2, target, reduction='none', delta=HUBER_DELTA)).mean()
        self.opt_critic.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        # TD errors for priority update
        td_err = torch.max((q1 - target).abs(),
                           (q2 - target).abs()).squeeze(1).detach().cpu().numpy()
        replay.update_priorities(indices, td_err)

        # ── Actor update (freeze critic) ─────────────────────────────────
        for p in self.critic.parameters():
            p.requires_grad = False
        new_act, new_lp = self.actor.sample(obs)
        q_pi = self.critic.q_min(obs, new_act)
        loss_a = (alpha * new_lp - q_pi).mean()
        self.opt_actor.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
        for p in self.critic.parameters():
            p.requires_grad = True

        # ── Temperature update (scheduled target, no clamp) ──────────────
        loss_t = -(self.log_alpha * (new_lp.detach() + target_h)).mean()
        self.opt_alpha.zero_grad()
        loss_t.backward()
        self.opt_alpha.step()

        # ── Soft-update target critic ────────────────────────────────────
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - TAU).add_(TAU * p.data)

        return loss_a.item(), loss_c.item(), -new_lp.mean().item(), self.alpha

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_alpha": self.opt_alpha.state_dict(),
        }

    def load_state_dict(self, d, device):
        self.actor.load_state_dict(d["actor"], strict=False)
        self.critic.load_state_dict(d["critic"], strict=False)
        self.critic_tgt.load_state_dict(d.get("critic_tgt", d["critic"]), strict=False)
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_critic.load_state_dict(d["opt_critic"])
        if "log_alpha" in d:
            self.log_alpha = d["log_alpha"].clone().to(device).requires_grad_(True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
            if "opt_alpha" in d:
                self.opt_alpha.load_state_dict(d["opt_alpha"])


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 TRAINING NODE
# ══════════════════════════════════════════════════════════════════════════════

class TrainAgentSACv10(Node):
    def __init__(self):
        super().__init__("train_agent_sac_v10")
        self.declare_parameter("run_id", "sac_v10_stam")
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
        self.agent = SACv10(self.device)
        self.replay = PrioritizedReplayBuffer(
            BUFFER_CAP, PER_ALPHA, PER_BETA0, PER_BETA_FRAMES)
        self.nstep = NStepBuffer(N_STEP, GAMMA)

        # Frame stacking
        self._scan_history = collections.deque(maxlen=N_FRAMES)

        # Paths
        log_dir = os.path.expanduser(_C.get("log_dir", "~/tb3_drl_logs/phase3"))
        model_dir = os.path.expanduser(
            os.path.join(_C.get("sac_model_dir", "~/tb3_drl_models/sac"), run_id))
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self._model_dir  = model_dir
        if EVAL_ONLY:
            self._log_path   = os.path.join(log_dir, f"{run_id}_eval.csv")
            self._update_log = os.path.join(log_dir, f"{run_id}_eval_updates.csv")
        else:
            self._log_path   = os.path.join(log_dir, f"{run_id}.csv")
            self._update_log = os.path.join(log_dir, f"{run_id}_updates.csv")
        self._events_log = os.path.join(log_dir, f"{run_id}_events.csv")
        self._run_id     = run_id
        self._run_note   = RUN_NOTE

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
        self._last_reset_time = 0.0

        # Resume or fresh
        if self._fresh:
            self._log_info("[SACv10] FRESH START — ignoring checkpoints.")
        else:
            self._resume()

        # CSV headers
        if self._ep == 0:
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode", "reward", "steps", "collision", "goal_reached",
                    "sr_100", "mean_steps_100", "updates", "elapsed_s"])
            with open(self._update_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "update", "episode", "actor_loss", "critic_loss",
                    "entropy", "alpha", "buffer_size", "elapsed_s"])
        if not os.path.exists(self._events_log):
            with open(self._events_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "event", "episode", "phase", "updates", "elapsed_s", "run_id", "note"])

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
        param_count = sum(p.numel() for p in self.agent.actor.parameters())
        self._log_info(
            f"[SACv10] Ready  run={run_id}  DZ_MODE={DZ_MODE}  mode={mode_str}  "
            f"obs_dim={OBS_DIM}  n_step={N_STEP}  frames={N_FRAMES}  "
            f"stam_heads={STAM_HEADS}  actor_hidden={HIDDEN}  critic_hidden={CRITIC_HIDDEN}  "
            f"huber_delta={HUBER_DELTA}  actor_params={param_count:,}  "
            f"reverse={'on' if REVERSE_ENABLED else 'off'}  "
            f"ent={ENT_START}→{ENT_END}/{ENT_ANNEAL}eps  "
            f"alpha={self.agent.alpha:.3f}  device={self.device}")
        if self._run_note:
            self._log_info(f"[SACv10] Note: {self._run_note}")

        start_event = "fresh_start" if self._fresh or self._ep == 0 else "resume_start"
        self._log_event(start_event, self._run_note)

    def _clean_log_text(self, text):
        if text.startswith("[SACv10] "):
            return text[len("[SACv10] "):]
        return text

    def _log_info(self, text):
        print(f"SAC v10 | {self._clean_log_text(text)}", flush=True)

    def _log_error(self, text):
        print(f"SAC v10 | ERROR | {self._clean_log_text(text)}", file=sys.stderr, flush=True)

    # ── frame stacking ─────────────────────────────────────────────────
    def _build_stacked_obs(self, raw_obs):
        """Convert 54-dim env obs → 78-dim stacked obs."""
        scan_t = np.array(raw_obs[:N_SECTORS], dtype=np.float32)
        self._scan_history.append(scan_t)
        # Pad if not enough frames yet
        while len(self._scan_history) < N_FRAMES:
            self._scan_history.appendleft(scan_t.copy())
        nav = np.array(raw_obs[48:54], dtype=np.float32)    # skip scan_vel
        # Stack: most recent first (scan_t, scan_{t-1}, scan_{t-2})
        stacked = np.concatenate(
            [self._scan_history[N_FRAMES - 1 - i] for i in range(N_FRAMES)]
            + [nav], dtype=np.float32)
        return stacked   # (78,)

    # ── scheduled entropy target ────────────────────────────────────────
    @property
    def _target_entropy(self):
        t = min(1.0, self._ep / max(1, ENT_ANNEAL))
        return ENT_START + t * (ENT_END - ENT_START)

    # ── resume ─────────────────────────────────────────────────────────
    def _resume(self):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
        if not ckpts:
            return
        last_error = None
        ckpt = None
        for path in reversed(ckpts):
            try:
                if os.path.getsize(path) == 0:
                    continue
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                break
            except Exception as e:
                last_error = e
                continue
        if ckpt is None:
            if last_error is not None:
                self._log_error(
                    f"[SACv10] Resume failed (no valid checkpoints: {last_error}) — fresh start.")
            return
        try:
            self.agent.load_state_dict(ckpt["agent"], self.device)
            if not EVAL_ONLY:
                self._ep = int(ckpt["episode"])
                self._total_steps = int(ckpt.get("total_steps", 0))
                self._updates = int(ckpt["updates"])
                self._outcomes = list(ckpt.get("outcomes", []))
                self._steps_hist = list(ckpt.get("steps_hist", []))
                self._elapsed_offset = float(ckpt.get("elapsed_s", 0.0))
                self._best_sr = float(ckpt.get("best_sr", 0.0))
            else:
                self._ep = 0
                self._total_steps = 0
                self._updates = int(ckpt["updates"])
                self._outcomes = []
                self._steps_hist = []
                self._elapsed_offset = 0.0
                self._best_sr = 0.0
            # Replay buffer
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
            self._log_info(
                f"[SACv10] Resumed ep={self._ep}  buf={len(self.replay)}  "
                f"upd={self._updates}  elapsed={self._elapsed_offset/3600:.1f}h")
            self._log_event("resume_loaded", f"checkpoint_ep={self._ep}")
        except Exception as e:
            self._log_error(f"[SACv10] Resume failed ({e}) — fresh start.")

    def _log_event(self, event, note=""):
        elapsed = self._elapsed_offset + max(0.0, time.time() - self._t0)
        with open(self._events_log, "a", newline="") as f:
            csv.writer(f).writerow([
                event,
                self._ep,
                self._cur_phase,
                self._updates,
                round(elapsed, 1),
                self._run_id,
                note,
            ])

    # ── checkpoint ─────────────────────────────────────────────────────
    def _save_ckpt(self, include_replay=False):
        path = os.path.join(self._model_dir, f"ckpt_ep{self._ep:06d}.pt")
        tmp_path = f"{path}.tmp"
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
        }, tmp_path)
        os.replace(tmp_path, path)
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
        path = os.path.join(self._model_dir, "replay_buffer.pt")
        tmp_path = f"{path}.tmp"
        torch.save({
            "replay": self.replay.to_list(),
            "episode": self._ep,
        }, tmp_path)
        os.replace(tmp_path, path)

    def _shutdown(self, signum, frame):
        self._log_info("[SACv10] Shutdown — saving with replay…")
        self._log_event("shutdown", f"signal={signum}")
        self.nstep.reset()
        self._save_ckpt(include_replay=True)
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass

    # ── ROS callbacks ──────────────────────────────────────────────────
    def _on_reset(self, msg):
        raw = np.array(msg.data[:RAW_OBS], dtype=np.float32)
        if not np.all(np.isfinite(raw)):
            return
        self._scan_history.clear()
        self.nstep.reset()
        self._current_obs = self._build_stacked_obs(raw)
        self._current_act = None
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._waiting_reset = False
        self._last_reset_time = time.time()
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

        if done and self._is_impossible_terminal(info):
            return

        nobs = self._build_stacked_obs(raw_nobs)
        if not (np.all(np.isfinite(nobs)) and math.isfinite(reward)):
            return

        self._ep_reward += reward
        self._ep_steps += 1
        self._total_steps += 1

        if not EVAL_ONLY:
            # Push through n-step buffer → PER replay
            transitions = self.nstep.add(
                self._current_obs, self._current_act, reward, nobs, float(done))
            for t in transitions:
                self.replay.add(*t)

            # SAC gradient updates
            if (self._total_steps > WARMUP
                    and len(self.replay) >= BATCH
                    and self._total_steps % UPDATE_EVERY == 0):
                for _ in range(GRAD_STEPS):
                    la, lc, ent, alpha = self.agent.update(
                        self.replay, self._target_entropy, self._total_steps)
                    self._updates += 1
                    if self._updates % LOG_EVERY == 0:
                        elapsed = self._elapsed_offset + (time.time() - self._t0)
                        with open(self._update_log, "a", newline="") as f:
                            csv.writer(f).writerow([
                                self._updates, self._ep,
                                round(la, 6), round(lc, 6),
                                round(ent, 6), round(alpha, 6),
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
            self._log_info(f"[SACv10] Phase {old} → {msg.data}")
            self._log_event("phase_advance", f"from={old},to={msg.data}")

    def _on_goal(self, msg):
        try:
            x, y = [float(v) for v in msg.data.split(",")]
            self._cur_goal = f"{_goal_label(x, y)} ({x:+.1f},{y:+.1f})"
        except Exception:
            self._cur_goal = "?"

    def _is_impossible_terminal(self, info):
        if info == 0:
            return False
        elapsed = time.time() - self._last_reset_time
        next_step = self._ep_steps + 1
        if elapsed < RESET_SETTLE_S:
            return True
        if info == 8 and next_step < MIN_STUCK_STEPS:
            return True
        if info == 4 and next_step < MIN_TIMEOUT_STEPS:
            return True
        return False

    # ── episode end ────────────────────────────────────────────────────
    def _end_episode(self, info):
        if info == 0:      # spawn artifact
            self._waiting_reset = True
            return

        self._ep += 1
        goal = 1 if info == 1 else 0
        collision = 1 if info == 2 else 0

        self._outcomes.append(goal)
        self._steps_hist.append(self._ep_steps)
        if len(self._outcomes) > 100:
            self._outcomes.pop(0)
        if len(self._steps_hist) > 100:
            self._steps_hist.pop(0)
        sr = 100.0 * sum(self._outcomes) / len(self._outcomes)
        ms = sum(self._steps_hist) / len(self._steps_hist)
        elapsed = self._elapsed_offset + (time.time() - self._t0)

        if goal:
            terminal_tag = "GOAL(1)"
        elif collision:
            terminal_tag = "COLL(0)"
        else:
            terminal_tag = "STUK(0)" if info == 8 else "TIME(0)"
        warmup_str = (f"  [WARMUP {self._total_steps}/{WARMUP}]"
                      if self._total_steps < WARMUP else "")
        self._log_info(
            f"Ep {self._ep:5d} | Ph{self._cur_phase} | goal={self._cur_goal} | "
            f"{terminal_tag} | "
            f"R={self._ep_reward:+8.1f} | steps={self._ep_steps:4d} | "
            f"SR={sr:5.1f}% | H_tgt={self._target_entropy:.2f} | "
            f"buf={len(self.replay):,} | upd={self._updates}{warmup_str}")

        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self._ep, round(self._ep_reward, 2), self._ep_steps,
                collision, goal, round(sr, 2), round(ms, 1),
                self._updates, round(elapsed, 1)])

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
                self._log_info(
                    f"[SACv10] *** NEW BEST SR={sr:.1f}% at ep={self._ep} ***")

        # Auto-stop
        if MAX_EPISODES > 0 and self._ep >= MAX_EPISODES:
            self._log_info(
                f"[SACv10] MAX_EPISODES={MAX_EPISODES} reached — SR={sr:.1f}%")
            self._save_ckpt()
            rclpy.shutdown()
            return

        self._waiting_reset = True

    # ── action ─────────────────────────────────────────────────────────
    def _send_action(self, obs):
        if EVAL_ONLY:
            tanh_act = self.agent.select_action(obs, deterministic=True)
        elif self._total_steps < WARMUP:
            tanh_act = np.random.uniform(-1, 1, ACT_DIM).astype(np.float32)
        else:
            tanh_act = self.agent.select_action(obs)

        self._current_act = tanh_act

        reverse_unlocked = False
        if REVERSE_ENABLED and len(obs) >= 2 and self._total_steps >= WARMUP:
            min_scan_m = float(obs[-2]) * 3.5
            min_approach = float(obs[-1])
            reverse_unlocked = (
                min_scan_m < REVERSE_UNLOCK_SCAN_DIST
                or min_approach < REVERSE_UNLOCK_APPROACH
            )

        if reverse_unlocked:
            lin = float(-REVERSE_SPEED + (tanh_act[0] + 1.0) / 2.0 * (LIN_MAX + REVERSE_SPEED))
            if lin > 0.0:
                lin = min(lin, LIN_MAX * max(0.0, min(1.0, REVERSE_CAUTION_FORWARD_FACTOR)))
        else:
            lin = float(LIN_MIN + (tanh_act[0] + 1.0) / 2.0 * (LIN_MAX - LIN_MIN))

        ang = float(tanh_act[1] * ANG_MAX)
        out = Float32MultiArray()
        out.data = [lin, ang]
        self._act_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrainAgentSACv10()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
