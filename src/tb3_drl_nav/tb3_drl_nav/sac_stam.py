#!/usr/bin/env python3
"""
sac_v8.py  Ablation Study: SAC + STAM + PER + 3-step
=========================================================================
Ablation role : Adds spatial-temporal attention + experience efficiency.
Architecture  : STAM v2 (multi-head attention over 3 LiDAR frames)
                + Residual MLP Actor (256) + Twin Residual Critic (256)
Observation   : 78-dim  (24×3 frame-stack + 6 nav)
Replay        : Prioritized Experience Replay  (200 K)
Returns       : 3-step TD  (faster credit assignment)
Critic loss   : IS-weighted MSE
Entropy target: Annealed  −1.0 → −2.0 over 500 episodes

Differences from sac_baseline:
  [+] STAM v2 — multi-head self-attention lets sectors attend to each other
  [+] 3-frame temporal LiDAR stack — explicit obstacle velocity signal
  [+] PER — prioritized replay focuses on high-error transitions
  [+] 3-step returns — propagates goal reward faster
  [+] Residual blocks — deeper per-timestep feature extraction
  [+] Entropy annealing — explore first, exploit later

Compare against:
  sac_baseline.py — no attention, no PER, 1-step, MSE
  sac_v10.py      — adds wider critic (384) + Huber loss
  sac_v11.py      — replaces frame-stack with GRU recurrence

Run:
  SEED=42 ros2 run tb3_drl_nav sac_v8 --ros-args \\
      -p run_id:=sac_v8_s42 -p fresh:=true
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
from std_msgs.msg import Float32MultiArray, Int32, String

_LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "phase3_ppo.yaml")
try:
    import yaml
    with open(_CFG_PATH) as _f:
        _C = yaml.safe_load(_f) or {}
except Exception:
    _C = {}

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED         = int(os.environ.get("SEED", _C.get("seed", 42)))
RAW_OBS      = _C.get("obs_dim", 54)
ACT_DIM      = _C.get("act_dim", 2)
LIN_MIN      = _C.get("lin_min", 0.0)
LIN_MAX      = _C.get("lin_max", 0.26)
ANG_MAX      = _C.get("ang_max", 1.82)
N_SECTORS    = 24

# ── [ABLATION] 3-frame temporal stack ─────────────────────────────────────────
N_FRAMES     = _C.get("sac_v8_frame_stack", 3)
OBS_DIM      = N_SECTORS * N_FRAMES + 6    # 78

# ── [ABLATION] Network width — 256 for both actor and critic ──────────────────
HIDDEN       = _C.get("sac_hidden", 256)

BUFFER_CAP   = _C.get("sac_buffer",    200_000)
BATCH        = _C.get("sac_batch",     256)
GAMMA        = _C.get("gamma",         0.99)
TAU          = _C.get("sac_tau",       0.005)
LR_ACTOR     = _C.get("sac_lr_actor",  3e-4)
LR_CRITIC    = _C.get("sac_lr_critic", 3e-4)
LR_ALPHA     = _C.get("sac_lr_alpha",  3e-4)
WARMUP       = _C.get("sac_warmup",    10_000)
GRAD_CLIP    = _C.get("sac_grad_clip", 1.0)
SAVE_EVERY   = _C.get("save_every",    50)
KEEP_CKPTS   = _C.get("keep_ckpts",    8)
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", _C.get("max_episodes", 0)))

# ── [ABLATION] PER parameters ─────────────────────────────────────────────────
PER_ALPHA        = _C.get("sac_v8_per_alpha",      0.6)
PER_BETA0        = _C.get("sac_v8_per_beta_start", 0.4)
PER_BETA_FRAMES  = _C.get("sac_v8_per_beta_frames", 500_000)

# ── [ABLATION] 3-step returns ─────────────────────────────────────────────────
N_STEP       = _C.get("sac_v8_nstep", 3)

# ── [ABLATION] STAM parameters ────────────────────────────────────────────────
STAM_HEADS   = _C.get("sac_v8_stam_heads",   2)
STAM_D       = _C.get("sac_v8_stam_d_model", 16)

# ── [ABLATION] Entropy annealing (positive values) ────────────────────────────
ENT_START    = float(_C.get("sac_v8_entropy_start",      1.0))
ENT_END      = float(_C.get("sac_v8_entropy_end",        ACT_DIM))
ENT_ANNEAL   = int(_C.get("sac_v8_entropy_anneal_eps",    500))

EVAL_ONLY    = os.environ.get("EVAL_ONLY", "").lower() in ("true", "1", "yes")
RESET_SETTLE_S = float(_C.get("trainer_runtime", {}).get("terminal_filters", {}).get("reset_settle_s", 0.10))
MIN_STUCK_STEPS = int(_C.get("trainer_runtime", {}).get("terminal_filters", {}).get("min_stuck_steps", 60))
MIN_TIMEOUT_STEPS = int(_C.get("trainer_runtime", {}).get("terminal_filters", {}).get("min_timeout_steps", 500))

GOAL_CATALOG = [
    # Core (1.2-2.1m) — Phase 1 goals
    (0.0, 1.2), (0.0, -1.2), (0.0, 1.9), (0.0, -1.9), (1.4, 1.4), (-1.4, 1.4), (1.4, -1.4),
    # Expand (Corridor entrances 2.5-3.2m) — Phase 2
    (0.0, 2.5), (0.0, -2.5), (2.5, 0.0), (-2.5, 0.0), (3.2, 1.5), (-3.2, 1.5), (-3.2, -1.5),
    # Mid (Room entrances 3.8m dia) — Phase 3
    (-2.8, 2.8), (2.8, 2.8), (-2.5, -2.5), (2.8, -2.8),
    # Inter (Bridge goals) — Phase 4
    (-3.3, 2.2), (3.3, 2.2), (-3.3, -2.2), (3.3, -2.2),
    (-0.5, 3.0), (0.5, 3.0), (0.0, -3.1), (-0.5, -3.0), (0.5, -3.0),
    # Deep (Extreme rooms & corridors) — Phase 5+
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
#  PRIORITIZED EXPERIENCE REPLAY  — [ABLATION] vs uniform in baseline
# ══════════════════════════════════════════════════════════════════════════════
class SumTree:
    __slots__ = ("capacity", "tree", "data", "write_idx", "n_entries")

    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, delta):
        p = (idx - 1) // 2
        self.tree[p] += delta
        if p > 0: self._propagate(p, delta)

    def _leaf(self, s):
        idx = 0
        while idx < self.capacity - 1:
            l = 2 * idx + 1
            if s <= self.tree[l]:
                idx = l
            else:
                s -= self.tree[l]
                idx = l + 1
        return idx

    @property
    def total(self): return self.tree[0]

    def add(self, priority, data):
        ti = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        delta = priority - self.tree[ti]
        self.tree[ti] = priority
        self._propagate(ti, delta)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, ti, priority):
        delta = priority - self.tree[ti]
        self.tree[ti] = priority
        self._propagate(ti, delta)

    def get(self, s):
        idx = self._leaf(s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]


class PrioritizedReplayBuffer:
    _MIN_P = 1e-6

    def __init__(self, capacity, alpha=0.6, beta0=0.4, beta_frames=500_000):
        self.tree        = SumTree(capacity)
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta0       = beta0
        self.beta_frames = beta_frames
        self._max_p      = 1.0

    def __len__(self):   return self.tree.n_entries

    def add(self, obs, act, rew, nobs, done):
        self.tree.add(self._max_p,
                      (np.asarray(obs, np.float32), np.asarray(act, np.float32),
                       float(rew), np.asarray(nobs, np.float32), float(done)))

    def sample(self, n, device, total_steps):
        beta    = min(1.0, self.beta0 + total_steps * (1.0 - self.beta0) / self.beta_frames)
        seg     = self.tree.total / n
        idxs, priorities, batch = [], [], []
        for i in range(n):
            s = random.uniform(seg * i, seg * (i + 1))
            ti, p, d = self.tree.get(s)
            if d is None:
                ti, p, d = self.tree.get(random.uniform(0, self.tree.total))
            idxs.append(ti); priorities.append(max(p, self._MIN_P)); batch.append(d)
        probs   = np.array(priorities) / self.tree.total
        weights = (len(self) * probs) ** (-beta)
        weights /= weights.max()
        o, a, r, no, d = zip(*batch)
        t = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)
        return (t(o), t(a), t(r).unsqueeze(1), t(no), t(d).unsqueeze(1),
                idxs, torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1))

    def update_priorities(self, idxs, td_errors):
        for ti, td in zip(idxs, td_errors):
            p = (abs(td) + self._MIN_P) ** self.alpha
            self._max_p = max(self._max_p, p)
            self.tree.update(ti, p)

    def to_list(self):
        return [self.tree.data[i] for i in range(self.tree.n_entries)
                if self.tree.data[i] is not None]

    @classmethod
    def from_list(cls, data, cap, alpha=0.6, beta0=0.4, beta_frames=500_000):
        buf = cls(cap, alpha, beta0, beta_frames)
        for item in data[-cap:]:
            buf.tree.add(buf._max_p, tuple(item))
        return buf


# ══════════════════════════════════════════════════════════════════════════════
#  N-STEP RETURN BUFFER  — [ABLATION] 3-step vs 1-step in baseline
# ══════════════════════════════════════════════════════════════════════════════
class NStepBuffer:
    def __init__(self, n=3, gamma=0.99):
        self.n = n; self.gamma = gamma
        self.buf = collections.deque(maxlen=n)

    def add(self, obs, act, rew, nobs, done):
        self.buf.append((obs, act, rew, nobs, done))
        if done:
            results = []
            while self.buf:
                results.append(self._compute()); self.buf.popleft()
            return results
        elif len(self.buf) == self.n:
            r = [self._compute()]; self.buf.popleft(); return r
        return []

    def _compute(self):
        obs0, act0 = self.buf[0][0], self.buf[0][1]
        R = 0.0
        for i, (_, _, r, nobs, done) in enumerate(self.buf):
            R += (self.gamma ** i) * r
            if done: return obs0, act0, R, nobs, True
        return obs0, act0, R, self.buf[-1][3], False

    def reset(self): self.buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  STAM v2  — [ABLATION] multi-head self-attention over LiDAR sectors
# ══════════════════════════════════════════════════════════════════════════════
class MultiHeadScanAttention(nn.Module):
    """
    Multi-head self-attention over 24 LiDAR sectors.
    Input  : (B, 24, N_FRAMES) — temporal scan per sector
    Output : (B, 48) — compressed attention-weighted representation
    """
    def __init__(self, n_sectors=N_SECTORS, n_frames=N_FRAMES,
                 d_model=STAM_D, n_heads=STAM_HEADS, d_out=48):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_sectors = n_sectors
        self.n_heads   = n_heads
        self.d_k       = d_model // n_heads
        self.d_model   = d_model

        self.proj_in  = nn.Linear(n_frames, d_model)
        self.pos_enc  = nn.Parameter(torch.randn(1, n_sectors, d_model) * 0.02)
        self.W_qkv    = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.compress = nn.Linear(n_sectors * d_model, d_out)

        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.W_qkv.weight)
        nn.init.xavier_uniform_(self.compress.weight)
        nn.init.zeros_(self.compress.bias)

    def forward(self, x):
        B   = x.size(0)
        h   = self.proj_in(x) + self.pos_enc
        qkv = self.W_qkv(h).reshape(B, self.n_sectors, 3, self.n_heads, self.d_k
                          ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, self.n_sectors, self.d_model)
        return self.compress(self.proj_out(out).reshape(B, -1))


class ResidualBlock(nn.Module):
    """Two-layer residual block with pre-activation."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim); self.fc2 = nn.Linear(dim, dim)
        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2)); nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return F.relu(x + self.fc2(F.relu(self.fc1(x))))


def _stam_split(obs):
    """Split 78-dim obs → (B,24,3) scan frames + (B,6) nav."""
    return obs[:, :N_SECTORS * N_FRAMES].reshape(-1, N_SECTORS, N_FRAMES), \
           obs[:, N_SECTORS * N_FRAMES:]


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stam = MultiHeadScanAttention()     # 48-dim output
        self.ln   = nn.LayerNorm(48 + 6)
        self.fc1  = nn.Linear(48 + 6, HIDDEN)
        self.fc2  = nn.Linear(HIDDEN, HIDDEN)
        self.res  = ResidualBlock(HIDDEN)
        self.mu      = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(HIDDEN, ACT_DIM)
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2)); nn.init.zeros_(m.bias)
        for m in [self.mu, self.log_std]:
            nn.init.orthogonal_(m.weight, gain=0.01); nn.init.zeros_(m.bias)

    def forward(self, obs):
        frames, nav = _stam_split(obs)
        x = self.ln(torch.cat([self.stam(frames), nav], dim=-1))
        x = self.res(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return self.mu(x), self.log_std(x).clamp(-5.0, 2.0).exp()

    def sample(self, obs):
        mu, std = self(obs)
        raw = Normal(mu, std).rsample(); act = torch.tanh(raw)
        lp  = (Normal(mu, std).log_prob(raw) -
               torch.log(1.0 - act.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return act, lp

    def deterministic(self, obs):
        mu, _ = self(obs); return torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.stam1 = MultiHeadScanAttention()
        self.stam2 = MultiHeadScanAttention()
        d = 48 + 6 + ACT_DIM
        self.q1_fc1 = nn.Linear(d, HIDDEN); self.q1_fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.q1_res = ResidualBlock(HIDDEN); self.q1_out = nn.Linear(HIDDEN, 1)
        self.q2_fc1 = nn.Linear(d, HIDDEN); self.q2_fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.q2_res = ResidualBlock(HIDDEN); self.q2_out = nn.Linear(HIDDEN, 1)
        for m in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2)); nn.init.zeros_(m.bias)
        for m in [self.q1_out, self.q2_out]:
            nn.init.orthogonal_(m.weight, gain=1.0); nn.init.zeros_(m.bias)

    def forward(self, obs, act):
        frames, nav = _stam_split(obs)
        x1 = torch.cat([self.stam1(frames), nav, act], dim=-1)
        x2 = torch.cat([self.stam2(frames), nav, act], dim=-1)
        q1 = self.q1_out(self.q1_res(F.relu(self.q1_fc2(F.relu(self.q1_fc1(x1))))))
        q2 = self.q2_out(self.q2_res(F.relu(self.q2_fc2(F.relu(self.q2_fc1(x2))))))
        return q1, q2

    def q_min(self, obs, act): return torch.min(*self(obs, act))


# ══════════════════════════════════════════════════════════════════════════════
#  SAC AGENT
# ══════════════════════════════════════════════════════════════════════════════
class SAC:
    def __init__(self, device):
        self.device     = device
        self.actor      = Actor().to(device)
        self.critic     = Critic().to(device)
        self.critic_tgt = Critic().to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters(): p.requires_grad = False
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha  = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

    @property
    def alpha(self): return self.log_alpha.exp().item()

    def select_action(self, obs, deterministic=False):
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor.deterministic(t) if deterministic else self.actor.sample(t)[0]
        return a.squeeze(0).cpu().numpy()

    def update(self, replay, target_h, total_steps):
        obs, act, rew, nobs, done, idxs, is_w = replay.sample(BATCH, self.device, total_steps)
        alpha = self.log_alpha.exp().detach()

        # ── Critic — [ABLATION] 3-step discount, IS-weighted MSE ─────────────
        with torch.no_grad():
            na, nlp = self.actor.sample(nobs)
            tq1, tq2 = self.critic_tgt(nobs, na)
            target = rew + (GAMMA ** N_STEP) * (1 - done) * (torch.min(tq1, tq2) - alpha * nlp)
        q1, q2  = self.critic(obs, act)
        loss_c  = (is_w * (q1 - target).pow(2)).mean() + \
                  (is_w * (q2 - target).pow(2)).mean()
        self.opt_critic.zero_grad(); loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()
        td_err = torch.max((q1 - target).abs(), (q2 - target).abs()
                           ).squeeze(1).detach().cpu().numpy()
        replay.update_priorities(idxs, td_err)

        # ── Actor ─────────────────────────────────────────────────────────────
        for p in self.critic.parameters(): p.requires_grad = False
        new_act, new_lp = self.actor.sample(obs)
        loss_a = (alpha * new_lp - self.critic.q_min(obs, new_act)).mean()
        self.opt_actor.zero_grad(); loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
        for p in self.critic.parameters(): p.requires_grad = True

        # ── Temperature — [FIXED] alpha increase when entropy < target ─────────
        loss_t = (self.log_alpha * (-new_lp.detach() - target_h)).mean()
        self.opt_alpha.zero_grad(); loss_t.backward(); self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - TAU).add_(TAU * p.data)

        return loss_a.item(), loss_c.item(), -new_lp.mean().item(), self.alpha

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    critic_tgt=self.critic_tgt.state_dict(),
                    opt_actor=self.opt_actor.state_dict(),
                    opt_critic=self.opt_critic.state_dict(),
                    log_alpha=self.log_alpha.detach().cpu(),
                    opt_alpha=self.opt_alpha.state_dict())

    def load_state_dict(self, d, device):
        self.actor.load_state_dict(d["actor"], strict=False)
        self.critic.load_state_dict(d["critic"], strict=False)
        self.critic_tgt.load_state_dict(d.get("critic_tgt", d["critic"]), strict=False)
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_critic.load_state_dict(d["opt_critic"])
        la = d["log_alpha"]
        self.log_alpha = (la if isinstance(la, torch.Tensor) else torch.tensor([la])
                          ).clone().to(device).requires_grad_(True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
        if "opt_alpha" in d: self.opt_alpha.load_state_dict(d["opt_alpha"])


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 TRAINING NODE
# ══════════════════════════════════════════════════════════════════════════════
class TrainNode(Node):
    NODE_NAME   = "sac_v8"
    DEFAULT_RUN = "sac_v8_s42"

    def __init__(self):
        super().__init__(self.NODE_NAME)
        self.declare_parameter("run_id", self.DEFAULT_RUN)
        self.declare_parameter("fresh",  False)
        run_id      = self.get_parameter("run_id").value
        self._fresh = self.get_parameter("fresh").value

        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent  = SAC(self.device)
        self.replay = PrioritizedReplayBuffer(BUFFER_CAP, PER_ALPHA, PER_BETA0, PER_BETA_FRAMES)
        self.nstep  = NStepBuffer(N_STEP, GAMMA)

        # 3-frame scan history for frame stacking
        self._scan_history = collections.deque(maxlen=N_FRAMES)

        log_dir   = os.path.expanduser(_C.get("log_dir", "~/tb3_drl_logs/phase3"))
        model_dir_base = os.environ.get("SAC_MODEL_DIR", _C.get("sac_model_dir", "~/tb3_drl_models/sac"))
        model_dir = os.path.expanduser(os.path.join(model_dir_base, run_id))
        os.makedirs(log_dir,   exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self._model_dir = model_dir
        suffix = "_eval" if EVAL_ONLY else ""
        _ph = os.environ.get("EVAL_PHASE", "")
        _ph_tag = f"_ph{_ph}" if (EVAL_ONLY and _ph) else ""
        _eval_tag = os.environ.get("EVAL_TAG", "").strip()
        _eval_tag = f"_{_eval_tag}" if (EVAL_ONLY and _eval_tag) else ""
        self._log_path = os.path.join(log_dir, f"{run_id}{_ph_tag}{_eval_tag}{suffix}.csv")

        self._ep = self._total_steps = self._updates = 0
        self._ep_reward = self._ep_steps = 0
        self._best_sr = self._elapsed_offset = 0.0
        self._t0      = time.time()
        self._outcomes   = collections.deque(maxlen=100)
        self._steps_hist = collections.deque(maxlen=100)
        self._waiting_reset = True
        self._current_obs = self._current_act = None
        self._cur_phase   = 1
        self._cur_goal    = "?"
        self._cur_goal_x  = float("nan")
        self._cur_goal_y  = float("nan")
        self._last_alpha  = 1.0
        # [ABLATION] entropy annealing schedule
        self._target_h    = ENT_START
        self._last_reset_time = time.time()

        if self._fresh:
            print(f"[{self.NODE_NAME}] FRESH START.", flush=True)
        else:
            resumed = self._resume(run_id)
            if EVAL_ONLY and not resumed:
                raise RuntimeError(
                    f"[{self.NODE_NAME}] EVAL_ONLY requested but no checkpoint or actor weights were found for run_id='{run_id}' in {self._model_dir}"
                )

        # ── Eval isolation: reset all rolling stats so CSV is self-contained ──
        if EVAL_ONLY:
            # [resume-patch] resume from however many episodes the CSV already holds
            _eval_done = 0
            if os.path.exists(self._log_path):
                try:
                    with open(self._log_path) as _rf:
                        _eval_done = sum(1 for _l in _rf
                                         if _l.split(',')[0].strip().isdigit())
                except OSError:
                    _eval_done = 0
            self._ep = _eval_done
            self._elapsed_offset  = 0.0
            self._t0              = time.time()
            self._outcomes.clear()
            self._steps_hist.clear()

        if self._ep == 0:            # [resume-patch] never truncate a resumed CSV
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode", "reward", "steps", "collision", "goal_reached",
                    "terminal_type", "sr_100", "mean_steps_100", "updates", "buffer_size",
                    "alpha", "phase", "elapsed_s", "goal_x", "goal_y", "goal_dist", "goal_cat"])

        self._act_pub = self.create_publisher(
            Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self._res_pub = self.create_publisher(
            Float32MultiArray, "/tb3_drl/training_result", 10)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/reset_obs",   self._on_reset, _LATCHED)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/step_result", self._on_step,  10)
        self.create_subscription(
            Int32,            "/tb3_drl/curriculum_phase", self._on_phase, _LATCHED)
        self.create_subscription(
            String,           "/tb3_drl/goal",             self._on_goal,  _LATCHED)

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        mode = "EVAL" if EVAL_ONLY else f"TRAIN  warmup={WARMUP:,}"
        print(f"\n[{self.NODE_NAME}] {mode}  device={self.device}  "
              f"ep={self._ep}  seed={SEED}  nstep={N_STEP}", flush=True)
        print("=" * 80, flush=True)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def _save_ckpt(self, tag=None):
        label = tag or f"ep{self._ep:06d}"
        path  = os.path.join(self._model_dir, f"ckpt_{label}.pt")
        elapsed = self._elapsed_offset + (time.time() - self._t0)

        # [FIX] Save replay buffer + scan history + outcome tracking to preserve full training state
        replay_data = self.replay.to_list() if len(self.replay) > 0 else []
        scan_hist = list(self._scan_history) if hasattr(self, '_scan_history') else []
        outcomes_list = list(self._outcomes)  # preserve SR-100 tracking
        steps_list = list(self._steps_hist)   # preserve mean steps tracking

        torch.save({
            'agent': self.agent.state_dict(),
            'episode': self._ep,
            'total_steps': self._total_steps,
            'updates': self._updates,
            'best_sr': self._best_sr,
            'elapsed_s': elapsed,
            'replay_buffer': replay_data,
            'scan_history': scan_hist,
            'outcomes': outcomes_list,
            'steps_hist': steps_list,
            'last_alpha': self._last_alpha,
            'cur_phase': self._cur_phase,
        }, path)
        torch.save(self.agent.actor.state_dict(),
                   os.path.join(self._model_dir, "actor_latest.pt"))
        if tag is None:
            for old in sorted(glob.glob(
                    os.path.join(self._model_dir, "ckpt_ep*.pt")))[:-KEEP_CKPTS]:
                try: os.remove(old)
                except OSError: pass

    def _resume(self, run_id):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
        actor_latest = os.path.join(self._model_dir, "actor_latest.pt")
        if not ckpts:
            if EVAL_ONLY and os.path.exists(actor_latest):
                try:
                    actor_state = torch.load(actor_latest, map_location="cpu", weights_only=False)
                    self.agent.actor.load_state_dict(actor_state, strict=False)
                    print(
                        f"[{self.NODE_NAME}] Loaded actor_latest.pt for eval-only run_id={run_id}",
                        flush=True,
                    )
                    return True
                except Exception as e:
                    print(f"[{self.NODE_NAME}] Eval weight load failed ({e}).", flush=True)
                    return False
            print(f"[{self.NODE_NAME}] No checkpoint found for run_id={run_id}.", flush=True)
            return False
        try:
            # Load by MODIFICATION TIME (most recent), not alphabetical order
            ckpt_latest = max(ckpts, key=os.path.getmtime)
            ckpt = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
            self.agent.load_state_dict(ckpt["agent"], self.device)
            self._ep             = ckpt.get("episode",     0)
            self._total_steps    = ckpt.get("total_steps", 0)
            self._updates        = ckpt.get("updates",     0)
            self._best_sr        = ckpt.get("best_sr",     0.0)
            self._elapsed_offset = ckpt.get("elapsed_s",   0.0)
            # Re-seed on resume for consistency
            torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
            
            # --- STABILIZATION BLOCK ---
            if os.environ.get("STABILIZE", "false").lower() == "true":
                print(f"[{self.NODE_NAME}] !!! STABILIZATION ACTIVE: Forcing alpha=0.1 and clearing SR history !!!", flush=True)
                self._last_alpha = 0.1
                # Outcomes and steps will be cleared below in the [FIX] blocks
            else:
                self._last_alpha = ckpt.get("last_alpha",  1.0)
                
            self._cur_phase      = ckpt.get("cur_phase",   1)
            self._target_h       = self._entropy_target()

            # [FIX] Restore replay buffer + scan history + outcome tracking
            if "replay_buffer" in ckpt and ckpt["replay_buffer"]:
                self.replay = PrioritizedReplayBuffer.from_list(
                    ckpt["replay_buffer"], BUFFER_CAP, PER_ALPHA, PER_BETA0, PER_BETA_FRAMES)
                buffer_info = f" | Buf:{len(self.replay)}"
            else:
                buffer_info = " | Buf:EMPTY"

            if "scan_history" in ckpt and ckpt["scan_history"]:
                self._scan_history.clear()
                self._scan_history.extend(ckpt["scan_history"])

            # [FIX] Restore outcomes + steps to preserve SR-100 continuity
            if "outcomes" in ckpt and ckpt["outcomes"] and os.environ.get("STABILIZE", "false").lower() != "true":
                self._outcomes.clear()
                self._outcomes.extend(ckpt["outcomes"])
                sr_info = f" | SR-100:{sum(self._outcomes)/len(self._outcomes)*100:.1f}%"
            else:
                self._outcomes.clear() # Ensure empty if stabilizing
                sr_info = " | SR-100:RESET" if os.environ.get("STABILIZE") else ""

            if "steps_hist" in ckpt and ckpt["steps_hist"] and os.environ.get("STABILIZE", "false").lower() != "true":
                self._steps_hist.clear()
                self._steps_hist.extend(ckpt["steps_hist"])
            else:
                self._steps_hist.clear()

            fname = os.path.basename(ckpt_latest)
            print(
                f"[{self.NODE_NAME}] Resumed from {fname}: ep={self._ep} | Ph{self._cur_phase}{buffer_info}{sr_info}",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"[{self.NODE_NAME}] Resume failed ({e}).", flush=True)
            return False

    def _shutdown(self, *_):
        print(f"\n[{self.NODE_NAME}] Shutdown — saving…", flush=True)
        self._save_ckpt(tag="shutdown")
        try: rclpy.shutdown()
        except Exception: pass

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _build_obs(self, raw):
        """Stack last N_FRAMES LiDAR scans + current nav state."""
        scan = raw[:N_SECTORS]; nav = raw[2 * N_SECTORS:]
        self._scan_history.append(scan)
        while len(self._scan_history) < N_FRAMES:
            self._scan_history.append(scan)
        return np.concatenate(list(self._scan_history) + [nav])

    def _entropy_target(self):
        """[ABLATION] Anneal entropy target normally, but force exploitation in Phase 7."""
        if self._cur_phase == 7:
            return -2.0 # Force high exploitation for expert timing
        if self._ep >= ENT_ANNEAL:
            return ENT_END
        return ENT_START + (ENT_END - ENT_START) * self._ep / ENT_ANNEAL

    def _is_impossible_terminal(self, info):
        elapsed = time.time() - self._last_reset_time
        next_step = self._ep_steps + 1
        if info == 0:
            return True
        if elapsed < RESET_SETTLE_S:
            return True
        if info == 8 and next_step < MIN_STUCK_STEPS:
            return True
        if info == 4 and next_step < MIN_TIMEOUT_STEPS:
            return True
        return False

    # ── ROS callbacks ─────────────────────────────────────────────────────────
    def _on_phase(self, msg: Int32): self._cur_phase = msg.data

    def _on_goal(self, msg: String):
        try:
            x, y = [float(v) for v in msg.data.split(",")]
            self._cur_goal   = f"{_goal_label(x, y)} ({x:+.1f},{y:+.1f})"
            self._cur_goal_x = round(x, 2)
            self._cur_goal_y = round(y, 2)
        except Exception:
            self._cur_goal   = "?"
            self._cur_goal_x = float("nan")
            self._cur_goal_y = float("nan")

    def _on_reset(self, msg: Float32MultiArray):
        raw = np.array(msg.data[:2 * N_SECTORS + 6], dtype=np.float32)
        self._scan_history.clear()
        obs = self._build_obs(raw)
        self._current_obs   = obs
        self._current_act   = None
        self._ep_reward     = 0.0
        self._ep_steps      = 0
        self._waiting_reset = False
        self._last_reset_time = time.time()
        self.nstep.reset()

        self._send_action(obs)

    def _on_step(self, msg: Float32MultiArray):
        if self._waiting_reset or self._current_act is None: return
        raw    = np.array(msg.data[:2 * N_SECTORS + 6], dtype=np.float32)
        reward = float(msg.data[2 * N_SECTORS + 6])
        done   = bool(msg.data[2 * N_SECTORS + 7])
        info   = int(msg.data[2 * N_SECTORS + 8])

        if done and self._is_impossible_terminal(info):
            return

        nobs = self._build_obs(raw)
        self._ep_reward   += reward
        self._ep_steps    += 1
        self._total_steps += 1

        # [ABLATION] n-step buffer → PER
        for t in self.nstep.add(self._current_obs, self._current_act, reward, nobs, done):
            self.replay.add(*t)
        self._current_obs = nobs

        if (not EVAL_ONLY
                and self._total_steps > WARMUP
                and len(self.replay) >= BATCH):
            self._target_h = self._entropy_target()
            _, _, _, alpha_val = self.agent.update(
                self.replay, self._target_h, self._total_steps)
            self._last_alpha = alpha_val
            self._updates   += 1

        if done:
            self._ep += 1
            goal = 1 if info == 1 else 0
            coll = 1 if info == 2 else 0
            self._outcomes.append(goal)
            self._steps_hist.append(self._ep_steps)
            sr = sum(self._outcomes) / len(self._outcomes) * 100
            ms = sum(self._steps_hist) / len(self._steps_hist)
            elapsed = self._elapsed_offset + (time.time() - self._t0)
            wup = f" | warmup {self._total_steps}/{WARMUP}" if self._total_steps < WARMUP else ""

            terminal_tag = "GOAL(1)" if goal else ("COLL(0)" if coll else ("STUK(0)" if info == 8 else "TIME(0)"))
            print(f"Ep {self._ep:5d} | Ph{self._cur_phase} | goal={self._cur_goal} | "
                f"{terminal_tag} | "
                  f"R={self._ep_reward:+8.1f} | steps={self._ep_steps:4d} | "
                  f"SR={sr:5.1f}% | H_tgt={self._target_h:.2f} | "
                f"buf={len(self.replay):,} | upd={self._updates}{wup}",
                  flush=True)

            with open(self._log_path, "a", newline="") as f:
                _gx = self._cur_goal_x; _gy = self._cur_goal_y
                _gd = round((_gx**2 + _gy**2)**0.5, 2) if _gx == _gx else float("nan")
                _gc = _goal_label(_gx, _gy) if _gx == _gx else "?"
                csv.writer(f).writerow([
                    self._ep, round(self._ep_reward, 2), self._ep_steps,
                    coll, goal, terminal_tag, round(sr, 2), round(ms, 1),
                    self._updates, len(self.replay),
                    round(self._last_alpha, 4), self._cur_phase, round(elapsed, 1),
                    _gx, _gy, _gd, _gc])

            # Inform Goal Manager about the outcome for curriculum advancement
            res_msg = Float32MultiArray()
            res_msg.data = [float(self._ep), float(self._ep_reward), float(self._ep_steps),
                            float(coll), float(goal), float(info)]
            self._res_pub.publish(res_msg)

            if not EVAL_ONLY:
                if self._ep % SAVE_EVERY == 0: self._save_ckpt()
                if sr > self._best_sr and len(self._outcomes) >= 20:
                    self._best_sr = sr; self._save_ckpt(tag="best_sr")
            if MAX_EPISODES > 0 and self._ep >= MAX_EPISODES:
                print(f"[{self.NODE_NAME}] MAX_EPISODES reached.", flush=True)
                if not EVAL_ONLY: self._save_ckpt()
                rclpy.shutdown(); return

            self._waiting_reset = True
        else:
            self._send_action(nobs)

    def _send_action(self, obs):
        if EVAL_ONLY or self._total_steps >= WARMUP:
            tanh_act = self.agent.select_action(obs, deterministic=EVAL_ONLY)
        else:
            tanh_act = np.random.uniform(-1, 1, ACT_DIM).astype(np.float32)
        self._current_act = tanh_act
        lin = float(LIN_MIN + (tanh_act[0] + 1.0) / 2.0 * (LIN_MAX - LIN_MIN))
        ang = float(tanh_act[1] * ANG_MAX)
        msg = Float32MultiArray(); msg.data = [lin, ang]
        self._act_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrainNode()
    try:    rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()


if __name__ == "__main__":
    main()
