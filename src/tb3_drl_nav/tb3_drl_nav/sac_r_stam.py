#!/usr/bin/env python3
"""
sac_v11.py  Ablation Study: SAC + GRU Recurrence
=====================================================================
Ablation role : Replaces frame-stack temporal memory with GRU recurrence.
Architecture  : STAM v2 (2-frame: scan + scan_vel) → GRU Actor (256)
                + STAM v2 → GRU Twin Critic (384)
Observation   : 54-dim raw (24 scan + 24 scan_vel + 6 nav) — same as baseline
Replay        : Uniform trajectory replay (200 K)  — sequences, not transitions
Returns       : 1-step TD with Burn-In warm-up for GRU hidden state
Critic loss   : MSE (plain — Huber depletes alpha, proven by v10 ablation)
Entropy target: Annealed  −1.0 → −2.0 over 500 episodes (matches v8)

Key difference from v8/v10:
  - Frame stack (3 explicit frames) → GRU (learned hidden state, full episode)
  - Sequences sampled from replay (length=16+8 burn-in) instead of i.i.d. transitions
  - No PER — uniform sequence sampling (PER + sequences = unstable)
  - Batch size 64 (smaller due to sequence dimension)

The GRU hypothesis: an agent that can remember where an obstacle was
  30 steps ago should plan around it better than one seeing 3 stacked frames.

Compare against:
  sac_baseline.py — no memory at all
  sac_v8.py       — 3-frame stack + PER + 3-step
  sac_v10.py      — same as v8 + wider critic + Huber

Run:
  SEED=42 ros2 run tb3_drl_nav sac_v11 --ros-args \\
      -p run_id:=sac_v11_s42 -p fresh:=true
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
OBS_DIM      = _C.get("obs_dim", 54)       # 24 scan + 24 scan_vel + 6 nav (raw)
ACT_DIM      = _C.get("act_dim", 2)
LIN_MIN      = _C.get("lin_min", 0.0)
LIN_MAX      = _C.get("lin_max", 0.26)
ANG_MAX      = _C.get("ang_max", 1.82)
N_SECTORS    = 24

# ── [ABLATION] Recurrent sequence parameters ──────────────────────────────────
SEQ_LEN      = int(_C.get("sac_v11_seq_len",  16))
BURN_IN      = int(_C.get("sac_v11_burn_in",   8))
TOTAL_SEQ    = SEQ_LEN + BURN_IN

# ── [ABLATION] Network widths — keep critic wider (from v10 finding) ──────────
ACTOR_HIDDEN  = int(_C.get("sac_v11_actor_hidden",  256))
CRITIC_HIDDEN = int(_C.get("sac_v11_critic_hidden", 384))
STAM_D_MODEL  = int(_C.get("sac_v11_stam_d_model",   16))
STAM_HEADS    = int(_C.get("sac_v11_stam_heads",      2))

BUFFER_CAP   = _C.get("sac_buffer",    200_000)
BATCH        = int(_C.get("sac_v11_batch", 64))
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

# ── [ABLATION] Entropy annealing (matches v8/v10 strategy) ────────────────────
# FIX 1 (wrong): ENT_START=-0.5 was too lenient — GRU satisfied it, alpha→0.04 by ep 250
# FIX 2 (wrong): ENT_START=-2.0 (=ENT_END) is too strict from ep 1 — GRU over-explores → alpha→0.07 by ep 300
# FIX 3 (correct): Match v8 exactly — start lenient at -1.0, anneal to -2.0 over 500 eps
# This gives the GRU time to warm up its hidden state before demanding tight entropy
ENT_START    = float(_C.get("sac_v11_entropy_start",       1.0))
ENT_END      = float(_C.get("sac_v11_entropy_end",         ACT_DIM))
ENT_ANNEAL   = int(_C.get("sac_v11_entropy_anneal_eps",     500))
# FIX 4: Alpha floor for recurrent policy — GRU initial hidden noise causes
# high-entropy outputs → alpha crashes to 0.04 → deterministic → can't explore
# distant Phase 3+ goals (43% timeout). Standard fix in recurrent SAC literature.
ALPHA_MIN    = float(_C.get("sac_v11_alpha_min",          0.1))

EVAL_ONLY    = os.environ.get("EVAL_ONLY", "").lower() in ("true", "1", "yes")
RESET_SETTLE_S = float(_C.get("trainer_runtime", {}).get("terminal_filters", {}).get("reset_settle_s", 0.10))
MIN_STUCK_STEPS = int(_C.get("trainer_runtime", {}).get("terminal_filters", {}).get("min_stuck_steps", 60))
MIN_TIMEOUT_STEPS = int(_C.get("trainer_runtime", {}).get("terminal_filters", {}).get("min_timeout_steps", 500))

GOAL_CATALOG = [
    # Core (1.2-2.1m)
    (0.0, 1.2), (0.0, -1.2), (0.0, 1.9), (0.0, -1.9), (1.4, 1.4), (-1.4, 1.4), (1.4, -1.4),
    # Expand (Corridor entrances 2.5-3.2m)
    (0.0, 2.5), (0.0, -2.5), (2.5, 0.0), (-2.5, 0.0), (3.2, 1.5), (-3.2, 1.5), (-3.2, -1.5),
    # Mid (Room entrances 3.8m dia)
    (-2.8, 2.8), (2.8, 2.8), (-2.5, -2.5), (2.8, -2.8),
    # Inter (Bridge goals)
    (-3.3, 2.2), (3.3, 2.2), (-3.3, -2.2), (3.3, -2.2),
    (-0.5, 3.0), (0.5, 3.0), (0.0, -3.1), (-0.5, -3.0), (0.5, -3.0),
    # Deep (Extreme rooms & corridors)
    (-4.4, 4.4), (4.4, 4.4), (-4.4, -4.4), (4.4, -4.4),
    (0.0, 4.4), (0.0, -4.4), (4.4, 0.0), (-4.4, 0.0),
    # Benchmark goals
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
#  TRAJECTORY REPLAY BUFFER  — [ABLATION] stores transitions, samples sequences
# ══════════════════════════════════════════════════════════════════════════════
class TrajectoryReplayBuffer:
    """
    Flat circular buffer that stores individual transitions.
    Sampling draws contiguous sequences of length TOTAL_SEQ,
    skipping any sequence that crosses an episode boundary (done=True inside).
    This gives the GRU coherent episode context without requiring full episodes.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.obs  = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.act  = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.rew  = np.zeros((capacity, 1),       dtype=np.float32)
        self.nobs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.done = np.zeros((capacity, 1),       dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def add(self, obs, act, rew, nobs, done):
        self.obs[self.ptr]  = obs
        self.act[self.ptr]  = act
        self.rew[self.ptr]  = rew
        self.nobs[self.ptr] = nobs
        self.done[self.ptr] = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, seq_length, device):
        batch = []
        _max_attempts = batch_size * 50  # prevent infinite loop with short episodes
        _attempts = 0
        while len(batch) < batch_size:
            _attempts += 1
            if _attempts > _max_attempts:
                # Fallback: duplicate existing samples to fill batch
                if not batch:
                    # Absolute fallback: pick any contiguous block
                    batch.append(0)
                while len(batch) < batch_size:
                    batch.append(random.choice(batch))
                break
            idx = random.randint(0, self.size - seq_length - 1)
            # Reject if sequence wraps around write pointer (stale data)
            if idx <= self.ptr < idx + seq_length:
                continue
            # Reject if any step (except the last) is terminal
            if np.any(self.done[idx : idx + seq_length - 1]):
                continue
            batch.append(idx)
        t = lambda arr: torch.tensor(
            np.stack([arr[i : i + seq_length] for i in batch]),
            dtype=torch.float32, device=device)
        return t(self.obs), t(self.act), t(self.rew), t(self.nobs), t(self.done)


# ══════════════════════════════════════════════════════════════════════════════
#  STAM v2  — 2-frame version (scan_t + scan_vel instead of 3 stacked frames)
# ══════════════════════════════════════════════════════════════════════════════
class MultiHeadScanAttention(nn.Module):
    """
    Multi-head self-attention over 24 LiDAR sectors.
    Input  : (B, 24, 2) — [scan_distance, scan_velocity] per sector
    Output : (B, 48) — compressed attention-weighted representation

    In v8/v10 this received 3 historical frames.
    In v11 it receives only current scan + velocity (the GRU tracks history).
    """
    def __init__(self, n_sectors=N_SECTORS, n_frames=2,
                 d_model=STAM_D_MODEL, n_heads=STAM_HEADS, d_out=48):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_sectors = n_sectors; self.n_heads = n_heads
        self.d_k = d_model // n_heads; self.d_model = d_model

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
        B = x.size(0)
        h = self.proj_in(x) + self.pos_enc
        qkv = self.W_qkv(h).reshape(B, self.n_sectors, 3, self.n_heads, self.d_k
                          ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k), dim=-1)
        out  = torch.matmul(attn, v).transpose(1,2).reshape(B, self.n_sectors, self.d_model)
        return self.compress(self.proj_out(out).reshape(B, -1))


def _extract_stam_input(obs_flat):
    """Split flat obs → (B,24,2) for STAM and (B,6) nav."""
    scan     = obs_flat[:, :N_SECTORS]
    scan_vel = obs_flat[:, N_SECTORS : 2 * N_SECTORS]
    nav      = obs_flat[:, 2 * N_SECTORS:]
    return torch.stack([scan, scan_vel], dim=-1), nav


# ── [ABLATION] Recurrent Actor: STAM → MLP → GRU ─────────────────────────────
class RecurrentActor(nn.Module):
    """
    Per-timestep: STAM(scan, scan_vel) → LayerNorm(cat[stam_out, nav]) → FC
    Across time:  GRU carries hidden state
    Output:       mu, std, h_new  (h_new passed back each step at inference)
    """
    def __init__(self):
        super().__init__()
        self.stam    = MultiHeadScanAttention()           # (B,24,2) → (B,48)
        self.ln      = nn.LayerNorm(48 + 6)
        self.fc      = nn.Sequential(
            nn.Linear(48 + 6, ACTOR_HIDDEN), nn.ReLU())
        self.gru     = nn.GRU(ACTOR_HIDDEN, ACTOR_HIDDEN, batch_first=True)
        self.mu      = nn.Linear(ACTOR_HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(ACTOR_HIDDEN, ACT_DIM)

    def forward(self, obs, h=None):
        """obs: (B, T, OBS_DIM)  →  mu/std: (B, T, ACT_DIM),  h_new: (1,B,H)"""
        B, T, _ = obs.shape
        flat     = obs.reshape(B * T, -1)
        stam_in, nav = _extract_stam_input(flat)
        x  = self.fc(self.ln(torch.cat([self.stam(stam_in), nav], dim=-1)))
        x  = x.reshape(B, T, ACTOR_HIDDEN)
        gru_out, h_new = self.gru(x, h)
        return self.mu(gru_out), self.log_std(gru_out).clamp(-5.0, 2.0).exp(), h_new

    def sample(self, obs, h=None):
        mu, std, h_new = self(obs, h)
        raw  = Normal(mu, std).rsample()
        act  = torch.tanh(raw)
        lp   = (Normal(mu, std).log_prob(raw) -
                torch.log(1.0 - act.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return act, lp, h_new


# ── [ABLATION] Recurrent Twin Critic: STAM → MLP → GRU ───────────────────────
class RecurrentCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.stam1 = MultiHeadScanAttention()
        self.stam2 = MultiHeadScanAttention()
        d = 48 + 6 + ACT_DIM
        self.q1_fc  = nn.Sequential(nn.Linear(d, CRITIC_HIDDEN), nn.ReLU())
        self.q1_gru = nn.GRU(CRITIC_HIDDEN, CRITIC_HIDDEN, batch_first=True)
        self.q1_out = nn.Linear(CRITIC_HIDDEN, 1)
        self.q2_fc  = nn.Sequential(nn.Linear(d, CRITIC_HIDDEN), nn.ReLU())
        self.q2_gru = nn.GRU(CRITIC_HIDDEN, CRITIC_HIDDEN, batch_first=True)
        self.q2_out = nn.Linear(CRITIC_HIDDEN, 1)

    def forward(self, obs, act, h1=None, h2=None):
        B, T, _ = obs.shape
        flat     = obs.reshape(B * T, -1)
        act_flat = act.reshape(B * T, -1)
        stam_in, nav = _extract_stam_input(flat)

        x1 = torch.cat([self.stam1(stam_in), nav, act_flat], dim=-1)
        x2 = torch.cat([self.stam2(stam_in), nav, act_flat], dim=-1)

        q1, h1_new = self.q1_gru(self.q1_fc(x1).reshape(B, T, CRITIC_HIDDEN), h1)
        q2, h2_new = self.q2_gru(self.q2_fc(x2).reshape(B, T, CRITIC_HIDDEN), h2)
        return self.q1_out(q1), self.q2_out(q2), h1_new, h2_new


# ══════════════════════════════════════════════════════════════════════════════
#  SAC AGENT  — Recurrent SAC with Burn-In
# ══════════════════════════════════════════════════════════════════════════════
class SAC:
    """
    Recurrent SAC with Burn-In (R2D2-style).
    Burn-in: first BURN_IN steps warm up GRU hidden states without gradients.
    Active:  last SEQ_LEN steps compute actor/critic/alpha losses.
    """
    def __init__(self, device):
        self.device     = device
        self.actor      = RecurrentActor().to(device)
        self.critic     = RecurrentCritic().to(device)
        self.critic_tgt = RecurrentCritic().to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters(): p.requires_grad = False
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha  = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

    @property
    def alpha(self): return self.log_alpha.exp().item()

    def select_action(self, obs_seq, h=None, deterministic=False):
        """obs_seq: (1,1,OBS_DIM) single step at inference."""
        with torch.no_grad():
            mu, std, h_new = self.actor(obs_seq, h)
            act = torch.tanh(mu) if deterministic else torch.tanh(Normal(mu, std).rsample())
        return act[0, -1, :].cpu().numpy(), h_new

    def update(self, replay, target_h):
        obs, act, rew, nobs, done = replay.sample(BATCH, TOTAL_SEQ, self.device)
        alpha = self.log_alpha.exp().detach()

        # ── Burn-in: warm up GRU hidden states (no gradient through burn-in) ──
        with torch.no_grad():
            obs_bi  = obs[:, :BURN_IN, :]
            nobs_bi = nobs[:, :BURN_IN, :]
            act_bi  = act[:, :BURN_IN, :]
            _, _, h_actor = self.actor(obs_bi)
            _, _, h_c1, h_c2 = self.critic(obs_bi, act_bi)
            na_bi, _, h_tgt_actor = self.actor.sample(nobs_bi)
            _, _, h_tc1, h_tc2 = self.critic_tgt(nobs_bi, na_bi)

        h_a  = h_actor.detach();    h_c1 = h_c1.detach();  h_c2  = h_c2.detach()
        h_ta = h_tgt_actor.detach(); h_tc1 = h_tc1.detach(); h_tc2 = h_tc2.detach()

        # ── Active window ─────────────────────────────────────────────────────
        obs_a  = obs[:, BURN_IN:, :];   nobs_a = nobs[:, BURN_IN:, :]
        act_a  = act[:, BURN_IN:, :];   rew_a  = rew[:, BURN_IN:, :]
        done_a = done[:, BURN_IN:, :]

        # ── Critic ── [ABLATION] MSE loss, 1-step TD ─────────────────────────
        # NOTE: Huber loss was here originally (from v10), but v10 ablation proved
        # Huber depletes alpha ~3x faster than MSE → Phase 6 plateau. Switched to MSE.
        with torch.no_grad():
            na, nlp, _ = self.actor.sample(nobs_a, h=h_ta)
            tq1, tq2, _, _ = self.critic_tgt(nobs_a, na, h_tc1, h_tc2)
            target = rew_a + GAMMA * (1 - done_a) * (torch.min(tq1, tq2) - alpha * nlp)
        q1, q2, _, _ = self.critic(obs_a, act_a, h_c1, h_c2)
        loss_c = (q1 - target).pow(2).mean() + (q2 - target).pow(2).mean()
        self.opt_critic.zero_grad(); loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        # ── Actor ─────────────────────────────────────────────────────────────
        for p in self.critic.parameters(): p.requires_grad = False
        new_act, new_lp, _ = self.actor.sample(obs_a, h=h_a)
        q1_pi, q2_pi, _, _ = self.critic(obs_a, new_act, h_c1, h_c2)
        loss_a = (alpha * new_lp - torch.min(q1_pi, q2_pi)).mean()
        self.opt_actor.zero_grad(); loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
        for p in self.critic.parameters(): p.requires_grad = True

        # ── Temperature ── [FIXED] alpha increase when entropy < target ─────────
        loss_t = (self.log_alpha * (-new_lp.detach() - target_h)).mean()
        self.opt_alpha.zero_grad(); loss_t.backward(); self.opt_alpha.step()
        # [FIX 4] Alpha floor — prevent GRU entropy noise from crashing alpha
        with torch.no_grad():
            self.log_alpha.clamp_(min=math.log(ALPHA_MIN))

        # ── Soft update ───────────────────────────────────────────────────────
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - TAU).add_(TAU * p.data)

        return loss_a.item(), loss_c.item(), self.alpha

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
    NODE_NAME   = "sac_v11"
    DEFAULT_RUN = "sac_v11_s42"

    def __init__(self):
        super().__init__(self.NODE_NAME)
        self.declare_parameter("run_id", self.DEFAULT_RUN)
        self.declare_parameter("fresh",  False)
        run_id      = self.get_parameter("run_id").value
        self._fresh = self.get_parameter("fresh").value

        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent  = SAC(self.device)
        self.replay = TrajectoryReplayBuffer(BUFFER_CAP)

        # GRU hidden state — carried across steps within an episode
        self._actor_h = None
        # Obs history window (for sequence sampling at inference: length 1)
        self._ep_obs_history = collections.deque(maxlen=TOTAL_SEQ)

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
        self._current_act   = np.zeros(ACT_DIM, dtype=np.float32)
        self._cur_phase     = 1
        self._cur_goal      = "?"
        self._cur_goal_x    = float("nan")
        self._cur_goal_y    = float("nan")
        self._last_alpha    = 1.0
        self._target_h      = ENT_START
        self._last_reset_time = time.time()

        if self._fresh: print(f"[{self.NODE_NAME}] FRESH START.", flush=True)
        else:           self._resume()

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
            Int32,   "/tb3_drl/curriculum_phase", self._on_phase, _LATCHED)
        self.create_subscription(
            String,  "/tb3_drl/goal",             self._on_goal,  _LATCHED)

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        mode = "EVAL" if EVAL_ONLY else f"TRAIN  warmup={WARMUP:,}"
        print(f"\n[{self.NODE_NAME}] {mode}  device={self.device}  "
              f"ep={self._ep}  seed={SEED}  seq={SEQ_LEN}+{BURN_IN}burn-in", flush=True)
        print("=" * 80, flush=True)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def _save_ckpt(self, tag=None):
        label = tag or f"ep{self._ep:06d}"
        path  = os.path.join(self._model_dir, f"ckpt_{label}.pt")
        buf_path = os.path.join(self._model_dir, f"ckpt_{label}_buf.npz")
        elapsed = self._elapsed_offset + (time.time() - self._t0)
        torch.save({
            'agent': self.agent.state_dict(),
            'episode': self._ep,
            'total_steps': self._total_steps,
            'updates': self._updates,
            'best_sr': self._best_sr,
            'elapsed_s': elapsed,
            'outcomes': list(self._outcomes),
            'steps_hist': list(self._steps_hist),
            'last_alpha': self._last_alpha,
            'cur_phase': self._cur_phase,
        }, path)
        # Save replay buffer alongside checkpoint
        buf = self.replay
        np.savez_compressed(buf_path,
            obs=buf.obs[:buf.size],
            act=buf.act[:buf.size],
            rew=buf.rew[:buf.size],
            nobs=buf.nobs[:buf.size],
            done=buf.done[:buf.size],
            ptr=np.array([buf.ptr]),
            size=np.array([buf.size]),
        )
        torch.save(self.agent.actor.state_dict(),
                   os.path.join(self._model_dir, "actor_latest.pt"))
        if tag is None:
            for old in sorted(glob.glob(
                    os.path.join(self._model_dir, "ckpt_ep*.pt")))[:-KEEP_CKPTS]:
                try:
                    os.remove(old)
                    buf_old = old.replace(".pt", "_buf.npz")
                    if os.path.exists(buf_old):
                        os.remove(buf_old)
                except OSError:
                    pass

    def _resume(self):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_*.pt")))
        actor_latest = os.path.join(self._model_dir, "actor_latest.pt")
        if not ckpts:
            if EVAL_ONLY and os.path.exists(actor_latest):
                try:
                    actor_state = torch.load(actor_latest, map_location="cpu", weights_only=False)
                    self.agent.actor.load_state_dict(actor_state, strict=False)
                    print(f"[{self.NODE_NAME}] Loaded actor_latest.pt for eval-only", flush=True)
                except Exception as e:
                    print(f"[{self.NODE_NAME}] Eval weight load failed ({e}).", flush=True)
            return
        try:
            ckpt_latest = max(ckpts, key=os.path.getmtime)
            ckpt = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
            self.agent.load_state_dict(ckpt["agent"], self.device)
            self._ep             = ckpt.get("episode",     0)
            self._total_steps    = ckpt.get("total_steps", 0)
            self._updates        = ckpt.get("updates",     0)
            self._best_sr        = ckpt.get("best_sr",     0.0)
            self._elapsed_offset = ckpt.get("elapsed_s",   0.0)
            # --- STABILIZATION BLOCK ---
            stabilize = os.environ.get("STABILIZE", "false").lower() == "true"
            boost = os.environ.get("ENT_BOOST", "false").lower() == "true"

            if stabilize:
                print(f"[{self.NODE_NAME}] !!! STABILIZATION ACTIVE: Forcing alpha=0.1 and clearing SR history !!!", flush=True)
                self._last_alpha = 0.1
                self._outcomes.clear()
                self._steps_hist.clear()
                sr_info = " | SR-100:RESET"
            else:
                self._last_alpha     = ckpt.get("last_alpha",  1.0)
                if "outcomes" in ckpt and ckpt["outcomes"]:
                    self._outcomes.clear()
                    self._outcomes.extend(ckpt["outcomes"])
                if "steps_hist" in ckpt and ckpt["steps_hist"]:
                    self._steps_hist.clear()
                    self._steps_hist.extend(ckpt["steps_hist"])
                sr_info = ""
                if self._outcomes:
                    sr_info = f" | SR-100:{sum(self._outcomes)/len(self._outcomes)*100:.1f}%"

            if boost:
                print(f"[{self.NODE_NAME}] !!! ENTROPY BOOST ACTIVE: Target forced to -1.5 !!!", flush=True)

            self._cur_phase      = ckpt.get("cur_phase",   1)
            self._target_h       = self._entropy_target()

            fname = os.path.basename(ckpt_latest)
            print(f"[{self.NODE_NAME}] Resumed from {fname}: ep={self._ep} | "
                  f"Ph{self._cur_phase}{sr_info}", flush=True)
            # Re-seed on resume for consistency
            torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
            # Restore replay buffer if available
            buf_path = ckpt_latest.replace(".pt", "_buf.npz")
            if os.path.exists(buf_path):
                try:
                    d = np.load(buf_path)
                    n = int(d["size"][0])
                    buf = self.replay
                    buf.obs[:n]  = d["obs"]
                    buf.act[:n]  = d["act"]
                    buf.rew[:n]  = d["rew"]
                    buf.nobs[:n] = d["nobs"]
                    buf.done[:n] = d["done"]
                    buf.ptr      = int(d["ptr"][0])
                    buf.size     = n
                    print(f"[{self.NODE_NAME}] Replay buffer restored: {n:,} transitions",
                          flush=True)
                except Exception as be:
                    print(f"[{self.NODE_NAME}] Buffer restore failed ({be}) — starting fresh",
                          flush=True)
        except Exception as e:
            print(f"[{self.NODE_NAME}] Resume failed ({e}).", flush=True)

    def _shutdown(self, *_):
        print(f"\n[{self.NODE_NAME}] Shutdown — saving…", flush=True)
        self._save_ckpt(tag="shutdown")
        try: rclpy.shutdown()
        except Exception: pass

    # ── ROS callbacks ─────────────────────────────────────────────────────────
    def _on_phase(self, msg: Int32): self._cur_phase = msg.data

    def _entropy_target(self):
        """[ANNEALING] Force exploitation in Phase 7, else normal annealing or boost."""
        if os.environ.get("ENT_BOOST", "false").lower() == "true":
            return -1.5  # Temporary exploration boost to break plateaus
        if self._cur_phase == 7:
            return -2.0  # Force high exploitation for expert timing
        if self._ep >= ENT_ANNEAL: return ENT_END
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
        obs = np.array(msg.data[:OBS_DIM], dtype=np.float32)
        self._ep_obs_history.clear()
        self._ep_obs_history.append(obs)
        self._actor_h       = None   # reset GRU state at episode boundary
        self._ep_reward     = 0.0
        self._ep_steps      = 0
        self._waiting_reset = False
        self._last_reset_time = time.time()
        self._send_action(obs)

    def _on_step(self, msg: Float32MultiArray):
        if self._waiting_reset: return
        nobs   = np.array(msg.data[:OBS_DIM],   dtype=np.float32)
        reward = float(msg.data[OBS_DIM])
        done   = bool(msg.data[OBS_DIM + 1])
        info   = int(msg.data[OBS_DIM + 2])

        if done and self._is_impossible_terminal(info):
            return

        self._ep_reward   += reward
        self._ep_steps    += 1
        self._total_steps += 1

        prev_obs = self._ep_obs_history[-1]
        self.replay.add(prev_obs, self._current_act, reward, nobs, float(done))
        self._ep_obs_history.append(nobs)

        if (not EVAL_ONLY
                and self._total_steps > WARMUP
                and self.replay.size > TOTAL_SEQ * 2):
            self._target_h = self._entropy_target()
            _, _, alpha_val = self.agent.update(self.replay, self._target_h)
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

            terminal_tag = "GOAL(1)" if goal else ("COLL(0)" if coll else ("STUK(0)" if info == 8 else "TIME(0)"))
            wup = (f"  [WARMUP {self._total_steps}/{WARMUP}]"
                   if self._total_steps < WARMUP else "")
            print(f"Ep {self._ep:5d} | Ph{self._cur_phase} | goal={self._cur_goal} | "
                f"{terminal_tag} | "
                  f"R={self._ep_reward:+8.1f} | steps={self._ep_steps:4d} | "
                  f"SR={sr:5.1f}% | H_tgt={self._target_h:.2f} | "
                  f"buf={self.replay.size:,} | upd={self._updates}{wup}",
                  flush=True)

            with open(self._log_path, "a", newline="") as f:
                _gx = self._cur_goal_x; _gy = self._cur_goal_y
                _gd = round((_gx**2 + _gy**2)**0.5, 2) if _gx == _gx else float("nan")
                _gc = _goal_label(_gx, _gy) if _gx == _gx else "?"
                csv.writer(f).writerow([
                    self._ep, round(self._ep_reward, 2), self._ep_steps,
                    coll, goal, terminal_tag, round(sr, 2), round(ms, 1),
                    self._updates, self.replay.size,
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
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                 device=self.device).unsqueeze(0).unsqueeze(0)
            tanh_act, self._actor_h = self.agent.select_action(
                obs_t, self._actor_h, deterministic=EVAL_ONLY)
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
