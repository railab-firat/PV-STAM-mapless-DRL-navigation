#!/usr/bin/env python3
"""
train_agent_sac_v11.py  Recurrent SAC (GRU + Burn-In)
==========================================================================
FIXED VERSION + ENHANCED LOGGING (Buffer Size, Entropy Target, Smoothing)
  + Warmup random exploration (10 K steps, from config)
  + Checkpoint save every 50 eps + best-SR + on shutdown
  + Resume from last checkpoint on restart
  + MAX_EPISODES auto-stop
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
from std_msgs.msg import Float32MultiArray, Int32

# ── config (same source-of-truth as v8/v10) ──────────────────────────────────
_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "phase3_ppo.yaml")
try:
    import yaml
    with open(_CFG_PATH) as _f:
        _C = yaml.safe_load(_f) or {}
except Exception:
    _C = {}

RESET_OBS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

# ── Configuration & Hyperparameters ──────────────────────────────────────────
SEED          = int(os.environ.get("SEED", _C.get("seed", 42)))
RAW_OBS       = _C.get("obs_dim", 54)
ACT_DIM       = _C.get("act_dim", 2)
LIN_MIN       = _C.get("lin_min", 0.0)
LIN_MAX       = _C.get("lin_max", 0.26)
ANG_MAX       = _C.get("ang_max", 1.82)

ACTOR_HIDDEN  = int(_C.get("sac_v11_actor_hidden", _C.get("sac_hidden", 256)))
CRITIC_HIDDEN = int(_C.get("sac_v11_critic_hidden", 384))
STAM_D_MODEL  = int(_C.get("sac_v11_stam_d_model", 16))
STAM_HEADS    = int(_C.get("sac_v11_stam_heads", 2))

SEQ_LEN       = int(_C.get("sac_v11_seq_len", 16))
BURN_IN       = int(_C.get("sac_v11_burn_in", 8))
TOTAL_SEQ     = SEQ_LEN + BURN_IN

BUFFER_CAP    = int(_C.get("sac_buffer", 200_000))
BATCH         = int(_C.get("sac_v11_batch", _C.get("sac_batch", 64)))
GAMMA         = _C.get("gamma", 0.99)
TAU           = _C.get("sac_tau", 0.005)
LR_ACTOR      = _C.get("sac_lr_actor", 3e-4)
LR_CRITIC     = _C.get("sac_lr_critic", 3e-4)
LR_ALPHA      = _C.get("sac_lr_alpha", 3e-4)
HUBER_DELTA   = float(_C.get("sac_v11_huber_delta", 10.0))
GRAD_CLIP     = float(_C.get("sac_grad_clip", 1.0))
ACTION_SMOOTH = float(_C.get("sac_v11_action_smooth", 0.0))

ENT_START     = float(_C.get("sac_v11_entropy_start", -1.0))
ENT_END       = float(_C.get("sac_v11_entropy_end", -ACT_DIM))
ENT_ANNEAL    = int(_C.get("sac_v11_entropy_anneal_eps", 500))

WARMUP        = int(os.environ.get("WARMUP", _C.get("sac_warmup", 10_000)))  # override via env for fine-tuning
SAVE_EVERY    = _C.get("save_every",       50)
KEEP_CKPTS    = _C.get("keep_ckpts",        8)
MAX_EPISODES  = int(os.environ.get("MAX_EPISODES", _C.get("max_episodes", 0)))
MAX_STEPS     = int(os.environ.get("MAX_STEPS", 0))   # stop after N total steps (0 = disabled)
LOAD_WEIGHTS_PATH = os.environ.get("LOAD_WEIGHTS_PATH", "")  # pre-load weights for fine-tuning

N_SECTORS     = 24
OBS_DIM       = RAW_OBS
STAM_N_FRAMES = 2

# ══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════
class TrajectoryReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.obs  = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.act  = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.rew  = np.zeros((capacity,  1),      dtype=np.float32)
        self.nobs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.done = np.zeros((capacity,  1),      dtype=np.float32)
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
        batch_obs, batch_act, batch_rew, batch_nobs, batch_done = [], [], [], [], []
        valid = 0
        while valid < batch_size:
            idx = random.randint(0, self.size - seq_length - 1)
            if idx <= self.ptr < idx + seq_length: continue
            if np.any(self.done[idx : idx + seq_length - 1]): continue
            batch_obs.append(self.obs[idx  : idx + seq_length])
            batch_act.append(self.act[idx  : idx + seq_length])
            batch_rew.append(self.rew[idx  : idx + seq_length])
            batch_nobs.append(self.nobs[idx : idx + seq_length])
            batch_done.append(self.done[idx : idx + seq_length])
            valid += 1
        t = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)
        return t(batch_obs), t(batch_act), t(batch_rew), t(batch_nobs), t(batch_done)

# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORKS
# ══════════════════════════════════════════════════════════════════════════════
class MultiHeadScanAttention(nn.Module):
    def __init__(self, n_sectors=N_SECTORS, n_frames=STAM_N_FRAMES,
                 d_model=STAM_D_MODEL, n_heads=STAM_HEADS, d_out=48):
        super().__init__()
        self.n_sectors = n_sectors
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.proj_in   = nn.Linear(n_frames, d_model)
        self.pos_enc   = nn.Parameter(torch.randn(1, n_sectors, d_model) * 0.02)
        self.W_qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out  = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.compress  = nn.Linear(n_sectors * d_model, d_out)

    def forward(self, x):
        B   = x.size(0)
        h   = self.proj_in(x) + self.pos_enc
        qkv = self.W_qkv(h).reshape(
            B, self.n_sectors, 3, self.n_heads, self.d_model // self.n_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1)), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, self.n_sectors, self.d_model)
        return self.compress(self.proj_out(out).reshape(B, -1))

def _extract_stam_input(obs_flat):
    scan     = obs_flat[:, :N_SECTORS]
    scan_vel = obs_flat[:, N_SECTORS:2 * N_SECTORS]
    return torch.stack([scan, scan_vel], dim=-1)

def _extract_nav(obs_flat):
    return obs_flat[:, 2 * N_SECTORS:]

class RecurrentActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stam    = MultiHeadScanAttention(n_frames=STAM_N_FRAMES, d_out=48)
        self.ln      = nn.LayerNorm(54)
        self.fc      = nn.Sequential(nn.Linear(54, ACTOR_HIDDEN), nn.ReLU())
        self.gru     = nn.GRU(ACTOR_HIDDEN, ACTOR_HIDDEN, batch_first=True)
        self.mu      = nn.Linear(ACTOR_HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(ACTOR_HIDDEN, ACT_DIM)

    def forward(self, obs, h=None):
        B, T, _ = obs.shape
        obs_flat     = obs.reshape(B * T, -1)
        stam_in      = _extract_stam_input(obs_flat)
        nav          = _extract_nav(obs_flat)
        stam_out     = self.stam(stam_in)
        x            = torch.cat([stam_out, nav], dim=-1)
        x            = self.fc(self.ln(x)).reshape(B, T, ACTOR_HIDDEN)
        gru_out, h_new = self.gru(x, h)
        return self.mu(gru_out), self.log_std(gru_out).clamp(-5.0, 2.0).exp(), h_new

    def sample(self, obs, h=None):
        mu, std, h_new = self(obs, h)
        dist     = Normal(mu, std)
        raw      = dist.rsample()
        act      = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1.0 - act.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return act, log_prob, h_new

class RecurrentCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.stam1   = MultiHeadScanAttention(n_frames=STAM_N_FRAMES, d_out=48)
        self.stam2   = MultiHeadScanAttention(n_frames=STAM_N_FRAMES, d_out=48)

        self.q1_fc   = nn.Sequential(nn.Linear(54 + ACT_DIM, CRITIC_HIDDEN), nn.ReLU())
        self.q1_gru  = nn.GRU(CRITIC_HIDDEN, CRITIC_HIDDEN, batch_first=True)
        self.q1_out  = nn.Linear(CRITIC_HIDDEN, 1)

        self.q2_fc   = nn.Sequential(nn.Linear(54 + ACT_DIM, CRITIC_HIDDEN), nn.ReLU())
        self.q2_gru  = nn.GRU(CRITIC_HIDDEN, CRITIC_HIDDEN, batch_first=True)
        self.q2_out  = nn.Linear(CRITIC_HIDDEN, 1)

    def forward(self, obs, act, h1=None, h2=None):
        B, T, _ = obs.shape
        obs_flat = obs.reshape(B * T, -1)
        act_flat = act.reshape(B * T, -1)

        stam_in  = _extract_stam_input(obs_flat)
        nav      = _extract_nav(obs_flat)

        s1       = self.stam1(stam_in)
        s2       = self.stam2(stam_in)

        x1       = torch.cat([s1, nav, act_flat], dim=-1)
        x2       = torch.cat([s2, nav, act_flat], dim=-1)

        q1       = self.q1_fc(x1).reshape(B, T, CRITIC_HIDDEN)
        q1_gru, h1_new = self.q1_gru(q1, h1)
        q1_val   = self.q1_out(q1_gru)

        q2       = self.q2_fc(x2).reshape(B, T, CRITIC_HIDDEN)
        q2_gru, h2_new = self.q2_gru(q2, h2)
        q2_val   = self.q2_out(q2_gru)

        return q1_val, q2_val, h1_new, h2_new

# ══════════════════════════════════════════════════════════════════════════════
#  AGENT LOGIC
# ══════════════════════════════════════════════════════════════════════════════
class RecurrentSAC:
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

    def select_action(self, obs_seq, h=None, eval_mode=False):
        with torch.no_grad():
            mu, std, h_new = self.actor(obs_seq, h)
            if eval_mode: act = torch.tanh(mu)
            else: act = torch.tanh(Normal(mu, std).rsample())
        return act[0, -1, :].cpu().numpy(), h_new

    def update(self, replay, target_entropy):
        obs, act, rew, nobs, done = replay.sample(BATCH, TOTAL_SEQ, self.device)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            obs_burn, nobs_burn, act_burn = obs[:,:BURN_IN,:], nobs[:,:BURN_IN,:], act[:,:BURN_IN,:]
            _, _, h_actor_warmed = self.actor(obs_burn, h=None)
            _, _, h_c1_warmed, h_c2_warmed = self.critic(obs_burn, act_burn, None, None)
            na_burn, _, h_tgt_actor_w = self.actor.sample(nobs_burn, h=None)
            _, _, h_tc1_warmed, h_tc2_warmed = self.critic_tgt(nobs_burn, na_burn, None, None)

        h_actor_w, h_c1_w, h_c2_w = h_actor_warmed.detach(), h_c1_warmed.detach(), h_c2_warmed.detach()
        h_tc1_w, h_tc2_w, h_tgt_a_w = h_tc1_warmed.detach(), h_tc2_warmed.detach(), h_tgt_actor_w.detach()

        obs_act, nobs_act = obs[:,BURN_IN:,:], nobs[:,BURN_IN:,:]
        act_act, rew_act, done_act = act[:,BURN_IN:,:], rew[:,BURN_IN:,:], done[:,BURN_IN:,:]

        with torch.no_grad():
            na, nlp, _ = self.actor.sample(nobs_act, h=h_tgt_a_w)
            tq1, tq2, _, _ = self.critic_tgt(nobs_act, na, h_tc1_w, h_tc2_w)
            target = rew_act + GAMMA * (1 - done_act) * (torch.min(tq1, tq2) - alpha * nlp)

        q1, q2, _, _ = self.critic(obs_act, act_act, h_c1_w, h_c2_w)
        loss_c = F.huber_loss(q1, target, delta=HUBER_DELTA) + F.huber_loss(q2, target, delta=HUBER_DELTA)
        self.opt_critic.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        for p in self.critic.parameters(): p.requires_grad = False
        new_act, new_lp, _ = self.actor.sample(obs_act, h=h_actor_w)
        q1_pi, q2_pi, _, _ = self.critic(obs_act, new_act, h_c1_w, h_c2_w)
        smooth = 0.0
        if ACTION_SMOOTH > 0.0 and new_act.size(1) > 1:
            smooth = ACTION_SMOOTH * (new_act[:, 1:] - new_act[:, :-1]).pow(2).mean()
        loss_a = (alpha * new_lp - torch.min(q1_pi, q2_pi)).mean() + smooth
        self.opt_actor.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
        for p in self.critic.parameters(): p.requires_grad = True

        loss_t = -(self.log_alpha * (new_lp.detach() + target_entropy)).mean()
        self.opt_alpha.zero_grad()
        loss_t.backward()
        nn.utils.clip_grad_norm_([self.log_alpha], GRAD_CLIP)
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - TAU).add_(TAU * p.data)

        return loss_a.item(), loss_c.item(), alpha.item(), float(smooth)

# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 TRAINING NODE
# ══════════════════════════════════════════════════════════════════════════════
class TrainAgentSACv11(Node):
    def __init__(self):
        super().__init__("train_agent_sac_v11")
        self.declare_parameter("run_id", "sac_v11_fixed")
        self.declare_parameter("fresh",  False)
        run_id      = self.get_parameter("run_id").value
        self._fresh = self.get_parameter("fresh").value

        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent  = RecurrentSAC(self.device)
        self.replay = TrajectoryReplayBuffer(BUFFER_CAP)

        self._ep_obs_history = collections.deque(maxlen=TOTAL_SEQ)
        self._actor_h        = None

        self._total_steps    = 0
        self._ep             = 0
        self._ep_reward      = 0.0
        self._ep_steps       = 0
        self._waiting_reset  = True
        self._current_act    = np.zeros(ACT_DIM, dtype=np.float32)
        self._elapsed_offset = 0.0

        self._cur_phase  = 1
        self._outcomes   = collections.deque(maxlen=100)
        self._steps_hist = collections.deque(maxlen=100)
        self._updates    = 0
        self._last_alpha = 1.0
        self._last_smooth = 0.0
        self._best_sr    = 0.0
        self._t0         = time.time()

        # ── paths ──────────────────────────────────────────────────────────────
        log_dir = os.path.expanduser(_C.get("log_dir", "~/tb3_drl_logs/phase3"))
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, f"{run_id}.csv")

        self._model_dir = os.path.expanduser(
            os.path.join(_C.get("sac_model_dir", "~/tb3_drl_models/sac"), run_id))
        os.makedirs(self._model_dir, exist_ok=True)

        # ── resume or fresh ────────────────────────────────────────────────────
        if self._fresh:
            print("[SACv11] FRESH START — ignoring any existing checkpoints.", flush=True)
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode", "reward", "steps", "collision", "goal_reached",
                    "sr_100", "mean_steps_100", "updates", "buffer_size", "alpha",
                    "target_entropy", "smooth", "elapsed_s"
                ])
        else:
            self._resume()
            # ── Fine-tuning: pre-load weights from a base checkpoint ──────────
            if LOAD_WEIGHTS_PATH and os.path.exists(LOAD_WEIGHTS_PATH) and self._ep == 0:
                try:
                    ckpt = torch.load(LOAD_WEIGHTS_PATH, map_location="cpu", weights_only=False)
                    self.agent.actor.load_state_dict(ckpt["actor"])
                    self.agent.critic.load_state_dict(ckpt["critic"])
                    self.agent.critic_tgt.load_state_dict(ckpt["critic_tgt"])
                    self._ep = 0; self._total_steps = 0; self._updates = 0
                    print(f"[SACv11] Fine-tune: loaded weights from {LOAD_WEIGHTS_PATH}", flush=True)
                except Exception as e:
                    print(f"[SACv11] LOAD_WEIGHTS_PATH failed ({e})", flush=True)
            if not os.path.exists(self._log_path):
                with open(self._log_path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "episode", "reward", "steps", "collision", "goal_reached",
                        "sr_100", "mean_steps_100", "updates", "buffer_size", "alpha",
                        "target_entropy", "smooth", "elapsed_s"
                    ])

        # ── signal handler (save on Ctrl+C / kill) ────────────────────────────
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        self._act_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self.create_subscription(Float32MultiArray, "/tb3_drl/reset_obs",   self._on_reset, RESET_OBS_QOS)
        self.create_subscription(Float32MultiArray, "/tb3_drl/step_result", self._on_step,  10)
        self.create_subscription(Int32, "/tb3_drl/curriculum_phase",        self._on_phase, RESET_OBS_QOS)

        warmup_str = f"WARMUP={WARMUP:,} steps" if WARMUP > 0 else "no warmup"
        print(f"\n[SACv11] Device={self.device}  Ep={self._ep}  Steps={self._total_steps}  {warmup_str}", flush=True)
        print("=" * 90, flush=True)

    @property
    def _target_entropy(self):
        t = min(1.0, self._ep / max(1, ENT_ANNEAL))
        return ENT_START + t * (ENT_END - ENT_START)

    # ── checkpoint helpers ─────────────────────────────────────────────────────
    def _save_ckpt(self, tag=None):
        label = tag or f"ep{self._ep:06d}"
        path  = os.path.join(self._model_dir, f"ckpt_{label}.pt")
        elapsed = self._elapsed_offset + (time.time() - self._t0)
        torch.save({
            "episode":     self._ep,
            "total_steps": self._total_steps,
            "updates":     self._updates,
            "best_sr":     self._best_sr,
            "elapsed_s":   elapsed,
            "actor":       self.agent.actor.state_dict(),
            "critic":      self.agent.critic.state_dict(),
            "critic_tgt":  self.agent.critic_tgt.state_dict(),
            "opt_actor":   self.agent.opt_actor.state_dict(),
            "opt_critic":  self.agent.opt_critic.state_dict(),
            "log_alpha":   self.agent.log_alpha.detach().cpu(),
            "opt_alpha":   self.agent.opt_alpha.state_dict(),
        }, path)
        # keep only KEEP_CKPTS rolling checkpoints (not best/shutdown)
        if tag is None:
            rolling = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
            for old in rolling[:-KEEP_CKPTS]:
                os.remove(old)
        print(f"[SACv11] Saved {path}", flush=True)

    def _resume(self):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
        if not ckpts:
            return
        try:
            ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
            self.agent.actor.load_state_dict(ckpt["actor"])
            self.agent.critic.load_state_dict(ckpt["critic"])
            self.agent.critic_tgt.load_state_dict(ckpt["critic_tgt"])
            self.agent.opt_actor.load_state_dict(ckpt["opt_actor"])
            self.agent.opt_critic.load_state_dict(ckpt["opt_critic"])
            self.agent.log_alpha = torch.tensor(
                [ckpt["log_alpha"].item()], requires_grad=True, device=self.device)
            self.agent.opt_alpha = torch.optim.Adam([self.agent.log_alpha], lr=LR_ALPHA)
            self.agent.opt_alpha.load_state_dict(ckpt["opt_alpha"])
            self._ep             = ckpt.get("episode",     0)
            self._total_steps    = ckpt.get("total_steps", 0)
            self._updates        = ckpt.get("updates",     0)
            self._best_sr        = ckpt.get("best_sr",     0.0)
            self._elapsed_offset = ckpt.get("elapsed_s",   0.0)
            print(f"[SACv11] Resumed from {ckpts[-1]}  ep={self._ep}  steps={self._total_steps}", flush=True)
        except Exception as e:
            print(f"[SACv11] Resume failed ({e}) — fresh start.", flush=True)

    def _shutdown(self, *_):
        print("\n[SACv11] Shutdown — saving checkpoint…", flush=True)
        self._save_ckpt(tag="shutdown")
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _on_phase(self, msg: Int32):
        self._cur_phase = msg.data

    def _on_reset(self, msg: Float32MultiArray):
        raw = np.array(msg.data[:RAW_OBS], dtype=np.float32)
        self._ep_obs_history.clear()
        self._actor_h   = None
        self._ep_reward = 0.0
        self._ep_steps  = 0
        self._waiting_reset = False
        self._ep_obs_history.append(raw)
        self._send_action()

    def _on_step(self, msg: Float32MultiArray):
        if self._waiting_reset: return
        raw_nobs = np.array(msg.data[:RAW_OBS],      dtype=np.float32)
        reward   = float(msg.data[RAW_OBS])
        done     = bool(msg.data[RAW_OBS + 1])
        info     = int(msg.data[RAW_OBS + 2])

        self._ep_reward    += reward
        self._ep_steps     += 1
        self._total_steps  += 1

        prev_obs = self._ep_obs_history[-1]
        self.replay.add(prev_obs, self._current_act, reward, raw_nobs, float(done))
        self._ep_obs_history.append(raw_nobs)

        if self._total_steps > WARMUP and self.replay.size > TOTAL_SEQ * 2:
            la, lc, alpha_val, smooth = self.agent.update(
                self.replay, target_entropy=float(self._target_entropy))
            self._last_alpha = alpha_val
            self._last_smooth = smooth
            self._updates += 1

        if done:
            self._ep += 1
            goal = 1 if info == 1 else 0
            coll = 1 if info == 2 else 0
            self._outcomes.append(goal)
            self._steps_hist.append(self._ep_steps)

            sr = sum(self._outcomes) / len(self._outcomes) * 100
            ms = sum(self._steps_hist) / len(self._steps_hist)
            elapsed = self._elapsed_offset + (time.time() - self._t0)

            tag = " 1  " if goal else (" 0  " if coll else " -  ")
            warmup_str = (f"  [WARMUP {self._total_steps}/{WARMUP}]"
                          if self._total_steps < WARMUP else "")
            print(
                f"Ep {self._ep:4d} | Ph {self._cur_phase} | {tag:4s} | "
                f"SR: {sr:5.1f}% | R: {self._ep_reward:+7.1f} | "
                f"Steps: {self._ep_steps:4d} | Upd: {self._updates:6d} | "
                f"Buf: {self.replay.size:5d} | Alp: {self._last_alpha:.3f} | "
                f"H_tgt: {self._target_entropy:.2f} | Sm: {self._last_smooth:.4f}{warmup_str}",
                flush=True)

            with open(self._log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self._ep, round(self._ep_reward, 2), self._ep_steps,
                    coll, goal, round(sr, 2), round(ms, 1),
                    self._updates, self.replay.size, round(self._last_alpha, 4),
                    round(self._target_entropy, 4), round(self._last_smooth, 6),
                    round(elapsed, 1)
                ])

            # ── checkpoint saves ───────────────────────────────────────────────
            if self._ep % SAVE_EVERY == 0:
                self._save_ckpt()
            if sr > self._best_sr and len(self._outcomes) >= 20:
                self._best_sr = sr
                self._save_ckpt(tag="best_sr")

            # ── MAX_EPISODES / MAX_STEPS stop ──────────────────────────────────
            if MAX_EPISODES > 0 and self._ep >= MAX_EPISODES:
                print(f"[SACv11] MAX_EPISODES={MAX_EPISODES} reached — SR={sr:.1f}%", flush=True)
                self._save_ckpt()
                rclpy.shutdown()
                return
            if MAX_STEPS > 0 and self._total_steps >= MAX_STEPS:
                print(f"[SACv11] MAX_STEPS={MAX_STEPS} reached — SR={sr:.1f}%", flush=True)
                self._save_ckpt(tag="finetune_final")
                rclpy.shutdown()
                return

            self._waiting_reset = True
        else:
            self._send_action()

    def _send_action(self):
        obs = np.array(self._ep_obs_history[-1], dtype=np.float32)
        if self._total_steps < WARMUP:
            tanh_act = np.random.uniform(-1, 1, ACT_DIM).astype(np.float32)
            self._current_act = tanh_act
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            tanh_act, self._actor_h = self.agent.select_action(obs_t, self._actor_h, eval_mode=False)
            self._current_act = tanh_act

        lin = float(LIN_MIN + (tanh_act[0] + 1.0) / 2.0 * (LIN_MAX - LIN_MIN))
        ang = float(tanh_act[1] * ANG_MAX)
        out = Float32MultiArray()
        out.data = [lin, ang]
        self._act_pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = TrainAgentSACv11()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == "__main__":
    main()
