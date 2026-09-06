#!/usr/bin/env python3
"""
sac_lstm.py  —  SAC-LSTM Baseline (Experiment 2)
==========================================================================
Plain LSTM encoder replacing the GRU+PV-STAM block from SAC-R-PV-STAM.

Architecture:
  Observation: 30-dim = lidar[24] + nav[6]
    nav = [dn, an, nav_vel_lin, nav_vel_ang, min_scan, min_approach]
    (same 6 nav slots as v11, but NO scan_vel — no velocity channel)

  LSTMActor:
    LayerNorm(30) → Linear(30→64) → ReLU → LSTM(64,64) → mu/log_std heads
  LSTMCritic:
    LayerNorm(30+2) → Linear(32→64) → ReLU → LSTM(64,64) → Q1, Q2

  ReplayBuffer: trajectory-based (SEQ_LEN=16, BURN_IN=8) same as v11
  SAC:          Huber critic, entropy annealing, same hypers as v11

Usage:
    SEED=42 ros2 run tb3_drl_nav sac_lstm --ros-args -p run_id:=sac_lstm_s42 -p fresh:=true
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

# ── Config ────────────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "phase3_ppo.yaml")
try:
    import yaml
    with open(_CFG_PATH) as _f:
        _C = yaml.safe_load(_f) or {}
except Exception:
    _C = {}

RESET_OBS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED         = int(os.environ.get("SEED", _C.get("seed", 42)))
RAW_OBS      = _C.get("obs_dim", 54)
ACT_DIM      = 2
LIN_MAX      = 0.26
ANG_MAX      = 1.82

LSTM_OBS_DIM = 30          # 24 LiDAR + 6 nav (no scan_vel)
LSTM_HIDDEN  = 64
SEQ_LEN      = int(_C.get("sac_v11_seq_len",  16))
BURN_IN      = int(_C.get("sac_v11_burn_in",   8))
TOTAL_SEQ    = SEQ_LEN + BURN_IN

BUFFER_CAP   = int(_C.get("sac_buffer",    200_000))
BATCH        = int(_C.get("sac_v11_batch", _C.get("sac_batch", 64)))
GAMMA        = _C.get("gamma",         0.99)
TAU          = _C.get("sac_tau",       0.005)
LR_ACTOR     = _C.get("sac_lr_actor",  3e-4)
LR_CRITIC    = _C.get("sac_lr_critic", 3e-4)
LR_ALPHA     = _C.get("sac_lr_alpha",  3e-4)
HUBER_DELTA  = float(_C.get("sac_v11_huber_delta", 10.0))
GRAD_CLIP    = float(_C.get("sac_grad_clip", 1.0))

ENT_START    = float(_C.get("sac_v11_entropy_start",       -1.0))
ENT_END      = float(_C.get("sac_v11_entropy_end",    -ACT_DIM))
ENT_ANNEAL   = int(_C.get("sac_v11_entropy_anneal_eps",      500))

WARMUP       = int(_C.get("sac_warmup",  10_000))
SAVE_EVERY   = int(_C.get("save_every",       50))
KEEP_CKPTS   = int(_C.get("keep_ckpts",        8))
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", _C.get("max_episodes", 0)))
EVAL_ONLY    = os.environ.get("EVAL_ONLY", "").lower() in ("true", "1", "yes")   # [lstm-eval]


def _build_lstm_obs(raw: np.ndarray) -> np.ndarray:
    """Slice 54-dim raw obs → 30-dim LSTM obs (drop scan_vel[24:48])."""
    return np.concatenate([raw[:24], raw[48:54]])


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER  (trajectory-based, identical to v11)
# ══════════════════════════════════════════════════════════════════════════════
class TrajectoryReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.obs  = np.zeros((capacity, LSTM_OBS_DIM), dtype=np.float32)
        self.act  = np.zeros((capacity, ACT_DIM),      dtype=np.float32)
        self.rew  = np.zeros((capacity, 1),            dtype=np.float32)
        self.nobs = np.zeros((capacity, LSTM_OBS_DIM), dtype=np.float32)
        self.done = np.zeros((capacity, 1),            dtype=np.float32)
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

    def state_dict(self):
        """Serialise the buffer so a resume does not lose accumulated experience.

        Only the first `size` rows hold data. While the buffer has not yet wrapped,
        ptr == size and rows [0:size] are in insertion order; once it has wrapped,
        size == capacity and [0:size] is the whole array. Saving [0:size] plus ptr
        is therefore exact in both cases, and keeps early checkpoints small.
        """
        n = self.size
        return {"obs":  self.obs[:n].copy(),  "act":  self.act[:n].copy(),
                "rew":  self.rew[:n].copy(),  "nobs": self.nobs[:n].copy(),
                "done": self.done[:n].copy(),
                "ptr":  int(self.ptr), "size": int(self.size),
                "capacity": int(self.capacity)}

    def load_state_dict(self, sd):
        n = min(int(sd.get("size", 0)), self.capacity)
        if n <= 0:
            return
        self.obs[:n]  = sd["obs"][:n]
        self.act[:n]  = sd["act"][:n]
        self.rew[:n]  = sd["rew"][:n]
        self.nobs[:n] = sd["nobs"][:n]
        self.done[:n] = sd["done"][:n]
        self.size = n
        self.ptr  = int(sd.get("ptr", n)) % self.capacity

    def sample(self, batch_size, seq_length, device):
        batch_obs, batch_act, batch_rew, batch_nobs, batch_done = [], [], [], [], []
        valid = 0
        while valid < batch_size:
            idx = random.randint(0, self.size - seq_length - 1)
            if idx <= self.ptr < idx + seq_length: continue
            if np.any(self.done[idx: idx + seq_length - 1]): continue
            batch_obs.append(self.obs[idx:  idx + seq_length])
            batch_act.append(self.act[idx:  idx + seq_length])
            batch_rew.append(self.rew[idx:  idx + seq_length])
            batch_nobs.append(self.nobs[idx: idx + seq_length])
            batch_done.append(self.done[idx: idx + seq_length])
            valid += 1
        t = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)
        return t(batch_obs), t(batch_act), t(batch_rew), t(batch_nobs), t(batch_done)


# ══════════════════════════════════════════════════════════════════════════════
#  NETWORKS
# ══════════════════════════════════════════════════════════════════════════════
class LSTMActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln      = nn.LayerNorm(LSTM_OBS_DIM)
        self.fc      = nn.Sequential(nn.Linear(LSTM_OBS_DIM, LSTM_HIDDEN), nn.ReLU())
        self.lstm    = nn.LSTM(LSTM_HIDDEN, LSTM_HIDDEN, batch_first=True)
        self.mu      = nn.Linear(LSTM_HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(LSTM_HIDDEN, ACT_DIM)

    def forward(self, obs, h=None):
        """obs: (B, T, 30);  h: tuple (h_n, c_n) or None."""
        B, T, _ = obs.shape
        x = self.fc(self.ln(obs.reshape(B * T, -1))).reshape(B, T, LSTM_HIDDEN)
        lstm_out, h_new = self.lstm(x, h)
        return (self.mu(lstm_out),
                self.log_std(lstm_out).clamp(-5.0, 2.0).exp(),
                h_new)

    def sample(self, obs, h=None):
        mu, std, h_new = self(obs, h)
        raw  = Normal(mu, std).rsample()
        act  = torch.tanh(raw)
        lp   = (Normal(mu, std).log_prob(raw)
                - torch.log(1 - act.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return act, lp, h_new

    def deterministic(self, obs, h=None):
        mu, _, h_new = self(obs, h)
        return torch.tanh(mu), h_new


class LSTMCritic(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = LSTM_OBS_DIM + ACT_DIM
        self.ln1    = nn.LayerNorm(in_dim)
        self.q1_fc  = nn.Sequential(nn.Linear(in_dim, LSTM_HIDDEN), nn.ReLU())
        self.q1_lstm = nn.LSTM(LSTM_HIDDEN, LSTM_HIDDEN, batch_first=True)
        self.q1_out  = nn.Linear(LSTM_HIDDEN, 1)

        self.ln2    = nn.LayerNorm(in_dim)
        self.q2_fc  = nn.Sequential(nn.Linear(in_dim, LSTM_HIDDEN), nn.ReLU())
        self.q2_lstm = nn.LSTM(LSTM_HIDDEN, LSTM_HIDDEN, batch_first=True)
        self.q2_out  = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, obs, act, h1=None, h2=None):
        B, T, _ = obs.shape
        x = torch.cat([obs, act], dim=-1)
        xf = x.reshape(B * T, -1)

        q1 = self.q1_fc(self.ln1(xf)).reshape(B, T, LSTM_HIDDEN)
        q1_out, h1_new = self.q1_lstm(q1, h1)
        q1_val = self.q1_out(q1_out)

        q2 = self.q2_fc(self.ln2(xf)).reshape(B, T, LSTM_HIDDEN)
        q2_out, h2_new = self.q2_lstm(q2, h2)
        q2_val = self.q2_out(q2_out)

        return q1_val, q2_val, h1_new, h2_new


# ══════════════════════════════════════════════════════════════════════════════
#  SAC AGENT
# ══════════════════════════════════════════════════════════════════════════════
class RecurrentSAC_LSTM:
    def __init__(self, device):
        self.device     = device
        self.actor      = LSTMActor().to(device)
        self.critic     = LSTMCritic().to(device)
        self.critic_tgt = LSTMCritic().to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters(): p.requires_grad = False

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha  = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

    def select_action(self, obs_seq, h=None, eval_mode=False):
        with torch.no_grad():
            mu, std, h_new = self.actor(obs_seq, h)
            act = torch.tanh(mu) if eval_mode else torch.tanh(Normal(mu, std).rsample())
        return act[0, -1, :].cpu().numpy(), h_new

    def update(self, replay, target_entropy):
        obs, act, rew, nobs, done = replay.sample(BATCH, TOTAL_SEQ, self.device)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            # Burn-in
            _, _, h_a  = self.actor(obs[:, :BURN_IN], h=None)
            na_b, _, h_ta = self.actor.sample(nobs[:, :BURN_IN], h=None)
            _, _, h_c1, h_c2 = self.critic(obs[:, :BURN_IN], act[:, :BURN_IN], None, None)
            _, _, h_tc1, h_tc2 = self.critic_tgt(nobs[:, :BURN_IN], na_b, None, None)

        h_a  = (h_a[0].detach(),  h_a[1].detach())
        h_ta = (h_ta[0].detach(), h_ta[1].detach())
        h_c1 = (h_c1[0].detach(), h_c1[1].detach())
        h_c2 = (h_c2[0].detach(), h_c2[1].detach())
        h_tc1 = (h_tc1[0].detach(), h_tc1[1].detach())
        h_tc2 = (h_tc2[0].detach(), h_tc2[1].detach())

        obs_a, nobs_a = obs[:, BURN_IN:], nobs[:, BURN_IN:]
        act_a, rew_a, done_a = act[:, BURN_IN:], rew[:, BURN_IN:], done[:, BURN_IN:]

        with torch.no_grad():
            na, nlp, _ = self.actor.sample(nobs_a, h=h_ta)
            tq1, tq2, _, _ = self.critic_tgt(nobs_a, na, h_tc1, h_tc2)
            target = rew_a + GAMMA * (1 - done_a) * (torch.min(tq1, tq2) - alpha * nlp)

        q1, q2, _, _ = self.critic(obs_a, act_a, h_c1, h_c2)
        loss_c = (F.huber_loss(q1, target, delta=HUBER_DELTA)
                  + F.huber_loss(q2, target, delta=HUBER_DELTA))
        self.opt_critic.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        for p in self.critic.parameters(): p.requires_grad = False
        new_act, new_lp, _ = self.actor.sample(obs_a, h=h_a)
        q1_pi, q2_pi, _, _ = self.critic(obs_a, new_act, h_c1, h_c2)
        loss_a = (alpha * new_lp - torch.min(q1_pi, q2_pi)).mean()
        self.opt_actor.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
        for p in self.critic.parameters(): p.requires_grad = True

        loss_t = -(self.log_alpha * (new_lp.detach() + target_entropy)).mean()
        self.opt_alpha.zero_grad()
        loss_t.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - TAU).add_(TAU * p.data)

        return loss_a.item(), loss_c.item(), self.log_alpha.exp().item()


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 TRAINING NODE
# ══════════════════════════════════════════════════════════════════════════════
class TrainAgentSACLSTM(Node):
    def __init__(self):
        super().__init__("train_agent_sac_lstm")
        self.declare_parameter("run_id", "sac_lstm_s42")
        self.declare_parameter("fresh",  False)
        run_id      = self.get_parameter("run_id").value
        self._fresh = self.get_parameter("fresh").value

        torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent  = RecurrentSAC_LSTM(self.device)
        self.replay = TrajectoryReplayBuffer(BUFFER_CAP)

        self._ep_obs_history = collections.deque(maxlen=TOTAL_SEQ)
        self._actor_h        = None   # LSTM (h, c) tuple

        self._total_steps = self._ep = self._updates = 0
        self._ep_reward   = self._ep_steps = 0
        self._waiting_reset = True
        self._current_act   = np.zeros(ACT_DIM, dtype=np.float32)
        self._elapsed_offset = 0.0
        self._cur_phase  = 1
        self._outcomes   = collections.deque(maxlen=100)
        self._steps_hist = collections.deque(maxlen=100)
        self._best_sr    = 0.0
        self._t0         = time.time()

        log_dir = os.path.expanduser(_C.get("log_dir", "~/tb3_drl_logs/phase3"))
        os.makedirs(log_dir, exist_ok=True)
        # [lstm-eval] evaluation writes its OWN file — never the training log
        _ph = os.environ.get("EVAL_PHASE", "")
        _ph_tag = f"_ph{_ph}" if (EVAL_ONLY and _ph) else ""
        _suffix = "_eval" if EVAL_ONLY else ""
        self._log_path = os.path.join(log_dir, f"{run_id}{_ph_tag}{_suffix}.csv")

        model_base = os.environ.get(
            "SAC_MODEL_DIR", _C.get("sac_model_dir", "~/tb3_drl_models/sac"))
        self._model_dir = os.path.expanduser(os.path.join(model_base, run_id))
        os.makedirs(self._model_dir, exist_ok=True)

        if self._fresh:
            print(f"[LSTM] FRESH START.", flush=True)
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode", "reward", "steps", "collision", "goal_reached",
                    "sr_100", "mean_steps_100", "updates", "buffer_size",
                    "alpha", "target_entropy", "phase", "elapsed_s"])
        else:
            self._resume()
            if EVAL_ONLY:                          # [lstm-eval]
                _done = 0
                if os.path.exists(self._log_path):
                    try:
                        with open(self._log_path) as _rf:
                            _done = sum(1 for _l in _rf
                                        if _l.split(',')[0].strip().isdigit())
                    except OSError:
                        _done = 0
                self._ep = _done
                self._outcomes.clear()
                self._steps_hist.clear()
            if not os.path.exists(self._log_path):
                with open(self._log_path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "episode", "reward", "steps", "collision", "goal_reached",
                        "sr_100", "mean_steps_100", "updates", "buffer_size",
                        "alpha", "target_entropy", "phase", "elapsed_s"])

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        self._act_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self.create_subscription(Float32MultiArray, "/tb3_drl/reset_obs",    self._on_reset, RESET_OBS_QOS)
        self.create_subscription(Float32MultiArray, "/tb3_drl/step_result",  self._on_step,  10)
        self.create_subscription(Int32, "/tb3_drl/curriculum_phase",         self._on_phase, RESET_OBS_QOS)

        print(f"\n[LSTM] device={self.device}  ep={self._ep}  seed={SEED}", flush=True)

    @property
    def _target_entropy(self):
        t = min(1.0, self._ep / max(1, ENT_ANNEAL))
        return ENT_START + t * (ENT_END - ENT_START)

    # ── Checkpoint ────────────────────────────────────────────────────────────
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
            # [BUFFER-PERSIST 2026-08-21] Without these the replay buffer and the
            # rolling-SR window were rebuilt from empty on every resume. In
            # lstm_forced_s42 that wiped a full 200,000-transition buffer at
            # episode 2551 (phase 7) and SR_100 fell 29% -> 12% immediately after.
            # sac_v10_matched.py already persisted its buffer, so the two agents
            # were not being trained under comparable conditions.
            "replay":      self.replay.state_dict(),
            "outcomes":    list(self._outcomes),
            "steps_hist":  list(self._steps_hist),
        }, path)
        if tag is None:
            rolling = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
            for old in rolling[:-KEEP_CKPTS]:
                os.remove(old)
        print(f"[LSTM] Saved {path}", flush=True)

    def _resume(self):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
        if not ckpts: return
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
            # [BUFFER-PERSIST 2026-08-21] Restore experience and the rolling-SR
            # window. `.get` keeps this backward-compatible: checkpoints written
            # before this change simply carry none, and behave as they did before.
            if ckpt.get("replay"):
                self.replay.load_state_dict(ckpt["replay"])
            if ckpt.get("outcomes"):
                self._outcomes.clear();   self._outcomes.extend(ckpt["outcomes"])
            if ckpt.get("steps_hist"):
                self._steps_hist.clear(); self._steps_hist.extend(ckpt["steps_hist"])
            print(f"[LSTM] Resumed ep={self._ep}  steps={self._total_steps}  "
                  f"buffer={self.replay.size:,}", flush=True)
        except Exception as e:
            print(f"[LSTM] Resume failed ({e}) — fresh start.", flush=True)

    def _shutdown(self, *_):
        print("\n[LSTM] Shutdown — saving…", flush=True)
        self._save_ckpt(tag="shutdown")
        try: rclpy.shutdown()
        except Exception: pass

    def _on_phase(self, msg: Int32):
        self._cur_phase = msg.data

    def _on_reset(self, msg: Float32MultiArray):
        raw = np.array(msg.data[:RAW_OBS], dtype=np.float32)
        lstm_obs = _build_lstm_obs(raw)
        self._ep_obs_history.clear()
        self._actor_h   = None
        self._ep_reward = 0.0
        self._ep_steps  = 0
        self._waiting_reset = False
        self._ep_obs_history.append(lstm_obs)
        self._send_action()

    def _on_step(self, msg: Float32MultiArray):
        if self._waiting_reset: return
        raw    = np.array(msg.data[:RAW_OBS], dtype=np.float32)
        lstm_obs = _build_lstm_obs(raw)
        reward = float(msg.data[RAW_OBS])
        done   = bool(msg.data[RAW_OBS + 1])
        info   = int(msg.data[RAW_OBS + 2])

        # Spawn artifact (collision at step<30): don't count as episode, just reset.
        if done and info == 0:
            self._ep_reward = 0.0; self._ep_steps = 0
            self._ep_obs_history.clear()
            self._waiting_reset = True; return

        cur_obs = np.array(self._ep_obs_history[-1], dtype=np.float32)
        self._ep_obs_history.append(lstm_obs)
        self._ep_reward   += reward
        self._ep_steps    += 1
        self._total_steps += 1

        if done and self._ep_steps > 0:
            self.replay.add(cur_obs, self._current_act, reward, lstm_obs, float(done))
        elif not done:
            self.replay.add(cur_obs, self._current_act, reward, lstm_obs, 0.0)

        if (self._total_steps > WARMUP
                and self.replay.size >= BATCH * TOTAL_SEQ):
            if not EVAL_ONLY:                      # [lstm-eval]
                self.agent.update(self.replay, self._target_entropy)
            self._updates += 1

        if done:
            self._ep += 1
            goal = 1 if info == 1 else 0
            coll = 1 if info == 2 else 0
            self._outcomes.append(goal)
            self._steps_hist.append(self._ep_steps)
            sr  = sum(self._outcomes) / len(self._outcomes) * 100
            ms  = sum(self._steps_hist) / len(self._steps_hist)
            ela = self._elapsed_offset + (time.time() - self._t0)
            tag = "GOAL" if goal else ("COLL" if coll else "TIME")
            print(f"Ep {self._ep:5d} | Ph{self._cur_phase} | {tag} | "
                  f"R={self._ep_reward:+8.1f} | steps={self._ep_steps:4d} | "
                  f"SR={sr:5.1f}% | buf={self.replay.size:,} | upd={self._updates}",
                  flush=True)
            with open(self._log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self._ep, round(self._ep_reward, 2), self._ep_steps,
                    coll, goal, round(sr, 2), round(ms, 1),
                    self._updates, self.replay.size,
                    round(self.agent.log_alpha.exp().item(), 4),
                    round(self._target_entropy, 3),
                    self._cur_phase, round(ela, 1)])

            if self._ep % SAVE_EVERY == 0:
                if not EVAL_ONLY:                  # [lstm-eval]
                    self._save_ckpt()
            if sr / 100.0 > self._best_sr:
                self._best_sr = sr / 100.0
                if not EVAL_ONLY:                  # [lstm-eval]
                    self._save_ckpt(tag="best_sr")

            if MAX_EPISODES > 0 and self._ep >= MAX_EPISODES:
                self._shutdown()
                return

            self._waiting_reset = True
            self._ep_obs_history.clear()
        else:
            self._send_action()

    def _send_action(self):
        # Build (B=1, T, 30) tensor from obs history
        hist = list(self._ep_obs_history)
        while len(hist) < TOTAL_SEQ:
            hist.insert(0, hist[0])
        obs_seq = torch.tensor(
            np.array(hist[-TOTAL_SEQ:]), dtype=torch.float32,
            device=self.device).unsqueeze(0)

        # [lstm-eval] deterministic policy, no exploration, during evaluation
        if EVAL_ONLY:
            act_np, self._actor_h = self.agent.select_action(
                obs_seq, self._actor_h, eval_mode=True)
        elif self._total_steps < WARMUP:
            act_np = np.random.uniform(-1, 1, 2).astype(np.float32)
            _, self._actor_h = self.agent.select_action(
                obs_seq, self._actor_h, eval_mode=False)
        else:
            act_np, self._actor_h = self.agent.select_action(
                obs_seq, self._actor_h, eval_mode=False)
        self._current_act = act_np

        lin = float(np.clip(act_np[0], -1, 1)) * LIN_MAX
        ang = float(np.clip(act_np[1], -1, 1)) * ANG_MAX
        lin = max(0.0, lin)   # forward-only clamp

        msg = Float32MultiArray()
        msg.data = [lin, ang]
        self._act_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrainAgentSACLSTM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
