#!/usr/bin/env python3
"""
train_agent_sac.py  SAC Trainer  (clean rewrite)
=====================================================================
Soft Actor-Critic (Haarnoja et al. 2018) for mapless navigation.

Based on CleanRL/SB3 reference implementations. Key design decisions:
  - NO reward scaling — raw rewards give strong learning signal (proven by v5)
  - NO alpha clamp — auto-entropy tuning works best unconstrained
  - Simple 2-layer MLPs — LayerNorm on actor input only
  - STAM module for DZ_MODE=stam (learned attention over LiDAR sectors)

USAGE:
    ros2 run tb3_drl_nav train_agent_sac --ros-args \\
        -p run_id:="sac_v7_velocity_dz" -p fresh:=true
"""
import os, csv, glob, time, signal, math, random, collections
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

# ── config ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.realpath(__file__))
_PKG  = os.path.dirname(_HERE)
_CFG_PATH = os.environ.get("PHASE3_CFG", os.path.join(_PKG, "config", "phase3_ppo.yaml"))
try:
    import yaml
    with open(_CFG_PATH) as f:
        _C = yaml.safe_load(f) or {}
except Exception:
    _C = {}

# ── hyperparameters ───────────────────────────────────────────────────────────
SEED       = int(os.environ.get("SEED", _C.get("seed", 42)))
OBS_DIM    = _C.get("obs_dim", 54)
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
TARGET_H   = float(_C.get("sac_target_entropy", -ACT_DIM))
DZ_MODE    = os.environ.get("DZ_MODE", "velocity_dz")
EVAL_ONLY  = os.environ.get("EVAL_ONLY", "").lower() in ("true", "1", "yes")
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", 0))  # 0 = unlimited


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORKS
# ══════════════════════════════════════════════════════════════════════════════

class ScanAttention(nn.Module):
    """STAM — learns per-sector importance from (distance, velocity) pairs."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, scan, scan_vel):
        x = torch.stack([scan, scan_vel], dim=-1)        # (B, 24, 2)
        w = self.net(x).squeeze(-1)                       # (B, 24)
        w = torch.softmax(w * 5.0, dim=-1) * 24.0        # normalise, mean≈1
        return scan * w, scan_vel * w


class Actor(nn.Module):
    """Squashed Gaussian policy. LayerNorm on input for observation scaling."""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self):
        super().__init__()
        self.attn = ScanAttention() if DZ_MODE == "stam" else None
        self.ln = nn.LayerNorm(OBS_DIM)
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.mu  = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(HIDDEN, ACT_DIM)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        for m in [self.mu, self.log_std]:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, obs):
        if self.attn is not None:
            s, v = obs[:, :24], obs[:, 24:48]
            s, v = self.attn(s, v)
            obs = torch.cat([s, v, obs[:, 48:]], dim=-1)
        x = F.relu(self.fc1(self.ln(obs)))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std.exp()

    def sample(self, obs):
        mu, std = self(obs)
        dist = Normal(mu, std)
        raw = dist.rsample()
        act = torch.tanh(raw)
        # Log-prob with tanh squashing correction
        log_prob = (dist.log_prob(raw) - torch.log(1.0 - act.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return act, log_prob

    def deterministic(self, obs):
        mu, _ = self(obs)
        return torch.tanh(mu)


class Critic(nn.Module):
    """Twin Q-networks. No LayerNorm — raw Q-values work fine (proven by v5)."""
    def __init__(self):
        super().__init__()
        self.attn = ScanAttention() if DZ_MODE == "stam" else None
        d = OBS_DIM + ACT_DIM
        # Q1
        self.q1_fc1 = nn.Linear(d, HIDDEN)
        self.q1_fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.q1_out = nn.Linear(HIDDEN, 1)
        # Q2
        self.q2_fc1 = nn.Linear(d, HIDDEN)
        self.q2_fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.q2_out = nn.Linear(HIDDEN, 1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        for m in [self.q1_out, self.q2_out]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def forward(self, obs, act):
        if self.attn is not None:
            s, v = obs[:, :24], obs[:, 24:48]
            s, v = self.attn(s, v)
            obs = torch.cat([s, v, obs[:, 48:]], dim=-1)
        x = torch.cat([obs, act], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2

    def q_min(self, obs, act):
        q1, q2 = self(obs, act)
        return torch.min(q1, q2)


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity):
        self._buf = collections.deque(maxlen=capacity)

    def add(self, obs, act, rew, nobs, done):
        self._buf.append((
            np.asarray(obs, np.float32), np.asarray(act, np.float32),
            float(rew), np.asarray(nobs, np.float32), float(done)))

    def sample(self, n, device):
        batch = random.sample(self._buf, n)
        o, a, r, no, d = zip(*batch)
        t = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)
        return t(o), t(a), t(r).unsqueeze(1), t(no), t(d).unsqueeze(1)

    def __len__(self):
        return len(self._buf)

    def to_list(self):
        return list(self._buf)

    @classmethod
    def from_list(cls, data, cap):
        buf = cls(cap)
        for item in data[-cap:]:
            buf._buf.append(tuple(item))
        return buf


# ══════════════════════════════════════════════════════════════════════════════
#  SAC AGENT (pure PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

class SAC:
    def __init__(self, device):
        self.device = device
        self.actor      = Actor().to(device)
        self.critic     = Critic().to(device)
        self.critic_tgt = Critic().to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Auto-entropy: log_alpha is unconstrained, alpha = exp(log_alpha)
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

    def update(self, replay):
        obs, act, rew, nobs, done = replay.sample(BATCH, self.device)
        alpha = self.log_alpha.exp().detach()

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            na, nlp = self.actor.sample(nobs)
            tq1, tq2 = self.critic_tgt(nobs, na)
            target = rew + GAMMA * (1 - done) * (torch.min(tq1, tq2) - alpha * nlp)

        q1, q2 = self.critic(obs, act)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_critic.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        # ── Actor update (freeze critic) ──────────────────────────────────
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

        # ── Temperature update (no clamp — let auto-tuning work) ─────────
        loss_t = -(self.log_alpha * (new_lp.detach() + TARGET_H)).mean()
        self.opt_alpha.zero_grad()
        loss_t.backward()
        self.opt_alpha.step()

        # ── Soft update target critics ────────────────────────────────────
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
            "log_alpha": self.log_alpha.item(),
            "opt_alpha": self.opt_alpha.state_dict(),
        }

    def load_state_dict(self, d, device):
        self.actor.load_state_dict(d["actor"], strict=False)
        self.critic.load_state_dict(d["critic"], strict=False)
        self.critic_tgt.load_state_dict(d["critic_tgt"], strict=False)
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_critic.load_state_dict(d["opt_critic"])
        self.log_alpha = torch.tensor(
            [d["log_alpha"]], dtype=torch.float32, requires_grad=True, device=device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
        self.opt_alpha.load_state_dict(d["opt_alpha"])
        self.actor.to(device)
        self.critic.to(device)
        self.critic_tgt.to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 TRAINER NODE
# ══════════════════════════════════════════════════════════════════════════════

class TrainAgentSAC(Node):
    def __init__(self):
        super().__init__("train_agent_sac")
        self.declare_parameter("run_id", "sac_v7_velocity_dz")
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

        # Agent + buffer
        self.agent  = SAC(self.device)
        self.replay = ReplayBuffer(BUFFER_CAP)

        # Paths
        log_dir = os.path.expanduser(_C.get("log_dir", "~/tb3_drl_logs/phase3"))
        model_dir_base = os.environ.get("SAC_MODEL_DIR", _C.get("sac_model_dir", "~/tb3_drl_models/sac"))
        model_dir = os.path.expanduser(os.path.join(model_dir_base, run_id))
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self._model_dir  = model_dir
        if EVAL_ONLY:
            self._log_path   = os.path.join(log_dir, f"{run_id}_eval.csv")
            self._update_log = os.path.join(log_dir, f"{run_id}_eval_updates.csv")
        else:
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

        # Resume or fresh
        if self._fresh:
            self.get_logger().info("[SAC] FRESH START — ignoring checkpoints.")
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

        # ROS2 pub/sub
        self._act_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self.create_subscription(Float32MultiArray, "/tb3_drl/reset_obs", self._on_reset, RESET_OBS_QOS)
        self.create_subscription(Float32MultiArray, "/tb3_drl/step_result", self._on_step, 10)
        self.create_subscription(String, "/tb3_drl/goal", self._on_goal, 10)
        self.create_subscription(Int32, "/tb3_drl/curriculum_phase", self._on_phase, RESET_OBS_QOS)

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        mode_str = "EVAL (deterministic, no training)" if EVAL_ONLY else "TRAIN"
        self.get_logger().info(
            f"[SAC] Ready  run={run_id}  DZ_MODE={DZ_MODE}  mode={mode_str}  "
            f"warmup={WARMUP}  grad_steps={GRAD_STEPS}  "
            f"target_H={TARGET_H}  alpha={self.agent.alpha:.3f}  "
            f"device={self.device}")

    # ── resume ────────────────────────────────────────────────────────────
    def _resume(self):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
        if not ckpts:
            return
        try:
            ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
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
            if ckpt.get("replay"):
                self.replay = ReplayBuffer.from_list(ckpt["replay"], BUFFER_CAP)
            if len(self.replay) == 0:
                rb = os.path.join(self._model_dir, "replay_buffer.pt")
                if os.path.exists(rb):
                    try:
                        d = torch.load(rb, map_location="cpu", weights_only=False)
                        self.replay = ReplayBuffer.from_list(d["replay"], BUFFER_CAP)
                    except Exception:
                        pass
            self.get_logger().info(
                f"[SAC] Resumed ep={self._ep}  buf={len(self.replay)}  "
                f"upd={self._updates}  elapsed={self._elapsed_offset/3600:.1f}h")
        except Exception as e:
            self.get_logger().error(f"[SAC] Resume failed ({e}) — fresh start.")

    # ── checkpoint ────────────────────────────────────────────────────────
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
            "replay": self.replay.to_list() if include_replay and len(self.replay) > 0 else None,
        }, path)
        torch.save(self.agent.actor.state_dict(),
                   os.path.join(self._model_dir, "actor_latest.pt"))
        # Prune old checkpoints
        for old in sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))[:-KEEP_CKPTS]:
            try: os.remove(old)
            except OSError: pass

    def _save_replay_backup(self):
        if len(self.replay) == 0:
            return
        torch.save({
            "replay": self.replay.to_list(),
            "episode": self._ep,
        }, os.path.join(self._model_dir, "replay_buffer.pt"))

    def _shutdown(self, signum, frame):
        self.get_logger().info("[SAC] Shutdown — saving with replay…")
        self._save_ckpt(include_replay=True)
        try: rclpy.shutdown()
        except RuntimeError: pass

    # ── ROS callbacks ─────────────────────────────────────────────────────
    def _on_reset(self, msg):
        if len(msg.data) < OBS_DIM:
            return
        obs = np.array(msg.data[:OBS_DIM], dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            return
        self._current_obs = obs
        self._current_act = None
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._waiting_reset = False
        self._send_action(obs)

    def _on_step(self, msg):
        if self._waiting_reset or self._current_act is None:
            return
        if len(msg.data) < OBS_DIM + 3:
            return

        nobs   = np.array(msg.data[:OBS_DIM], dtype=np.float32)
        reward = float(msg.data[OBS_DIM])
        done   = bool(msg.data[OBS_DIM + 1])
        info   = int(msg.data[OBS_DIM + 2])

        if not (np.all(np.isfinite(nobs)) and math.isfinite(reward)):
            return

        self._ep_reward += reward
        self._ep_steps += 1
        self._total_steps += 1

        if not EVAL_ONLY:
            # Store raw reward — NO scaling
            self.replay.add(self._current_obs, self._current_act, reward, nobs, float(done))

            # SAC gradient updates
            if (self._total_steps > WARMUP
                    and len(self.replay) >= BATCH
                    and self._total_steps % UPDATE_EVERY == 0):
                for _ in range(GRAD_STEPS):
                    la, lc, ent, alpha = self.agent.update(self.replay)
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
            self.get_logger().info(f"[SAC] Phase {old} → {msg.data}")

    def _on_goal(self, msg):
        try:
            x, y = [float(v) for v in msg.data.split(",")]
            self._cur_goal = f"({x:+.1f},{y:+.1f}) {math.hypot(x,y):.1f}m"
        except Exception:
            self._cur_goal = "?"

    # ── episode end ───────────────────────────────────────────────────────
    def _end_episode(self, info):
        if info == 0:  # spawn artifact
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

        sr = sum(self._outcomes) / len(self._outcomes) * 100
        ms = sum(self._steps_hist) / len(self._steps_hist)
        elapsed = self._elapsed_offset + (time.time() - self._t0)

        tag = "GOAL" if goal else ("COLL" if collision else ("STUK" if info == 8 else "TIME"))
        warmup = f"  [WARMUP {self._total_steps}/{WARMUP}]" if self._total_steps < WARMUP else ""
        self.get_logger().info(
            f"Ep {self._ep:5d} | Ph{self._cur_phase} | {tag} | "
            f"R={self._ep_reward:+8.1f} | steps={self._ep_steps:4d} | "
            f"goal={self._cur_goal} | SR={sr:5.1f}% | "
            f"buf={len(self.replay):,} | upd={self._updates}{warmup}")

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
                    "total_steps": self._total_steps, "updates": self._updates,
                    "outcomes": self._outcomes, "steps_hist": self._steps_hist,
                    "elapsed_s": elapsed, "replay": None,
                }, best)
                torch.save(self.agent.actor.state_dict(),
                           os.path.join(self._model_dir, "actor_best_sr.pt"))
                self.get_logger().info(f"[SAC] *** NEW BEST SR={sr:.1f}% at ep={self._ep} ***")

        # Auto-stop after MAX_EPISODES
        if MAX_EPISODES > 0 and self._ep >= MAX_EPISODES:
            self.get_logger().info(
                f"[SAC] Reached MAX_EPISODES={MAX_EPISODES} — stopping. Final SR={sr:.1f}%")
            self._save_ckpt()
            rclpy.shutdown()
            return

        self._waiting_reset = True

    # ── action ────────────────────────────────────────────────────────────
    def _send_action(self, obs):
        if EVAL_ONLY:
            tanh_act = self.agent.select_action(obs, deterministic=True)
        elif self._total_steps < WARMUP:
            tanh_act = np.random.uniform(-1, 1, ACT_DIM).astype(np.float32)
        else:
            tanh_act = self.agent.select_action(obs)

        self._current_act = tanh_act
        lin = float(LIN_MIN + (tanh_act[0] + 1.0) / 2.0 * (LIN_MAX - LIN_MIN))
        ang = float(tanh_act[1] * ANG_MAX)
        out = Float32MultiArray()
        out.data = [lin, ang]
        self._act_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrainAgentSAC()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try: node.destroy_node()
        except Exception: pass
        try:
            if rclpy.ok(): rclpy.shutdown()
        except Exception: pass


if __name__ == "__main__":
    main()
