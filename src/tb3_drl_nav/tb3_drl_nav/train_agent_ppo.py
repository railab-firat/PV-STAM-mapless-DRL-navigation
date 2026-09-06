#!/usr/bin/env python3
"""
train_agent_ppo.py  Phase 3  v8
=====================================================
PPO trainer node. Auto-resumes from latest checkpoint on every run.

CHANGES (v8):
  - Loads all hyperparameters from config/phase3_ppo.yaml (single source of truth).
  - Python constants are kept as fallback defaults if config file is missing.
  - Failure position CSV path now derived from config log_dir.

CHANGES (v7):
  - SEED=42 set for torch + numpy → reproducible results across runs.
  - LIN_MIN=0.0: robot can now stop (was 0.05); required for safe obstacle
    avoidance in disaster scenarios.
  - Value-target clipping added to critic loss (standard PPO).
  - Approximate KL divergence computed and logged per update.
  - Separate {run_id}_updates.csv logs per-update losses for paper figures.

SAVES:   Every SAVE_EVERY episodes (keeps last KEEP_CKPTS checkpoints)
         + on clean Ctrl+C shutdown.

GPU NOTE: Quadro M2000M is Maxwell (sm_50). PyTorch 1.12.1+cu113 supports
         sm_50 — GPU is used automatically. PyTorch >= 1.13 would need CPU.

USAGE:
    ros2 run tb3_drl_nav train_agent_ppo --ros-args -p run_id:="phase3_v1"
"""
import os
import csv
import glob
import time
import signal
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String

# TRANSIENT_LOCAL so trainer always receives reset_obs even if it starts late
RESET_OBS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ── config loading ─────────────────────────────────────────────────────────────
# Looks for config/phase3_ppo.yaml relative to the package root.
# With --symlink-install, __file__ resolves via symlink to the actual source file.
_HERE     = os.path.dirname(os.path.realpath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_CFG_PATH = os.environ.get(
    "PHASE3_CFG",
    os.path.join(_PKG_ROOT, "config", "phase3_ppo.yaml"))
try:
    import yaml
    with open(_CFG_PATH) as _f:
        _CFG = yaml.safe_load(_f) or {}
except Exception:
    _CFG = {}

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = _CFG.get("seed", 42)

# ── hyper-parameters ──────────────────────────────────────────────────────────
OBS_DIM       = _CFG.get("obs_dim",        54)
ACT_DIM       = _CFG.get("act_dim",         2)
ROLLOUT_LEN   = _CFG.get("rollout_len",  2048)   # Phase-2-proven steps per update
BATCH_SIZE    = _CFG.get("batch_size",    256)
EPOCHS        = _CFG.get("epochs",         10)
GAMMA         = _CFG.get("gamma",        0.99)
LAM           = _CFG.get("lam",          0.95)
CLIP_EPS      = _CFG.get("clip_eps",     0.20)
ENTROPY_COEF  = _CFG.get("entropy_coef", 0.01)   # Phase-2-proven
VF_COEF       = _CFG.get("vf_coef",      0.50)
LR            = _CFG.get("lr",          3e-4)
MAX_GRAD_NORM = _CFG.get("max_grad_norm", 0.5)
MAX_EPISODES  = _CFG.get("max_episodes", 99999)
SAVE_EVERY    = _CFG.get("save_every",    50)
KEEP_CKPTS    = _CFG.get("keep_ckpts",     3)
# Phase-2-proven: scale rewards before storing in buffer (100x smaller targets
# → faster, more stable critic convergence)
REWARD_SCALE  = _CFG.get("reward_scale",  0.01)
# Phase-2-proven: stop PPO epochs early if KL divergence exceeds this threshold
# prevents over-optimisation per update (Schulman et al. 2017)
TARGET_KL     = _CFG.get("target_kl",    0.02)

# ── velocity mapping ──────────────────────────────────────────────────────────
# tanh output ∈ (-1, 1) → lin ∈ [LIN_MIN, LIN_MAX]  ang ∈ [-ANG_MAX, ANG_MAX]
# LIN_MIN=0.0 allows the robot to stop — critical for dynamic obstacle avoidance
LIN_MIN = _CFG.get("lin_min", 0.0)
LIN_MAX = _CFG.get("lin_max", 0.26)
ANG_MAX = _CFG.get("ang_max", 1.82)


# ── actor-critic network ───────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(OBS_DIM),
            nn.Linear(OBS_DIM, 256), nn.Tanh(),
            nn.Linear(256, 128),     nn.Tanh(),
        )
        self.actor_head  = nn.Linear(128, ACT_DIM)
        self.critic_head = nn.Linear(128, 1)
        self.log_std     = nn.Parameter(torch.zeros(ACT_DIM))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x):
        f   = self.shared(x)
        mu  = self.actor_head(f)
        std = self.log_std.exp().expand_as(mu)
        val = self.critic_head(f).squeeze(-1)
        return mu, std, val

    def act(self, obs):
        """Sample action and return (tanh_action, raw_sample, log_prob, value)."""
        mu, std, val = self(obs)
        dist = Normal(mu, std)
        raw  = dist.rsample()
        act  = torch.tanh(raw)
        logp = (dist.log_prob(raw) - torch.log(1 - act.pow(2) + 1e-6)).sum(-1)
        return act, raw, logp, val

    def evaluate(self, obs, raw_act):
        """Re-evaluate stored actions for PPO update."""
        mu, std, val = self(obs)
        dist = Normal(mu, std)
        act  = torch.tanh(raw_act)
        logp = (dist.log_prob(raw_act) - torch.log(1 - act.pow(2) + 1e-6)).sum(-1)
        ent  = dist.entropy().sum(-1)
        return logp, val, ent


# ── rollout buffer ─────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.obs      = []
        self.raw_acts = []
        self.logps    = []
        self.vals     = []
        self.rews     = []
        self.dones    = []

    def add(self, obs, raw_act, logp, val, rew, done):
        self.obs.append(obs)
        self.raw_acts.append(raw_act)
        self.logps.append(logp)
        self.vals.append(val)
        self.rews.append(rew)
        self.dones.append(done)

    def __len__(self):
        return len(self.rews)

    def compute_gae(self, last_val, gamma, lam):
        adv, ret = [], []
        gae  = 0.0
        vals = self.vals + [last_val]
        for i in reversed(range(len(self.rews))):
            delta = self.rews[i] + gamma * vals[i + 1] * (1 - self.dones[i]) - vals[i]
            gae   = delta + gamma * lam * (1 - self.dones[i]) * gae
            adv.insert(0, gae)
            ret.insert(0, gae + vals[i])
        return adv, ret

    def tensors(self, device):
        t = lambda x: torch.tensor(np.array(x), dtype=torch.float32).to(device)
        return t(self.obs), t(self.raw_acts), t(self.logps), t(self.vals)

    def to_dict(self):
        return {
            "obs":      [o.tolist() if hasattr(o, "tolist") else list(o) for o in self.obs],
            "raw_acts": [a.tolist() if hasattr(a, "tolist") else list(a) for a in self.raw_acts],
            "logps":    list(self.logps),
            "vals":     list(self.vals),
            "rews":     list(self.rews),
            "dones":    list(self.dones),
        }

    @classmethod
    def from_dict(cls, d):
        buf          = cls()
        buf.obs      = [np.array(o, dtype=np.float32) for o in d["obs"]]
        buf.raw_acts = [np.array(a, dtype=np.float32) for a in d["raw_acts"]]
        buf.logps    = list(d["logps"])
        buf.vals     = list(d["vals"])
        buf.rews     = list(d["rews"])
        buf.dones    = list(d["dones"])
        return buf


# ── trainer node ──────────────────────────────────────────────────────────────
class TrainAgentPPO(Node):
    def __init__(self):
        super().__init__("train_agent_ppo")
        self.declare_parameter("run_id", "phase3_v1")
        run_id = self.get_parameter("run_id").value

        # ── reproducibility ───────────────────────────────────────────────────
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        self.get_logger().info(f"[Trainer] Seed: {SEED}")

        # ── device selection ──────────────────────────────────────────────────
        # PyTorch 2.0.1+cu118 supports sm_50+ (Maxwell and newer).
        # Quadro M2000M is Maxwell (sm_50) — fully supported by PyTorch 2.0.1+cu118.
        cuda_ok = False
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            sm  = cap[0] * 10 + cap[1]
            if sm >= 50:
                cuda_ok = True
            else:
                self.get_logger().warn(
                    f"[Trainer] GPU sm_{sm} not supported by this PyTorch "
                    f"(requires sm_50+) — using CPU.")
        self.device = torch.device("cuda" if cuda_ok else "cpu")
        if cuda_ok:
            dev_name = torch.cuda.get_device_name(0)
            self.get_logger().info(
                f"[Trainer] Device: cuda  ({dev_name}, sm_{sm})")
        else:
            self.get_logger().info("[Trainer] Device: cpu")

        # ── model / optimiser ─────────────────────────────────────────────────
        self.model = ActorCritic().to(self.device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.buf   = RolloutBuffer()

        # ── paths (from config/phase3_ppo.yaml; override via PHASE3_CFG env var)
        log_dir   = os.path.expanduser(_CFG.get("log_dir",   "~/tb3_drl_logs/phase3"))
        model_dir = os.path.expanduser(
            os.path.join(_CFG.get("model_dir", "~/tb3_drl_models/phase3"), run_id))
        os.makedirs(log_dir,   exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self._model_dir    = model_dir
        self._log_path     = os.path.join(log_dir, f"{run_id}.csv")
        self._update_log   = os.path.join(log_dir, f"{run_id}_updates.csv")
        self._run_id       = run_id

        # ── episode statistics ────────────────────────────────────────────────
        self._ep             = 0
        self._updates        = 0
        self._ep_reward      = 0.0
        self._ep_steps       = 0
        self._elapsed_offset = 0.0
        self._t0             = time.time()
        self._outcomes       = []    # rolling last 100: 1=goal 0=other
        self._steps_hist     = []
        self._best_sr        = 0.0

        # ── transaction state ─────────────────────────────────────────────────
        self._waiting_reset = True
        self._current_obs   = None
        self._current_raw   = None
        self._current_logp  = 0.0

        # ── resume from checkpoint ────────────────────────────────────────────
        self._resume()

        # ── CSV log ───────────────────────────────────────────────────────────
        if self._ep == 0:
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode", "reward", "steps",
                    "collision", "goal_reached",
                    "sr_100", "mean_steps_100",
                    "updates", "elapsed_s"])
            with open(self._update_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "update", "episode", "actor_loss", "critic_loss",
                    "entropy", "approx_kl", "elapsed_s"])
            self.get_logger().info(f"[Trainer] New log: {self._log_path}")
        else:
            self.get_logger().info(f"[Trainer] Appending to: {self._log_path}")

        # ── ROS pubs/subs ─────────────────────────────────────────────────────
        self._act_pub  = self.create_publisher(
            Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/reset_obs",   self._on_reset, RESET_OBS_QOS)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/step_result", self._on_step,  10)
        self._cur_goal = "?"   # updated by goal topic — stored as "#N"
        self.create_subscription(
            String, "/tb3_drl/goal", self._on_goal, 10)

        # ── signal handlers ───────────────────────────────────────────────────
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.get_logger().info(
            f"[Trainer] PPO v8 ready  run={run_id}  "
            f"ep={self._ep}  upd={self._updates}  "
            f"obs={OBS_DIM}  params={n_params:,}  "
            f"buf={len(self.buf)}/{ROLLOUT_LEN}  lr={LR}  "
            f"reward_scale={REWARD_SCALE}  target_kl={TARGET_KL}")
        self.get_logger().info("[Trainer] Waiting for environment reset_obs…")

    # ── resume ─────────────────────────────────────────────────────────────────
    def _resume(self):
        pattern = os.path.join(self._model_dir, "ckpt_ep*.pt")
        ckpts   = sorted(glob.glob(pattern))
        if not ckpts:
            self.get_logger().info("[Trainer] No checkpoint found — starting fresh.")
            return
        latest = ckpts[-1]
        try:
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["optimizer"])
            self._ep             = int(ckpt["episode"])
            self._updates        = int(ckpt["updates"])
            self._outcomes       = list(ckpt.get("outcomes",    []))
            self._steps_hist     = list(ckpt.get("steps_hist",  []))
            self._elapsed_offset = float(ckpt.get("elapsed_s",  0.0))
            self._best_sr        = float(ckpt.get("best_sr",    0.0))
            if "buffer" in ckpt and ckpt["buffer"]:
                try:
                    self.buf = RolloutBuffer.from_dict(ckpt["buffer"])
                except Exception as e:
                    self.get_logger().warn(
                        f"[Trainer] Buffer restore failed ({e}) — fresh buffer.")
                    self.buf = RolloutBuffer()
            self.model.to(self.device)
            sr = (sum(self._outcomes) / len(self._outcomes) * 100
                  if self._outcomes else 0.0)
            self.get_logger().info(
                f"[Trainer] Resumed: {os.path.basename(latest)}"
                f"  ep={self._ep}  upd={self._updates}"
                f"  SR={sr:.1f}%  buf={len(self.buf)}"
                f"  elapsed={self._elapsed_offset/3600:.2f}h")
        except Exception as e:
            self.get_logger().error(
                f"[Trainer] Checkpoint load FAILED ({e}) — starting fresh.")

    # ── checkpoint ─────────────────────────────────────────────────────────────
    def _save_ckpt(self):
        path    = os.path.join(self._model_dir, f"ckpt_ep{self._ep:06d}.pt")
        elapsed = self._elapsed_offset + (time.time() - self._t0)
        torch.save({
            "model":      self.model.state_dict(),
            "optimizer":  self.opt.state_dict(),
            "episode":    self._ep,
            "updates":    self._updates,
            "outcomes":   self._outcomes,
            "steps_hist": self._steps_hist,
            "elapsed_s":  elapsed,
            "best_sr":    self._best_sr,
            "buffer":     self.buf.to_dict() if len(self.buf) > 0 else None,
        }, path)
        torch.save(self.model.state_dict(),
                   os.path.join(self._model_dir, "model_latest.pt"))
        # Prune old checkpoints
        all_ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_ep*.pt")))
        for old in all_ckpts[:-KEEP_CKPTS]:
            try:
                os.remove(old)
            except OSError:
                pass
        self.get_logger().info(
            f"[Trainer] Checkpoint saved → {os.path.basename(path)}")

    # ── shutdown ────────────────────────────────────────────────────────────────
    def _shutdown(self, signum, frame):
        self.get_logger().info("[Trainer] Shutdown — saving checkpoint…")
        self._save_ckpt()
        self.get_logger().info(
            f"[Trainer] Saved at ep={self._ep}. Resume with the same command.")
        # rclpy may have already begun shutdown on SIGINT; guard against that.
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _on_reset(self, msg: Float32MultiArray):
        if len(msg.data) < OBS_DIM:
            self.get_logger().warn(f"[Trainer] Bad reset_obs len={len(msg.data)}")
            return
        obs = np.array(msg.data[:OBS_DIM], dtype=np.float32)
        self._current_obs   = obs
        self._current_raw   = None
        self._current_logp  = 0.0
        self._ep_reward     = 0.0
        self._ep_steps      = 0
        self._waiting_reset = False
        self._send_action(obs)

    def _on_step(self, msg: Float32MultiArray):
        if self._waiting_reset or self._current_raw is None:
            return
        if len(msg.data) < OBS_DIM + 3:
            self.get_logger().warn(f"[Trainer] Bad step_result len={len(msg.data)}")
            return

        obs    = np.array(msg.data[:OBS_DIM], dtype=np.float32)
        reward = float(msg.data[OBS_DIM])
        done   = bool(msg.data[OBS_DIM + 1])
        info   = int(msg.data[OBS_DIM + 2])

        self._ep_reward += reward
        self._ep_steps  += 1

        with torch.no_grad():
            obs_t = torch.tensor(
                self._current_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, _, val = self.model(obs_t)

        self.buf.add(
            self._current_obs,
            self._current_raw,
            self._current_logp,
            val.item(),
            reward * REWARD_SCALE,   # Phase-2-proven: scale rewards → stable critic
            float(done))

        self._current_obs = obs

        if done:
            self._end_episode(info)
        else:
            self._send_action(obs)

    def _on_goal(self, msg: String):
        # msg.data = "x.xxxx,y.yyyy" — display as coordinates
        try:
            x, y = [float(v) for v in msg.data.split(",")]
            d = math.sqrt(x*x + y*y)
            self._cur_goal = f"({x:+.1f},{y:+.1f}) {d:.1f}m"
        except Exception:
            self._cur_goal = "?"

    # ── episode end ────────────────────────────────────────────────────────────
    def _end_episode(self, info: int):
        # info=0 → spawn artifact (physically impossible collision at step ≤ BAD_SPAWN_GRACE)
        # Treat as a silent retry: don't count as an episode, don't update SR.
        if info == 0:
            self.get_logger().warn(
                f"[Trainer] SPAWN_ART  steps={self._ep_steps} — retry, not counted")
            self._waiting_reset = True
            return

        self._ep += 1
        goal      = 1 if info == 1 else 0
        collision = 1 if info == 2 else 0
        stuck     = 1 if info == 8 else 0

        self._outcomes.append(goal)
        self._steps_hist.append(self._ep_steps)
        if len(self._outcomes)   > 100: self._outcomes.pop(0)
        if len(self._steps_hist) > 100: self._steps_hist.pop(0)

        sr100   = sum(self._outcomes)   / len(self._outcomes)   * 100
        ms100   = sum(self._steps_hist) / len(self._steps_hist)
        elapsed = self._elapsed_offset + (time.time() - self._t0)

        tag = ("GOAL" if goal else
               "COLL" if collision else
               "STUK" if stuck else "TIME")
        self.get_logger().info(
            f"Ep {self._ep:5d} | {tag:4s} | "
            f"R={self._ep_reward:+8.1f} | "
            f"steps={self._ep_steps:4d} | "
            f"goal=({self._cur_goal}) | "
            f"SR={sr100:5.1f}% | "
            f"buf={len(self.buf)}/{ROLLOUT_LEN} | "
            f"upd={self._updates}")

        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self._ep, round(self._ep_reward, 2), self._ep_steps,
                collision, goal,
                round(sr100, 2), round(ms100, 1),
                self._updates, round(elapsed, 1)])

        if len(self.buf) >= ROLLOUT_LEN:
            self._ppo_update()

        if self._ep % SAVE_EVERY == 0:
            self._save_ckpt()

        # Best-SR checkpoint — saved whenever rolling SR improves
        if sr100 > self._best_sr and self._ep >= 50:
            self._best_sr = sr100
            best_path = os.path.join(self._model_dir, "ckpt_best_sr.pt")
            elapsed = self._elapsed_offset + (time.time() - self._t0)
            torch.save({
                "model":      self.model.state_dict(),
                "optimizer":  self.opt.state_dict(),
                "episode":    self._ep,
                "updates":    self._updates,
                "outcomes":   self._outcomes,
                "steps_hist": self._steps_hist,
                "elapsed_s":  elapsed,
                "best_sr":    self._best_sr,
                "buffer":     None,
            }, best_path)
            torch.save(self.model.state_dict(),
                       os.path.join(self._model_dir, "model_best_sr.pt"))
            self.get_logger().info(
                f"[Trainer] *** NEW BEST SR={sr100:.1f}% at ep={self._ep} → ckpt_best_sr.pt ***")

        self._waiting_reset = True

    # ── action ────────────────────────────────────────────────────────────────
    def _send_action(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act, raw, logp, _ = self.model.act(obs_t)
        self._current_raw  = raw.squeeze(0).cpu().numpy()
        self._current_logp = logp.item()
        # tanh ∈ (-1,1) → lin ∈ [LIN_MIN, LIN_MAX]  ang ∈ [-ANG_MAX, ANG_MAX]
        lin = float(LIN_MIN + (act[0, 0].item() + 1.0) / 2.0 * (LIN_MAX - LIN_MIN))
        ang = float(act[0, 1].item() * ANG_MAX)
        out      = Float32MultiArray()
        out.data = [lin, ang]
        self._act_pub.publish(out)

    # ── PPO update ────────────────────────────────────────────────────────────
    def _ppo_update(self):
        t0 = time.time()
        with torch.no_grad():
            last_t = torch.tensor(
                self._current_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, _, lv = self.model(last_t)
        adv, ret = self.buf.compute_gae(lv.item(), GAMMA, LAM)

        obs_t, raw_t, old_lp_t, old_val_t = self.buf.tensors(self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n  = len(obs_t)
        ta = tc = te = tkl = nb = 0.0

        kl_exceeded = False
        for _ in range(EPOCHS):
            if kl_exceeded:
                break
            for b in torch.randperm(n).split(BATCH_SIZE):
                if len(b) == 0:
                    continue
                logp, val, ent = self.model.evaluate(obs_t[b], raw_t[b])
                ratio = (logp - old_lp_t[b]).exp()

                # Actor loss (clipped surrogate objective)
                s1 = ratio * adv_t[b]
                s2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[b]
                al = -torch.min(s1, s2).mean()

                # Critic loss with value-target clipping (standard PPO)
                val_clipped = old_val_t[b] + (val - old_val_t[b]).clamp(
                    -CLIP_EPS, CLIP_EPS)
                cl = torch.max(
                    (ret_t[b] - val).pow(2),
                    (ret_t[b] - val_clipped).pow(2)).mean()

                loss = al + VF_COEF * cl - ENTROPY_COEF * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.opt.step()

                # Approximate KL divergence (Schulman's estimator)
                with torch.no_grad():
                    approx_kl = (old_lp_t[b] - logp).mean().item()
                ta  += al.item()
                tc  += cl.item()
                te  += ent.mean().item()
                tkl += approx_kl
                nb  += 1

                # Phase-2-proven: early stop if KL too large (prevents over-optimisation)
                if approx_kl > TARGET_KL:
                    kl_exceeded = True
                    break

        self._updates += 1
        elapsed = self._elapsed_offset + (time.time() - self._t0)
        avg_al  = ta  / max(nb, 1)
        avg_cl  = tc  / max(nb, 1)
        avg_ent = te  / max(nb, 1)
        avg_kl  = tkl / max(nb, 1)

        self.buf = RolloutBuffer()
        self.get_logger().info(
            f"[Trainer] Update #{self._updates}  "
            f"actor={avg_al:.4f}  "
            f"critic={avg_cl:.4f}  "
            f"entropy={avg_ent:.4f}  "
            f"kl={avg_kl:.4f}  "
            f"({time.time()-t0:.1f}s)")

        # Write to update log for paper figures
        with open(self._update_log, "a", newline="") as f:
            csv.writer(f).writerow([
                self._updates, self._ep,
                round(avg_al,  6), round(avg_cl,  6),
                round(avg_ent, 6), round(avg_kl,  6),
                round(elapsed, 1)])


def main(args=None):
    rclpy.init(args=args)
    node = TrainAgentPPO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
