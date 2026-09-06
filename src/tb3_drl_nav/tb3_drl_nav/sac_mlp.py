#!/usr/bin/env python3
"""
sac_baseline.py  Ablation Study: BASELINE SAC
=================================================================
Ablation role : Control group — no temporal memory, no attention.
Architecture  : MLP Actor (256→256) + Twin MLP Critic (256→256)
Observation   : 54-dim raw (24 scan + 24 scan_vel + 6 nav)
Replay        : Uniform replay buffer  (200 K)
Returns       : 1-step TD
Critic loss   : MSE
Entropy target: Fixed  −act_dim = −2.0

Compare against:
  sac_v8.py   — adds STAM attention + PER + 3-step returns
  sac_v10.py  — adds wider critic (384) + Huber loss
  sac_v11.py  — adds GRU recurrence (replaces frame-stack)

Run:
  SEED=42 ros2 run tb3_drl_nav sac_baseline --ros-args \\
      -p run_id:=sac_baseline_s42 -p fresh:=true
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

# ── QoS for latched reset_obs topic ──────────────────────────────────────────
_LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

# ── Config (shared source-of-truth with all agents) ──────────────────────────
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
OBS_DIM      = _C.get("obs_dim", 54)        # 24 scan + 24 scan_vel + 6 nav
ACT_DIM      = _C.get("act_dim", 2)
LIN_MIN      = _C.get("lin_min", 0.0)
LIN_MAX      = _C.get("lin_max", 0.26)
ANG_MAX      = _C.get("ang_max", 1.82)

# ── [ABLATION] Network width — actor and critic both 256 (no widening) ────────
HIDDEN       = 256

BUFFER_CAP   = _C.get("sac_buffer", 200_000)
BATCH        = _C.get("sac_batch",  256)
GAMMA        = _C.get("gamma",      0.99)
TAU          = _C.get("sac_tau",    0.005)
LR_ACTOR     = _C.get("sac_lr_actor",  3e-4)
LR_CRITIC    = _C.get("sac_lr_critic", 3e-4)
LR_ALPHA     = _C.get("sac_lr_alpha",  3e-4)
WARMUP       = _C.get("sac_warmup",    10_000)
GRAD_CLIP    = _C.get("sac_grad_clip", 1.0)
SAVE_EVERY   = _C.get("save_every",    50)
KEEP_CKPTS   = _C.get("keep_ckpts",    8)
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", _C.get("max_episodes", 0)))

# ── [ABLATION] Entropy target — standard SAC: -act_dim ───────────────────────
TARGET_ENTROPY = float(_C.get("sac_target_entropy", -ACT_DIM))  # default -2.0

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
#  REPLAY BUFFER  — [ABLATION] Uniform sampling (no prioritization)
# ══════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity):
        self._buf = collections.deque(maxlen=capacity)

    def add(self, obs, act, rew, nobs, done):
        self._buf.append((
            np.asarray(obs,  np.float32), np.asarray(act, np.float32),
            float(rew), np.asarray(nobs, np.float32), float(done)))

    def sample(self, n, device):
        batch = random.sample(self._buf, n)
        o, a, r, no, d = zip(*batch)
        t = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)
        return t(o), t(a), t(r).unsqueeze(1), t(no), t(d).unsqueeze(1)

    def __len__(self):
        return len(self._buf)

    def to_list(self):   return list(self._buf)

    @classmethod
    def from_list(cls, data, cap):
        buf = cls(cap)
        for item in data[-cap:]:
            buf._buf.append(tuple(item))
        return buf


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORKS  — [ABLATION] Plain MLP, no attention, no recurrence
# ══════════════════════════════════════════════════════════════════════════════
class Actor(nn.Module):
    """Squashed-Gaussian policy. LayerNorm on input; two hidden layers."""
    def __init__(self):
        super().__init__()
        self.ln  = nn.LayerNorm(OBS_DIM)
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN,  HIDDEN)
        self.mu      = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(HIDDEN, ACT_DIM)
        self._init()

    def _init(self):
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2)); nn.init.zeros_(m.bias)
        for m in [self.mu, self.log_std]:
            nn.init.orthogonal_(m.weight, gain=0.01); nn.init.zeros_(m.bias)

    def forward(self, obs):
        x = F.relu(self.fc1(self.ln(obs)))
        x = F.relu(self.fc2(x))
        return self.mu(x), self.log_std(x).clamp(-5.0, 2.0).exp()

    def sample(self, obs):
        mu, std = self(obs)
        raw = Normal(mu, std).rsample()
        act = torch.tanh(raw)
        lp  = (Normal(mu, std).log_prob(raw) -
               torch.log(1.0 - act.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return act, lp

    def deterministic(self, obs):
        mu, _ = self(obs); return torch.tanh(mu)


class Critic(nn.Module):
    """Twin Q-networks. No LayerNorm on critic (proven stable without it)."""
    def __init__(self):
        super().__init__()
        d = OBS_DIM + ACT_DIM
        self.q1_fc1 = nn.Linear(d, HIDDEN);  self.q1_fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.q1_out = nn.Linear(HIDDEN, 1)
        self.q2_fc1 = nn.Linear(d, HIDDEN);  self.q2_fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.q2_out = nn.Linear(HIDDEN, 1)
        self._init()

    def _init(self):
        for m in [self.q1_fc1, self.q1_fc2, self.q2_fc1, self.q2_fc2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2)); nn.init.zeros_(m.bias)
        for m in [self.q1_out, self.q2_out]:
            nn.init.orthogonal_(m.weight, gain=1.0); nn.init.zeros_(m.bias)

    def forward(self, obs, act):
        x  = torch.cat([obs, act], dim=-1)
        q1 = self.q1_out(F.relu(self.q1_fc2(F.relu(self.q1_fc1(x)))))
        q2 = self.q2_out(F.relu(self.q2_fc2(F.relu(self.q2_fc1(x)))))
        return q1, q2

    def q_min(self, obs, act):
        return torch.min(*self(obs, act))


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
        for p in self.critic_tgt.parameters():
            p.requires_grad = False
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha  = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, obs, deterministic=False):
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor.deterministic(t) if deterministic else self.actor.sample(t)[0]
        return a.squeeze(0).cpu().numpy()

    def update(self, replay):
        obs, act, rew, nobs, done = replay.sample(BATCH, self.device)
        alpha = self.log_alpha.exp().detach()

        # ── Critic ── [ABLATION] 1-step TD, MSE loss ─────────────────────────
        with torch.no_grad():
            na, nlp = self.actor.sample(nobs)
            tq1, tq2 = self.critic_tgt(nobs, na)
            target = rew + GAMMA * (1 - done) * (torch.min(tq1, tq2) - alpha * nlp)
        q1, q2  = self.critic(obs, act)
        loss_c  = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_critic.zero_grad(); loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        # ── Actor ─────────────────────────────────────────────────────────────
        for p in self.critic.parameters(): p.requires_grad = False
        new_act, new_lp = self.actor.sample(obs)
        loss_a = (alpha * new_lp - self.critic.q_min(obs, new_act)).mean()
        self.opt_actor.zero_grad(); loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
        for p in self.critic.parameters(): p.requires_grad = True

        # ── Temperature ── [FIXED] alpha increase when entropy < target ────────
        loss_t = (self.log_alpha * (-new_lp.detach() - TARGET_ENTROPY)).mean()
        self.opt_alpha.zero_grad(); loss_t.backward(); self.opt_alpha.step()

        # ── Soft update target ─────────────────────────────────────────────────
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
        if "opt_alpha" in d:
            self.opt_alpha.load_state_dict(d["opt_alpha"])


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 TRAINING NODE  (identical infrastructure across all ablation agents)
# ══════════════════════════════════════════════════════════════════════════════
class TrainNode(Node):
    # ── Node name and default run_id ──────────────────────────────────────────
    NODE_NAME   = "sac_baseline"
    DEFAULT_RUN = "sac_baseline_s42"

    def __init__(self):
        super().__init__(self.NODE_NAME)
        self.declare_parameter("run_id", self.DEFAULT_RUN)
        self.declare_parameter("fresh",  False)
        run_id      = self.get_parameter("run_id").value
        self._fresh = self.get_parameter("fresh").value

        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent  = SAC(self.device)
        self.replay = ReplayBuffer(BUFFER_CAP)

        # ── Paths ─────────────────────────────────────────────────────────────
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

        # ── Counters ──────────────────────────────────────────────────────────
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
        self._last_reset_time = time.time()

        # ── Resume / fresh ────────────────────────────────────────────────────
        if self._fresh:
            print(f"[{self.NODE_NAME}] FRESH START.", flush=True)
        else:
            self._resume()

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

        # ── ROS2 I/O ──────────────────────────────────────────────────────────
        self._act_pub = self.create_publisher(
            Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self._res_pub = self.create_publisher(
            Float32MultiArray, "/tb3_drl/training_result", 10)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/reset_obs",   self._on_reset, _LATCHED)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/step_result", self._on_step,  10)
        self.create_subscription(
            Int32, "/tb3_drl/curriculum_phase",        self._on_phase, _LATCHED)
        self.create_subscription(
            String, "/tb3_drl/goal", self._on_goal, _LATCHED)

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        mode = "EVAL" if EVAL_ONLY else f"TRAIN  warmup={WARMUP:,}"
        print(f"\n[{self.NODE_NAME}] {mode}  device={self.device}  "
              f"ep={self._ep}  seed={SEED}", flush=True)
        print("=" * 80, flush=True)

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def _save_ckpt(self, tag=None):
        label = tag or f"ep{self._ep:06d}"
        path  = os.path.join(self._model_dir, f"ckpt_{label}.pt")
        elapsed = self._elapsed_offset + (time.time() - self._t0)
        torch.save(dict(agent=self.agent.state_dict(), episode=self._ep,
                        total_steps=self._total_steps, updates=self._updates,
                        best_sr=self._best_sr, elapsed_s=elapsed,
                        outcomes=list(self._outcomes),
                        steps_hist=list(self._steps_hist)), path)
        torch.save(self.agent.actor.state_dict(),
                   os.path.join(self._model_dir, "actor_latest.pt"))
        # Save replay buffer alongside checkpoint
        buf_path = os.path.join(self._model_dir, f"ckpt_{label}_buf.npz")
        try:
            items = self.replay.to_list()
            if items:
                obs_arr  = np.array([it[0] for it in items], dtype=np.float32)
                act_arr  = np.array([it[1] for it in items], dtype=np.float32)
                rew_arr  = np.array([float(it[2]) for it in items], dtype=np.float32)
                nobs_arr = np.array([it[3] for it in items], dtype=np.float32)
                done_arr = np.array([float(it[4]) for it in items], dtype=np.float32)
                np.savez_compressed(buf_path, obs=obs_arr, act=act_arr, rew=rew_arr,
                                    nobs=nobs_arr, done=done_arr)
        except Exception as be:
            print(f"[{self.NODE_NAME}] Buffer save failed ({be})", flush=True)
        if tag is None:
            for old in sorted(glob.glob(
                    os.path.join(self._model_dir, "ckpt_ep*.pt")))[:-KEEP_CKPTS]:
                try:
                    os.remove(old)
                    old_buf = old.replace(".pt", "_buf.npz")
                    if os.path.exists(old_buf): os.remove(old_buf)
                except OSError: pass

    def _resume(self):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir, "ckpt_*.pt")))
        if not ckpts: return
        try:
            ckpt_latest = max(ckpts, key=os.path.getmtime)
            ckpt = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
            self.agent.load_state_dict(ckpt["agent"], self.device)
            self._ep             = ckpt.get("episode",     0)
            self._total_steps    = ckpt.get("total_steps", 0)
            self._updates        = ckpt.get("updates",     0)
            self._best_sr        = ckpt.get("best_sr",     0.0)
            self._elapsed_offset = ckpt.get("elapsed_s",   0.0)
            if "outcomes" in ckpt and ckpt["outcomes"]:
                self._outcomes.clear()
                self._outcomes.extend(ckpt["outcomes"])
            if "steps_hist" in ckpt and ckpt["steps_hist"]:
                self._steps_hist.clear()
                self._steps_hist.extend(ckpt["steps_hist"])
            sr_info = ""
            if self._outcomes:
                sr_info = f" | SR-100:{sum(self._outcomes)/len(self._outcomes)*100:.1f}%"
            fname = os.path.basename(ckpt_latest)
            print(f"[{self.NODE_NAME}] Resumed from {fname}: ep={self._ep}{sr_info}",
                  flush=True)
            # Re-seed on resume for consistency
            torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
            # Restore replay buffer if available
            buf_path = ckpt_latest.replace(".pt", "_buf.npz")
            if os.path.exists(buf_path):
                try:
                    d = np.load(buf_path)
                    items = list(zip(d["obs"], d["act"],
                                     d["rew"].tolist(), d["nobs"], d["done"].tolist()))
                    self.replay = ReplayBuffer.from_list(items, BUFFER_CAP)
                    print(f"[{self.NODE_NAME}] Replay buffer restored: "
                          f"{len(self.replay):,} transitions", flush=True)
                except Exception as be:
                    print(f"[{self.NODE_NAME}] Buffer restore failed ({be}) — starting fresh",
                          flush=True)
        except Exception as e:
            print(f"[{self.NODE_NAME}] Resume failed ({e}) — fresh start.", flush=True)

    def _shutdown(self, *_):
        print(f"\n[{self.NODE_NAME}] Shutdown — saving…", flush=True)
        self._save_ckpt(tag="shutdown")
        try: rclpy.shutdown()
        except Exception: pass

    # ── ROS callbacks ─────────────────────────────────────────────────────────
    def _on_phase(self, msg: Int32):
        self._cur_phase = msg.data

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

    def _on_reset(self, msg: Float32MultiArray):
        obs = np.array(msg.data[:OBS_DIM], dtype=np.float32)
        self._current_obs   = obs
        self._current_act   = None
        self._ep_reward     = 0.0
        self._ep_steps      = 0
        self._waiting_reset = False
        self._last_reset_time = time.time()
        self._send_action(obs)

    def _on_step(self, msg: Float32MultiArray):
        if self._waiting_reset or self._current_act is None: return
        nobs   = np.array(msg.data[:OBS_DIM], dtype=np.float32)
        reward = float(msg.data[OBS_DIM])
        done   = bool(msg.data[OBS_DIM + 1])
        info   = int(msg.data[OBS_DIM + 2])

        if done and self._is_impossible_terminal(info):
            return

        self._ep_reward   += reward
        self._ep_steps    += 1
        self._total_steps += 1

        # ── [ABLATION] Store 1-step transition directly (no n-step buffer) ───
        self.replay.add(self._current_obs, self._current_act,
                        reward, nobs, float(done))
        self._current_obs = nobs

        if (not EVAL_ONLY
                and self._total_steps > WARMUP
                and len(self.replay) >= BATCH):
            _, _, _, alpha_val = self.agent.update(self.replay)
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
                  f"SR={sr:5.1f}% | H_tgt={TARGET_ENTROPY:.2f} | "
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
                if self._ep % SAVE_EVERY == 0:
                    self._save_ckpt()
                if sr > self._best_sr and len(self._outcomes) >= 20:
                    self._best_sr = sr
                    self._save_ckpt(tag="best_sr")
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
