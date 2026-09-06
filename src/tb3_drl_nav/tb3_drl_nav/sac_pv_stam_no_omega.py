#!/usr/bin/env python3
"""
sac_pv_stam_no_omega.py  —  Experiment 4: Ego-rotation ablation
================================================================
Identical to sac_stam.py (SAC-PV-STAM / v8) EXCEPT:
  - Nav vector drops nav_vel_ang (index 51 of raw obs)
  - Nav dims = 5  →  OBS_DIM = 24*3 + 5 = 77

Obs slice from raw 54-dim:
    scan  = raw[:24]              (current LiDAR)
    nav5  = [dn, an, nav_vel_lin, min_scan, min_approach]
           = raw[[48,49,50,52,53]]   (skip index 51 = nav_vel_ang)
Frame-stacked obs = [scan_t, scan_t-1, scan_t-2, nav5] = 77 dims
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

_LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "..", "config", "phase3_ppo.yaml")
try:
    import yaml
    with open(_CFG_PATH) as _f: _C = yaml.safe_load(_f) or {}
except Exception: _C = {}

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED         = int(os.environ.get("SEED", _C.get("seed", 42)))
RAW_OBS      = _C.get("obs_dim", 54)
ACT_DIM      = _C.get("act_dim", 2)
LIN_MIN      = _C.get("lin_min", 0.0)
LIN_MAX      = _C.get("lin_max", 0.26)
ANG_MAX      = _C.get("ang_max", 1.82)
N_SECTORS    = 24
N_FRAMES     = _C.get("sac_v8_frame_stack", 3)
NAV_DIMS     = 5                          # drop nav_vel_ang
OBS_DIM      = N_SECTORS * N_FRAMES + NAV_DIMS   # = 77

HIDDEN       = _C.get("sac_hidden", 256)
STAM_HEADS   = _C.get("sac_v8_stam_heads",   2)
STAM_D       = _C.get("sac_v8_stam_d_model", 16)
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
PER_ALPHA    = _C.get("sac_v8_per_alpha",      0.6)
PER_BETA0    = _C.get("sac_v8_per_beta_start", 0.4)
PER_BETA_F   = _C.get("sac_v8_per_beta_frames", 500_000)
N_STEP       = _C.get("sac_v8_nstep", 3)
ENT_START    = float(_C.get("sac_v8_entropy_start", 1.0))
ENT_END      = float(_C.get("sac_v8_entropy_end",   ACT_DIM))
ENT_ANNEAL   = int(_C.get("sac_v8_entropy_anneal_eps", 500))
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", _C.get("max_episodes", 0)))
EVAL_ONLY    = os.environ.get("EVAL_ONLY", "").lower() in ("true", "1", "yes")


def _build_no_omega_obs_raw(raw54: np.ndarray) -> tuple:
    """Returns (scan[24], nav5[5]) from raw 54-dim obs."""
    scan = raw54[:N_SECTORS]
    # nav5 = [dn, an, nav_vel_lin, min_scan, min_approach] — skip index 51
    nav5 = np.array([raw54[48], raw54[49], raw54[50], raw54[52], raw54[53]], dtype=np.float32)
    return scan, nav5


def _stam_split(obs):
    """Split 77-dim stacked obs → (B,24,3) frames + (B,5) nav."""
    frames = obs[:, :N_SECTORS * N_FRAMES].reshape(-1, N_SECTORS, N_FRAMES)
    nav    = obs[:, N_SECTORS * N_FRAMES:]
    return frames, nav


# ── Replay (uniform, same as baseline) ───────────────────────────────────────
class ReplayBuffer:
    def __init__(self, cap):
        self.obs  = np.zeros((cap, OBS_DIM), np.float32)
        self.act  = np.zeros((cap, ACT_DIM), np.float32)
        self.rew  = np.zeros((cap,), np.float32)
        self.nobs = np.zeros((cap, OBS_DIM), np.float32)
        self.done = np.zeros((cap,), np.float32)
        self.ptr = self.size = 0; self.cap = cap

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i]=o; self.act[i]=a; self.rew[i]=r; self.nobs[i]=no; self.done[i]=d
        self.ptr = (i+1)%self.cap; self.size = min(self.size+1, self.cap)

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, n)
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        return (t(self.obs[idx]), t(self.act[idx]), t(self.rew[idx]).unsqueeze(1),
                t(self.nobs[idx]), t(self.done[idx]).unsqueeze(1))


# ── STAM (identical to v8) ────────────────────────────────────────────────────
class MultiHeadScanAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_sectors = N_SECTORS
        self.n_heads   = STAM_HEADS
        self.d_k       = STAM_D // STAM_HEADS
        self.proj_in   = nn.Linear(N_FRAMES, STAM_D)
        self.pos_enc   = nn.Parameter(torch.randn(1, N_SECTORS, STAM_D) * 0.02)
        self.W_qkv     = nn.Linear(STAM_D, 3*STAM_D, bias=False)
        self.proj_out  = nn.Sequential(nn.Linear(STAM_D, STAM_D), nn.ReLU())
        self.compress  = nn.Linear(N_SECTORS*STAM_D, 48)

    def forward(self, x):
        B = x.size(0)
        h = self.proj_in(x) + self.pos_enc
        qkv = self.W_qkv(h).reshape(B, N_SECTORS, 3, self.n_heads, self.d_k).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = F.softmax(q@k.transpose(-2,-1)/math.sqrt(self.d_k), dim=-1)
        out  = (attn@v).transpose(1,2).reshape(B, N_SECTORS, STAM_D)
        return self.compress(self.proj_out(out).reshape(B,-1))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stam    = MultiHeadScanAttention()
        self.ln      = nn.LayerNorm(48 + NAV_DIMS)
        self.fc1     = nn.Linear(48 + NAV_DIMS, HIDDEN)
        self.fc2     = nn.Linear(HIDDEN, HIDDEN)
        self.mu      = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(HIDDEN, ACT_DIM)

    def forward(self, obs):
        frames, nav = _stam_split(obs)
        x = self.ln(torch.cat([self.stam(frames), nav], dim=-1))
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.mu(x), self.log_std(x).clamp(-5.0, 2.0).exp()

    def sample(self, obs):
        mu, std = self(obs)
        raw = Normal(mu, std).rsample(); act = torch.tanh(raw)
        lp  = (Normal(mu, std).log_prob(raw) - torch.log(1-act.pow(2)+1e-6)).sum(-1,keepdim=True)
        return act, lp

    def deterministic(self, obs):
        mu, _ = self(obs); return torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.stam1 = MultiHeadScanAttention()
        self.stam2 = MultiHeadScanAttention()
        d = 48 + NAV_DIMS + ACT_DIM
        self.q1 = nn.Sequential(nn.Linear(d,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,1))
        self.q2 = nn.Sequential(nn.Linear(d,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,1))

    def forward(self, obs, act):
        frames, nav = _stam_split(obs)
        x1 = torch.cat([self.stam1(frames), nav, act], dim=-1)
        x2 = torch.cat([self.stam2(frames), nav, act], dim=-1)
        return self.q1(x1), self.q2(x2)


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

    def select_action(self, obs, deterministic=False):
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor.deterministic(t) if deterministic else self.actor.sample(t)[0]
        return a.squeeze(0).cpu().numpy()

    def update(self, replay, target_h):
        obs, act, rew, nobs, done = replay.sample(BATCH, self.device)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            na, nlp = self.actor.sample(nobs)
            tq1, tq2 = self.critic_tgt(nobs, na)
            target = rew + GAMMA*(1-done)*(torch.min(tq1,tq2)-alpha*nlp)
        q1, q2 = self.critic(obs, act)
        loss_c = (q1-target).pow(2).mean()+(q2-target).pow(2).mean()
        self.opt_critic.zero_grad(); loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()
        for p in self.critic.parameters(): p.requires_grad=False
        new_act,new_lp = self.actor.sample(obs)
        q1p,q2p = self.critic(obs,new_act)
        loss_a = (alpha*new_lp-torch.min(q1p,q2p)).mean()
        self.opt_actor.zero_grad(); loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
        for p in self.critic.parameters(): p.requires_grad=True
        loss_t = (self.log_alpha*(-new_lp.detach()-target_h)).mean()
        self.opt_alpha.zero_grad(); loss_t.backward(); self.opt_alpha.step()
        with torch.no_grad():
            for p,pt in zip(self.critic.parameters(),self.critic_tgt.parameters()):
                pt.data.mul_(1-TAU).add_(TAU*p.data)
        return loss_a.item(), loss_c.item(), self.log_alpha.exp().item()


# ── ROS2 Training Node ────────────────────────────────────────────────────────
class TrainNoOmega(Node):
    NODE_NAME   = "sac_pv_stam_no_omega"
    DEFAULT_RUN = "sac_pv_stam_no_omega_s42"

    def __init__(self):
        super().__init__(self.NODE_NAME)
        self.declare_parameter("run_id", self.DEFAULT_RUN)
        self.declare_parameter("fresh",  False)
        run_id      = self.get_parameter("run_id").value
        self._fresh = self.get_parameter("fresh").value

        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent   = SAC(self.device)
        self.replay  = ReplayBuffer(BUFFER_CAP)
        self._scan_history = collections.deque(maxlen=N_FRAMES)

        log_dir   = os.path.expanduser(_C.get("log_dir", "~/tb3_drl_logs/phase3"))
        model_base = os.environ.get("SAC_MODEL_DIR", _C.get("sac_model_dir","~/tb3_drl_models/sac"))
        self._model_dir = os.path.expanduser(os.path.join(model_base, run_id))
        os.makedirs(log_dir, exist_ok=True); os.makedirs(self._model_dir, exist_ok=True)

        _eval_tag = os.environ.get("EVAL_TAG","").strip()
        _ev = f"_{_eval_tag}" if (EVAL_ONLY and _eval_tag) else ""
        self._log_path = os.path.join(log_dir, f"{run_id}{_ev}{'_eval' if EVAL_ONLY else ''}.csv")

        self._ep = self._total_steps = self._updates = 0
        self._ep_reward = self._ep_steps = 0
        self._best_sr = self._elapsed_offset = 0.0
        self._t0      = time.time()
        self._outcomes   = collections.deque(maxlen=100)
        self._steps_hist = collections.deque(maxlen=100)
        self._waiting_reset = True
        self._current_obs = self._current_act = None
        self._cur_phase   = 1
        self._target_h    = ENT_START

        if self._fresh:
            print(f"[NoOmega] FRESH START.", flush=True)
        else:
            self._resume(run_id)
        if EVAL_ONLY:
            self._ep=0; self._elapsed_offset=0.0; self._t0=time.time()
            self._outcomes.clear(); self._steps_hist.clear()

        with open(self._log_path, "w" if (self._ep==0 or EVAL_ONLY) else "a", newline="") as f:
            if self._ep == 0 or EVAL_ONLY:
                csv.writer(f).writerow(["episode","reward","steps","collision",
                    "goal_reached","sr_100","updates","alpha","phase","elapsed_s"])

        self._act_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self.create_subscription(Float32MultiArray, "/tb3_drl/reset_obs",   self._on_reset, _LATCHED)
        self.create_subscription(Float32MultiArray, "/tb3_drl/step_result", self._on_step,  10)
        self.create_subscription(Int32, "/tb3_drl/curriculum_phase",        self._on_phase, _LATCHED)
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        print(f"[NoOmega] device={self.device} ep={self._ep} seed={SEED} "
              f"obs_dim={OBS_DIM} EVAL={EVAL_ONLY}", flush=True)

    def _build_obs(self, raw):
        scan, nav5 = _build_no_omega_obs_raw(raw)
        self._scan_history.append(scan)
        while len(self._scan_history) < N_FRAMES:
            self._scan_history.append(scan)
        return np.concatenate(list(self._scan_history) + [nav5])

    def _entropy_target(self):
        if self._cur_phase == 7: return -float(ACT_DIM)
        if self._ep >= ENT_ANNEAL: return ENT_END
        return ENT_START + (ENT_END-ENT_START)*self._ep/ENT_ANNEAL

    def _save_ckpt(self, tag=None):
        label = tag or f"ep{self._ep:06d}"
        path  = os.path.join(self._model_dir, f"ckpt_{label}.pt")
        torch.save({"agent": {"actor": self.agent.actor.state_dict(),
                               "critic": self.agent.critic.state_dict(),
                               "critic_tgt": self.agent.critic_tgt.state_dict(),
                               "opt_actor": self.agent.opt_actor.state_dict(),
                               "opt_critic": self.agent.opt_critic.state_dict(),
                               "log_alpha": self.agent.log_alpha.detach().cpu(),
                               "opt_alpha": self.agent.opt_alpha.state_dict()},
                    "episode": self._ep, "total_steps": self._total_steps,
                    "updates": self._updates, "best_sr": self._best_sr,
                    "elapsed_s": self._elapsed_offset+(time.time()-self._t0)}, path)
        torch.save(self.agent.actor.state_dict(),
                   os.path.join(self._model_dir, "actor_latest.pt"))
        if tag is None:
            for old in sorted(glob.glob(os.path.join(self._model_dir,"ckpt_ep*.pt")))[:-KEEP_CKPTS]:
                try: os.remove(old)
                except: pass
        print(f"[NoOmega] Saved {os.path.basename(path)}", flush=True)

    def _resume(self, run_id):
        ckpts = sorted(glob.glob(os.path.join(self._model_dir,"ckpt_ep*.pt")))
        actor_latest = os.path.join(self._model_dir,"actor_latest.pt")
        if not ckpts:
            if EVAL_ONLY and os.path.exists(actor_latest):
                sd = torch.load(actor_latest, map_location="cpu", weights_only=False)
                self.agent.actor.load_state_dict(sd, strict=False)
                print(f"[NoOmega] Loaded actor_latest for eval", flush=True)
            return
        try:
            ckpt = torch.load(max(ckpts,key=os.path.getmtime), map_location="cpu", weights_only=False)
            d = ckpt["agent"]
            self.agent.actor.load_state_dict(d["actor"], strict=False)
            self.agent.critic.load_state_dict(d["critic"], strict=False)
            self.agent.critic_tgt.load_state_dict(d.get("critic_tgt",d["critic"]), strict=False)
            self.agent.opt_actor.load_state_dict(d["opt_actor"])
            self.agent.opt_critic.load_state_dict(d["opt_critic"])
            la = d["log_alpha"]
            self.agent.log_alpha = (la if isinstance(la,torch.Tensor) else torch.tensor([la])
                                   ).clone().to(self.device).requires_grad_(True)
            self.agent.opt_alpha = torch.optim.Adam([self.agent.log_alpha], lr=LR_ALPHA)
            if "opt_alpha" in d: self.agent.opt_alpha.load_state_dict(d["opt_alpha"])
            self._ep             = ckpt.get("episode",0)
            self._total_steps    = ckpt.get("total_steps",0)
            self._updates        = ckpt.get("updates",0)
            self._best_sr        = ckpt.get("best_sr",0.0)
            self._elapsed_offset = ckpt.get("elapsed_s",0.0)
            print(f"[NoOmega] Resumed ep={self._ep}", flush=True)
        except Exception as e:
            print(f"[NoOmega] Resume failed ({e})", flush=True)

    def _shutdown(self, *_):
        self._save_ckpt(tag="shutdown")
        try: rclpy.shutdown()
        except: pass

    def _on_phase(self, msg): self._cur_phase = msg.data

    def _on_reset(self, msg: Float32MultiArray):
        raw = np.array(msg.data[:RAW_OBS], dtype=np.float32)
        self._scan_history.clear()
        obs = self._build_obs(raw)
        self._current_obs   = obs
        self._current_act   = None
        self._ep_reward     = 0.0
        self._ep_steps      = 0
        self._waiting_reset = False
        self._send_action(obs)

    def _on_step(self, msg: Float32MultiArray):
        if self._waiting_reset or self._current_act is None: return
        raw    = np.array(msg.data[:RAW_OBS], dtype=np.float32)
        reward = float(msg.data[RAW_OBS])
        done   = bool(msg.data[RAW_OBS+1])
        info   = int(msg.data[RAW_OBS+2])
        # Spawn artifact (collision at step<30): don't count as episode, just reset.
        if done and info == 0:
            self._ep_reward = 0.0; self._ep_steps = 0
            self._waiting_reset = True; return
        nobs   = self._build_obs(raw)
        self._ep_reward += reward; self._ep_steps += 1; self._total_steps += 1
        self.replay.add(self._current_obs, self._current_act, reward, nobs, float(done))
        self._current_obs = nobs
        if not EVAL_ONLY and self._total_steps > WARMUP and self.replay.size >= BATCH:
            self._target_h = self._entropy_target()
            self.agent.update(self.replay, self._target_h)
            self._updates += 1
        if done:
            self._ep += 1
            goal = 1 if info==1 else 0; coll = 1 if info==2 else 0
            self._outcomes.append(goal); self._steps_hist.append(self._ep_steps)
            sr = sum(self._outcomes)/len(self._outcomes)*100
            ela = self._elapsed_offset+(time.time()-self._t0)
            tag = "GOAL" if goal else ("COLL" if coll else "TIME")
            print(f"Ep {self._ep:5d}|Ph{self._cur_phase}|{tag}|"
                  f"R={self._ep_reward:+8.1f}|SR={sr:5.1f}%|upd={self._updates}", flush=True)
            with open(self._log_path,"a",newline="") as f:
                csv.writer(f).writerow([self._ep,round(self._ep_reward,2),self._ep_steps,
                    coll,goal,round(sr,2),self._updates,
                    round(self.agent.log_alpha.exp().item(),4),self._cur_phase,round(ela,1)])
            if not EVAL_ONLY:
                if self._ep % SAVE_EVERY == 0: self._save_ckpt()
                if sr/100>self._best_sr: self._best_sr=sr/100; self._save_ckpt(tag="best_sr")
            if MAX_EPISODES > 0 and self._ep >= MAX_EPISODES:
                self._shutdown(); return
            self._waiting_reset = True
        else:
            self._send_action(nobs)

    def _send_action(self, obs):
        if EVAL_ONLY or self._total_steps >= WARMUP:
            act = self.agent.select_action(obs, deterministic=EVAL_ONLY)
        else:
            act = np.random.uniform(-1, 1, ACT_DIM).astype(np.float32)
        self._current_act = act
        lin = float(LIN_MIN + (act[0] + 1.0) / 2.0 * (LIN_MAX - LIN_MIN))
        ang = float(act[1] * ANG_MAX)
        msg = Float32MultiArray(); msg.data = [lin, ang]
        self._act_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrainNoOmega()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()
