#!/usr/bin/env python3
"""
canonical_eval.py
=============================================================================
Locked down canonical evaluation node.
Enforces deterministic evaluation (mean actions, alpha=0, no exploration).
Strictly maps variant and seed to verified checkpoint file paths.
Logs exact per-episode CSV results to prevent any silent fallback issues.
"""
import os
import sys
import glob
import math
import time
import csv
import collections
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Int32, String

# Sane default limits
LIN_MIN = 0.0
LIN_MAX = 0.26
ANG_MAX = 1.82
N_SECTORS = 24
N_FRAMES = 3
OBS_DIM_FS = N_SECTORS * N_FRAMES + 6 # 78
OBS_DIM_RAW = 54

# Add the parent package directory to sys.path to allow importing from other agent files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tb3_drl_nav.sac_mlp import Actor as ActorMLP
from tb3_drl_nav.sac_mlp_fs import ActorFS
from tb3_drl_nav.sac_stam import Actor as ActorStam
from tb3_drl_nav.sac_stam_huber import Actor as ActorStamHuber
from tb3_drl_nav.sac_r_stam import RecurrentActor
from tb3_drl_nav.sac_lstm import LSTMActor, _build_lstm_obs
from tb3_drl_nav.sac_pv_stam_no_omega import Actor as ActorNoOmega, _build_no_omega_obs_raw
N_FRAMES_NO_OMEGA = 3   # frame-stack depth for no_omega variant

# Hardcoded exact checkpoints map
CHECKPOINTS = {
    ("baseline", 42): "/home/anas/tb3_drl_models/sac/sac_baseline_s42/ckpt_shutdown.pt",
    ("baseline", 777): "/home/anas/tb3_drl_models/sac/sac_baseline_s777/ckpt_shutdown.pt",
    ("baseline", 123): "/home/anas/tb3_drl_models/sac/sac_baseline_s123/ckpt_shutdown.pt",
    
    ("mlp_fs", 42): "/home/anas/tb3_drl_models/sac/sac_mlp_fs_s42/ckpt_ep003950.pt",
    ("mlp_fs", 777): "/home/anas/tb3_drl_models/sac/sac_mlp_fs_s777/ckpt_ep004000.pt",
    ("mlp_fs", 123): "/home/anas/tb3_drl_models/sac/sac_mlp_fs_s123/ckpt_ep004100.pt",
    
    ("v8", 42): "/home/anas/tb3_drl_models/sac/sac_v8_s42/ckpt_ep001650.pt",
    ("v8", 777): "/home/anas/tb3_drl_models/sac/sac_v8_s777/ckpt_ep004000.pt",
    ("v8", 123): "/home/anas/tb3_drl_models/sac/sac_v8_s123/ckpt_ep002203.pt",
    
    ("v10", 42): "/home/anas/tb3_drl_models/sac/sac_v10_s42/ckpt_shutdown.pt",
    ("v10", 777): "/home/anas/tb3_drl_models/sac/sac_v10_s777/ckpt_ep004150.pt",
    ("v10", 123): "/home/anas/tb3_drl_models/sac/sac_v10_s123/ckpt_ep001448.pt",
    
    ("v11", 42):  "/home/anas/tb3_drl_models/sac/sac_v11_s42/ckpt_shutdown.pt",
    ("v11", 777): "/home/anas/tb3_drl_models/sac/sac_v11_s777/ckpt_shutdown.pt",
    ("v11", 123): "/home/anas/tb3_drl_models/sac/sac_v11_s123/ckpt_shutdown.pt",

    # ── Journal experiments ───────────────────────────────────────────────────
    ("lstm",     42):  "/home/anas/tb3_drl_models/sac/sac_lstm_s42/ckpt_shutdown.pt",
    ("lstm",     777): "/home/anas/tb3_drl_models/sac/sac_lstm_s777/ckpt_shutdown.pt",
    ("lstm",     123): "/home/anas/tb3_drl_models/sac/sac_lstm_s123/ckpt_shutdown.pt",
    ("no_omega", 42):  "/home/anas/tb3_drl_models/sac/sac_pv_stam_no_omega_s42/ckpt_shutdown.pt",

    # ── Task 8 reruns (2026-08) ───────────────────────────────────────────────
    # v10_matched : v10 with the critic width matched to v8 (256, not 384), which
    #               isolates the Huber-loss change from the width confound.
    #               Architecturally identical ACTOR to v10, so it reuses that class.
    # lstm_forced : LSTM baseline on the forced curriculum. Same LSTMActor as "lstm".
    ("v10_matched", 42):  "/home/anas/tb3_drl_models/sac/sac_v10_matched_s42/ckpt_shutdown.pt",
    ("v10_matched", 777): "/home/anas/tb3_drl_models/sac/sac_v10_matched_s777/ckpt_shutdown.pt",
    ("v10_matched", 123): "/home/anas/tb3_drl_models/sac/sac_v10_matched_s123/ckpt_shutdown.pt",

    ("lstm_forced", 42):  "/home/anas/tb3_drl_models/sac/sac_lstm_forced_s42/ckpt_shutdown.pt",
    ("lstm_forced", 777): "/home/anas/tb3_drl_models/sac/sac_lstm_forced_s777/ckpt_shutdown.pt",
    ("lstm_forced", 123): "/home/anas/tb3_drl_models/sac/sac_lstm_forced_s123/ckpt_shutdown.pt",
    ("finetune_benchc", 42): "/home/anas/tb3_drl_models/sac/sac_v11_s42_finetune_benchC/ckpt_shutdown.pt",
}

def _resolve_ckpt(path):
    """Return the checkpoint that actually represents the end of the run.

    Runs that finish naturally do not all use the same filename: sac_lstm writes
    ckpt_shutdown.pt from its shutdown handler, while sac_v10_matched only writes
    ckpt_ep######.pt. Resolving here keeps CHECKPOINTS declarative and stops a
    missing shutdown file from aborting an evaluation.

    A ckpt_shutdown.pt is NOT automatically preferred: a run that was interrupted
    and later resumed leaves a stale shutdown file behind, and evaluating it would
    silently score weights from thousands of episodes earlier. The newest file on
    disk is the correct one in both cases, so selection is by modification time.
    """
    import glob as _glob, re as _re
    d = os.path.dirname(path)
    cands = [f for f in _glob.glob(os.path.join(d, "ckpt_*.pt")) if os.path.exists(f)]
    if not cands:
        return path                      # let the caller raise a clear FileNotFoundError
    newest = max(cands, key=os.path.getmtime)
    eps = [f for f in cands if _re.search(r"ckpt_ep0*(\d+)\.pt$", f)]
    if eps:
        hi = max(eps, key=lambda f: int(_re.search(r"ckpt_ep0*(\d+)\.pt$", f).group(1)))
        # if the numbered checkpoint is newer than whatever `path` points at, use it
        if os.path.getmtime(hi) > os.path.getmtime(newest) - 1:
            newest = max((hi, newest), key=os.path.getmtime)
    return newest


_LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1)

class CanonicalEvalNode(Node):
    def __init__(self):
        super().__init__("canonical_eval")
        
        self.declare_parameter("variant", "v10")
        self.declare_parameter("seed", 42)
        self.declare_parameter("max_episodes", 100)
        self.declare_parameter("eval_tag", "benchmark_dqn_stage4")
        self.declare_parameter("log_dir", "/home/anas/tb3_drl_logs/canonical")
        
        self.variant = self.get_parameter("variant").value
        self.seed = self.get_parameter("seed").value
        self.max_episodes = self.get_parameter("max_episodes").value
        self.eval_tag = self.get_parameter("eval_tag").value
        self.log_dir = self.get_parameter("log_dir").value
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, f"sac_{self.variant}_s{self.seed}_{self.eval_tag}_eval.csv")
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the frozen, correct checkpoint path
        ckpt_key = (self.variant, self.seed)
        if ckpt_key not in CHECKPOINTS:
            raise ValueError(f"No checkpoint configured for variant={self.variant}, seed={self.seed}")
        self.ckpt_path = _resolve_ckpt(CHECKPOINTS[ckpt_key])

        # Verify file exists and print size
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint path does not exist: {self.ckpt_path}")
        
        file_size_mb = os.path.getsize(self.ckpt_path) / (1024 * 1024)
        print("="*80, flush=True)
        print(f"CANONICAL EVALUATION STARTUP", flush=True)
        print(f"  Variant       : {self.variant}", flush=True)
        print(f"  Seed          : {self.seed}", flush=True)
        print(f"  Max Episodes  : {self.max_episodes}", flush=True)
        print(f"  Eval Tag      : {self.eval_tag}", flush=True)
        print(f"  Checkpoint    : {self.ckpt_path}", flush=True)
        print(f"  File Size     : {file_size_mb:.2f} MB", flush=True)
        print(f"  Device        : {self.device}", flush=True)
        print(f"  CSV Output    : {self.csv_path}", flush=True)
        print("="*80, flush=True)
        
        # Initialize Actor
        self.actor = self._create_actor()
        self.actor.to(self.device)
        self.actor.eval()
        
        # Load weights strictly
        self._load_actor_weights()
        
        # Episode stats tracker
        self.episode_records = []
        
        # Determine starting episode by counting valid canonical eval rows in CSV.
        # A valid row has 6 columns where all are numeric, steps > 0, and outcome sums ≤ 1.
        # This guards against corrupted files written by other nodes (e.g. tagd_goals).
        self._ep = 0
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            try:
                seen_eps = set()
                with open(self.csv_path, "r", newline="") as f:
                    for row in csv.reader(f):
                        try:
                            if len(row) != 6 or row[0] == "episode":
                                continue
                            ep  = int(row[0])
                            gr  = int(row[1]); cr  = int(row[2]); to_ = int(row[3])
                            st  = int(row[4]); _   = float(row[5])
                            if gr + cr + to_ <= 1 and 0 < st <= 600:
                                seen_eps.add(ep)
                        except Exception:
                            continue
                self._ep = len(seen_eps)
            except Exception:
                self._ep = 0
            
        self._ep_steps = 0
        self._ep_reward = 0.0
        self._waiting_reset = True
        self._current_obs = None
        self._current_act = None
        
        # For temporal frame-stacking (v8/v10/mlp_fs and no_omega)
        self._scan_history    = collections.deque(maxlen=N_FRAMES)
        self._no_omega_history = collections.deque(maxlen=N_FRAMES_NO_OMEGA)

        # For recurrent state: v11 uses GRU tensor; lstm uses (h,c) tuple
        self._actor_h = None   # GRU hidden state (v11) or LSTM (h,c) tuple
        
        # CSV writing headers if file is new
        if self._ep == 0:
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["episode", "goal_reached", "collision", "timeout", "steps", "cumulative_reward"])
            
        # ROS2 Pub/Sub
        self._act_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/action_continuous", 10)
        self._res_pub = self.create_publisher(Float32MultiArray, "/tb3_drl/training_result", 10)
        
        self.create_subscription(Float32MultiArray, "/tb3_drl/reset_obs", self._on_reset, _LATCHED)
        self.create_subscription(Float32MultiArray, "/tb3_drl/step_result", self._on_step, 10)
        self.create_subscription(Int32, "/tb3_drl/curriculum_phase", lambda msg: None, _LATCHED) # Ignore phase changes during eval
        self.create_subscription(String, "/tb3_drl/goal", lambda msg: None, _LATCHED) # Ignore goal positions during eval
        
    def _create_actor(self):
        if self.variant == "baseline":
            return ActorMLP()
        elif self.variant == "mlp_fs":
            return ActorFS()
        elif self.variant == "v8":
            return ActorStam()
        elif self.variant in ("v10", "v10_matched"):
            return ActorStamHuber()
        elif self.variant in ("v11", "finetune_benchc"):
            return RecurrentActor()
        elif self.variant in ("lstm", "lstm_forced"):
            return LSTMActor()
        elif self.variant == "no_omega":
            return ActorNoOmega()
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
            
    def _load_actor_weights(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        if "agent" in ckpt:
            state_dict = ckpt["agent"]["actor"]
            print(f"Loaded actor state dict from agent key (epoch: {ckpt.get('episode', 'N/A')})", flush=True)
        elif "actor" in ckpt:
            state_dict = ckpt["actor"]
            print("Loaded actor state dict from actor key", flush=True)
        else:
            state_dict = ckpt
            print("Loaded raw actor state dict keys directly", flush=True)
            
        # Ensure strict matching of keys
        self.actor.load_state_dict(state_dict, strict=True)
        print("Checkpoint weight keys matched successfully and loaded strictly.", flush=True)
        
    def _build_obs(self, raw):
        """Build observation for the correct variant."""
        raw = raw[:OBS_DIM_RAW]
        if self.variant in ("mlp_fs", "v8", "v10", "v10_matched"):
            # 3-frame LiDAR stack + 6 nav = 78 dims
            scan = raw[:N_SECTORS]
            nav  = raw[2 * N_SECTORS:]
            self._scan_history.append(scan)
            while len(self._scan_history) < N_FRAMES:
                self._scan_history.append(scan)
            return np.concatenate(list(self._scan_history) + [nav])
        elif self.variant in ("lstm", "lstm_forced"):
            # 30-dim: lidar[24] + nav6[48:54] (no scan_vel)
            return _build_lstm_obs(raw)
        elif self.variant == "no_omega":
            # 3-frame stack + 5 nav (no nav_vel_ang) = 77 dims
            scan, nav5 = _build_no_omega_obs_raw(raw)
            self._no_omega_history.append(scan)
            while len(self._no_omega_history) < N_FRAMES_NO_OMEGA:
                self._no_omega_history.append(scan)
            return np.concatenate(list(self._no_omega_history) + [nav5])
        else:
            # baseline & v11: raw 54-dim
            return raw
            
    def _select_action(self, obs):
        with torch.no_grad():
            if self.variant in ("v11", "finetune_benchc"):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                mu, _, self._actor_h = self.actor(obs_t, self._actor_h)
                act = torch.tanh(mu).squeeze(0).squeeze(0).cpu().numpy()
            elif self.variant in ("lstm", "lstm_forced"):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                mu, _, self._actor_h = self.actor(obs_t, self._actor_h)  # actor_h = (h, c) tuple
                act = torch.tanh(mu).squeeze(0).squeeze(0).cpu().numpy()
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                act = self.actor.deterministic(obs_t).squeeze(0).cpu().numpy()
        return act
        
    def _is_impossible_terminal(self, info):
        elapsed = time.time() - self._last_reset_time
        next_step = self._ep_steps + 1
        if info == 0:
            return True
        if elapsed < 0.10:  # RESET_SETTLE_S
            return True
        if info == 8 and next_step < 60:  # MIN_STUCK_STEPS
            return True
        if info == 4 and next_step < 500:  # MIN_TIMEOUT_STEPS
            return True
        return False

    def _on_reset(self, msg: Float32MultiArray):
        raw = np.array(msg.data, dtype=np.float32)
        self._scan_history.clear()
        self._actor_h = None # Reset recurrent state at boundary
        
        obs = self._build_obs(raw)
        self._current_obs = obs
        self._current_act = None
        self._ep_steps = 0
        self._ep_reward = 0.0
        self._waiting_reset = False
        self._last_reset_time = time.time()
        
        self._send_action(obs)
        
    def _on_step(self, msg: Float32MultiArray):
        if self._waiting_reset:
            return
            
        raw = np.array(msg.data, dtype=np.float32)
        
        # Slice depending on observation shape of variant
        if self.variant in ("baseline", "v11"):
            reward = float(msg.data[OBS_DIM_RAW])
            done = bool(msg.data[OBS_DIM_RAW + 1])
            info = int(msg.data[OBS_DIM_RAW + 2])
        else: # mlp_fs, v8, v10
            reward = float(msg.data[2 * N_SECTORS + 6])
            done = bool(msg.data[2 * N_SECTORS + 7])
            info = int(msg.data[2 * N_SECTORS + 8])
            
        if done and self._is_impossible_terminal(info):
            return

        obs = self._build_obs(raw)
        self._ep_reward += reward
        self._ep_steps += 1
        
        if done:
            self._ep += 1
            goal_reached = 1 if info == 1 else 0
            collision = 1 if info == 2 else 0
            timeout = 1 if info == 4 else 0
            
            # Print episode completion status
            outcome = "GOAL" if goal_reached else ("COLLISION" if collision else "TIMEOUT")
            print(f"Ep {self._ep:3d}/{self.max_episodes} | {outcome:<9} | Reward: {self._ep_reward:+8.2f} | Steps: {self._ep_steps:4d}", flush=True)
            
            # Record results
            self.episode_records.append({
                "episode": self._ep,
                "goal_reached": goal_reached,
                "collision": collision,
                "timeout": timeout,
                "steps": self._ep_steps,
                "cumulative_reward": self._ep_reward
            })
            
            # Write to CSV immediately
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([self._ep, goal_reached, collision, timeout, self._ep_steps, round(self._ep_reward, 2)])
                
            # Publish result to trigger goal manager reset
            res_msg = Float32MultiArray()
            res_msg.data = [float(self._ep), float(self._ep_reward), float(self._ep_steps), float(collision), float(goal_reached), float(info)]
            self._res_pub.publish(res_msg)
            
            self._waiting_reset = True
            
            # Terminate if max episodes reached
            if self._ep >= self.max_episodes:
                self._print_summary_and_exit()
        else:
            self._current_obs = obs
            self._send_action(obs)
            
    def _send_action(self, obs):
        act = self._select_action(obs)
        self._current_act = act
        lin = float(LIN_MIN + (act[0] + 1.0) / 2.0 * (LIN_MAX - LIN_MIN))
        ang = float(act[1] * ANG_MAX)
        msg = Float32MultiArray()
        msg.data = [lin, ang]
        self._act_pub.publish(msg)
        
    def _print_summary_and_exit(self):
        eps = len(self.episode_records)
        if eps == 0:
            print("No episodes completed.", flush=True)
            sys.exit(0)
            
        sr = sum(r["goal_reached"] for r in self.episode_records) / eps * 100
        cr = sum(r["collision"] for r in self.episode_records) / eps * 100
        tor = sum(r["timeout"] for r in self.episode_records) / eps * 100
        mean_steps = sum(r["steps"] for r in self.episode_records) / eps
        mean_reward = sum(r["cumulative_reward"] for r in self.episode_records) / eps
        
        print("\n" + "="*80, flush=True)
        print(f"CANONICAL EVALUATION SUMMARY (overall SR={sr:.1f}%)", flush=True)
        print(f"  Episodes Run     : {eps}", flush=True)
        print(f"  Success Rate     : {sr:.2f}%", flush=True)
        print(f"  Collision Rate   : {cr:.2f}%", flush=True)
        print(f"  Timeout Rate     : {tor:.2f}%", flush=True)
        print(f"  Mean Steps       : {mean_steps:.2f}", flush=True)
        print(f"  Mean Reward      : {mean_reward:+.2f}", flush=True)
        print("="*80 + "\n", flush=True)
        
        print("MAX_EPISODES reached", flush=True)
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = CanonicalEvalNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
