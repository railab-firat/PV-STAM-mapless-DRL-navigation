#!/usr/bin/env python3
"""
train_agent_dqn.py  Phase 1
================================================
DQN trainer ROS2 node.  Trains a discrete-action Q-network to navigate
a TurtleBot3 Waffle Pi to fixed goal positions in a static NIST arena.

Result: 98% rolling success rate over 5 000 episodes.

Architecture:
  QNet — 3-layer MLP: Linear(26→128) ReLU → Linear(128→128) ReLU → Linear(128→5)
  5 discrete actions: Stop / Forward / Forward+Left / Forward+Right / Backward

Key design choices:
  - ε-greedy exploration: ε decays 1.0 → 0.05 over 2 000 episodes
  - Experience replay buffer: 50 000 transitions (uniform sampling)
  - Target network: hard update every 1 000 steps
  - Batch size 64, Adam lr=1e-3, γ=0.99

USAGE:
    ros2 run tb3_drl_nav train_agent_dqn --ros-args -p run_id:="phase1_v1"

Auto-resumes from checkpoint if the same run_id is reused.
See docs/phase1_dqn/README.md for the full launch sequence and monitoring guide.
"""
import os, csv, math, time, random
from pathlib import Path
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def ensure_csv(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def read_progress_from_log(log_path: str):
    if not os.path.exists(log_path):
        return -1, 0
    last_ep = -1
    total_steps = 0
    try:
        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(float(row.get("episode", -1)))
                    st = int(float(row.get("steps", 0)))
                except Exception:
                    continue
                last_ep = max(last_ep, ep)
                total_steps += max(0, st)
    except Exception:
        return -1, 0
    return last_ep, total_steps


class DQNTrainer(Node):
    def __init__(self):
        super().__init__("tb3_drl_train_agent")

        self.act_pub = self.create_publisher(Int32, "/tb3_drl/action", 10)
        self.step_sub = self.create_subscription(Float32MultiArray, "/tb3_drl/step_result", self.on_step, 10)
        self.reset_sub = self.create_subscription(Float32MultiArray, "/tb3_drl/reset_obs", self.on_reset_obs, 10)
        self.reset_cli = self.create_client(Empty, "/tb3_drl/reset")

        self.last_reset = None
        self.last_step = None

        # Core hyperparams
        self.declare_parameter("episodes", 2600)
        self.declare_parameter("max_steps", 600)
        self.declare_parameter("gamma", 0.99)
        self.declare_parameter("lr", 1e-3)
        self.declare_parameter("batch", 64)
        self.declare_parameter("start_learn", 2000)
        self.declare_parameter("target_update", 1000)
        self.declare_parameter("buffer", 50000)

        # Modes
        self.declare_parameter("eval_mode", False)
        self.declare_parameter("eval_episodes", 200)

        # Manual override (optional)
        self.declare_parameter("epsilon_override", -1.0)

        # Experiment management (research-grade)
        self.declare_parameter("run_id", "run_" )
        self.declare_parameter("runs_dir", str(Path.home() / "tubitak_2209_ws" / "drl_runs"))

        # AUTO epsilon (this is what you want)
        self.declare_parameter("auto_epsilon", True)
        self.declare_parameter("eps_min", 0.01)
        self.declare_parameter("eps_max", 0.15)
        self.declare_parameter("eps_up", 0.01)       # increase when stuck
        self.declare_parameter("eps_down", 0.005)    # decrease when doing well
        self.declare_parameter("eps_window", 50)
        self.declare_parameter("success_hi", 0.85)   # if >= this -> decrease eps
        self.declare_parameter("success_lo", 0.60)   # if <= this -> increase eps

        self.declare_parameter("step_timeout", 5.0)

        self.episodes_total = int(self.get_parameter("episodes").value)
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.gamma = float(self.get_parameter("gamma").value)
        self.lr = float(self.get_parameter("lr").value)
        self.batch = int(self.get_parameter("batch").value)
        self.start_learn = int(self.get_parameter("start_learn").value)
        self.target_update = int(self.get_parameter("target_update").value)
        self.buffer_size = int(self.get_parameter("buffer").value)

        self.eval_mode = bool(self.get_parameter("eval_mode").value)
        self.eval_episodes = int(self.get_parameter("eval_episodes").value)
        self.epsilon_override = float(self.get_parameter("epsilon_override").value)

        self.auto_epsilon = bool(self.get_parameter("auto_epsilon").value)
        self.eps_min = float(self.get_parameter("eps_min").value)
        self.eps_max = float(self.get_parameter("eps_max").value)
        self.eps_up = float(self.get_parameter("eps_up").value)
        self.eps_down = float(self.get_parameter("eps_down").value)
        self.eps_window = int(self.get_parameter("eps_window").value)
        self.success_hi = float(self.get_parameter("success_hi").value)
        self.success_lo = float(self.get_parameter("success_lo").value)

        self.step_timeout = float(self.get_parameter("step_timeout").value)

        self.n_actions = 5
        self.obs_dim = None

        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        # ---- Per-run outputs (runs_dir/run_id) ----
        self.run_id = str(self.get_parameter("run_id").value)
        # If user passes empty run_id, keep it deterministic
        if not self.run_id or self.run_id.strip() == "":
            self.run_id = "run_"

        self.runs_dir = Path(self.get_parameter("runs_dir").value).expanduser()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = (self.runs_dir / self.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = str(self.run_dir / "dqn_model.pt")
        self.ckpt_path  = str(self.run_dir / "dqn_checkpoint.pt")
        self.train_log  = str(self.run_dir / "train_log.csv")
        self.eval_log   = str(self.run_dir / "eval_log.csv")
        ensure_csv(self.train_log, ["episode", "ep_reward", "steps", "success", "epsilon"])
        ensure_csv(self.eval_log,  ["episode", "ep_reward", "steps", "success", "epsilon", "goal_x", "goal_y"])

        last_ep, total_steps = read_progress_from_log(self.train_log)
        self.resume_episode = last_ep + 1
        self.global_step = total_steps

        self.rb = ReplayBuffer(self.buffer_size)
        self.q = None
        self.q_t = None
        self.opt = None

        # Adaptive epsilon state
        self.eps_current = min(max(0.05, self.eps_min), self.eps_max)  # start moderate
        self.recent_success = deque(maxlen=self.eps_window)

        mode = "EVAL" if self.eval_mode else "TRAIN"
        # ── Per-goal logging ──────────────────────────────────────
        self._current_goal_x = 0.0
        self._current_goal_y = 0.0
        self.create_subscription(
            PoseStamped, "/tb3_drl/goal",
            self._on_goal_msg, 10
        )
        self.get_logger().info(f"DQN {mode} ready. Resume episode={self.resume_episode}, approx global_step={self.global_step}.")

    def on_reset_obs(self, msg: Float32MultiArray):
        self.last_reset = msg.data

    def on_step(self, msg: Float32MultiArray):
        self.last_step = msg.data

    def spin_wait(self, cond_fn, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if cond_fn():
                return True
        return False

    def call_reset(self):
        if not self.reset_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Reset service /tb3_drl/reset not available.")
            return False
        self.last_reset = None
        fut = self.reset_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        return self.spin_wait(lambda: self.last_reset is not None, timeout=12.0)

    def parse_reset(self, data):
        ep_id = int(round(data[0]))
        obs = np.array(data[1:], dtype=np.float32)
        return ep_id, obs

    def parse_step(self, data):
        ep_id = int(round(data[0]))
        step_id = int(round(data[1]))
        reward = float(data[2])
        done = (float(data[3]) > 0.5)
        success = (float(data[4]) > 0.5)
        obs = np.array(data[5:], dtype=np.float32)
        return ep_id, step_id, reward, done, success, obs

    def maybe_init_nets(self, obs_dim):
        if self.q is not None:
            return
        self.obs_dim = int(obs_dim)
        self.q = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.q_t = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.opt = optim.Adam(self.q.parameters(), lr=self.lr)

        # Resume checkpoint if present
        if os.path.exists(self.ckpt_path):
            try:
                ckpt = torch.load(self.ckpt_path, map_location="cpu")
                self.q.load_state_dict(ckpt["q_state"])
                self.q_t.load_state_dict(ckpt["qt_state"])
                self.opt.load_state_dict(ckpt["opt_state"])
                self.resume_episode = int(ckpt.get("next_episode", self.resume_episode))
                self.global_step = int(ckpt.get("global_step", self.global_step))
                self.eps_current = float(ckpt.get("eps_current", self.eps_current))
                self.get_logger().info(f"Resumed FULL checkpoint from {self.ckpt_path} (ep={self.resume_episode}, step={self.global_step})")
            except Exception as e:
                self.get_logger().warn(f"Checkpoint load failed: {e}")
                self.q_t.load_state_dict(self.q.state_dict())
        else:
            self.q_t.load_state_dict(self.q.state_dict())

        self.q_t.eval()
        self.get_logger().info(f"Initialized DQN: obs_dim={self.obs_dim}, actions={self.n_actions}")

    def save_ckpt(self, next_episode):
        if self.q is None or self.opt is None:
            return
        ckpt = {
            "next_episode": int(next_episode),
            "global_step": int(self.global_step),
            "obs_dim": int(self.obs_dim),
            "q_state": self.q.state_dict(),
            "qt_state": self.q_t.state_dict(),
            "opt_state": self.opt.state_dict(),
            "eps_current": float(self.eps_current),
        }
        torch.save(ckpt, self.ckpt_path)
        torch.save(self.q.state_dict(), self.model_path)

    def epsilon(self):
        # EVAL is always greedy
        if self.eval_mode:
            return 0.0
        # manual override if user sets it
        if self.epsilon_override >= 0.0:
            return float(self.epsilon_override)
        # automatic epsilon
        if self.auto_epsilon:
            return float(self.eps_current)
        # fallback fixed value
        return 0.05

    def update_auto_epsilon(self):
        if not self.auto_epsilon:
            return
        if len(self.recent_success) < max(10, self.eps_window // 3):
            return
        sr = sum(self.recent_success) / len(self.recent_success)
        if sr >= self.success_hi:
            self.eps_current = max(self.eps_min, self.eps_current - self.eps_down)
        elif sr <= self.success_lo:
            self.eps_current = min(self.eps_max, self.eps_current + self.eps_up)
        else:
            # gentle decay toward eps_min
            self.eps_current = max(self.eps_min, self.eps_current - 0.2 * self.eps_down)

    def choose_action(self, obs, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.q(x).cpu().numpy()[0]
        return int(np.argmax(qv))

    def optimize(self):
        if len(self.rb) < self.batch:
            return 0.0
        s, a, r, s2, d = self.rb.sample(self.batch)
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            q_next = self.q_t(s2).max(dim=1, keepdim=True)[0]
            target = r + (1.0 - d) * self.gamma * q_next

        loss = nn.MSELoss()(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        return float(loss.item())

    def publish_action(self, a):
        self.last_step = None
        self.act_pub.publish(Int32(data=int(a)))

    def append_log(self, path, row):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    
    def _on_goal_msg(self, msg):
        """Store latest goal position for eval logging."""
        self._current_goal_x = round(msg.pose.position.x, 4)
        self._current_goal_y = round(msg.pose.position.y, 4)

    def run_eval(self):
        self.get_logger().info(f"Starting EVALUATION for {self.eval_episodes} episodes (epsilon={self.epsilon():.3f})")
        start_ep = self.resume_episode
        for k in range(self.eval_episodes):
            ep = start_ep + k
            if not self.call_reset():
                self.get_logger().error("Reset failed during eval.")
                break
            _, obs = self.parse_reset(self.last_reset)
            self.maybe_init_nets(obs.shape[0])

            ep_reward = 0.0
            success_flag = 0
            steps = 0

            for t in range(self.max_steps):
                a = self.choose_action(obs, 0.0)  # greedy
                self.publish_action(a)

                ok = self.spin_wait(lambda: self.last_step is not None, timeout=self.step_timeout)
                if not ok:
                    self.get_logger().warn("Eval: step_result timeout -> terminating episode as failure.")
                    steps = t + 1
                    success_flag = 0
                    break

                _, _, r, done, success, obs2 = self.parse_step(self.last_step)
                obs = obs2
                ep_reward += r
                steps = t + 1
                if done:
                    success_flag = 1 if success else 0
                    break

            self.append_log(self.eval_log, [ep, ep_reward, steps, success_flag, 0.0, self._current_goal_x, self._current_goal_y])
            self.get_logger().info(f"[EVAL] Ep {ep} | R {ep_reward:8.2f} | steps {steps:04d} | success {success_flag} | eps 0.000")

    def run_train(self):
        if self.episodes_total <= self.resume_episode:
            self.get_logger().warn(f"episodes_total={self.episodes_total} <= resume_episode={self.resume_episode}. Increase episodes.")
            return

        self.get_logger().info(f"Starting TRAIN from episode {self.resume_episode} to {self.episodes_total-1}...")

        for ep in range(self.resume_episode, self.episodes_total):
            if not self.call_reset():
                self.get_logger().error("Failed to get reset observation. Is environment node running?")
                break

            _, obs = self.parse_reset(self.last_reset)
            self.maybe_init_nets(obs.shape[0])

            ep_reward = 0.0
            success_flag = 0
            steps = 0

            for t in range(self.max_steps):
                eps = self.epsilon()
                a = self.choose_action(obs, eps)
                self.publish_action(a)

                ok = self.spin_wait(lambda: self.last_step is not None, timeout=self.step_timeout)
                if not ok:
                    self.get_logger().warn("Train: step_result timeout -> terminating episode as failure.")
                    steps = t + 1
                    success_flag = 0
                    break

                _, step_id, r, done, success, obs2 = self.parse_step(self.last_step)

                # Skip instant-fail artifacts (should be rare now)
                is_instant_fail = (step_id <= 1 and done and (r <= -99.0) and (not success))
                if not is_instant_fail:
                    self.rb.push(obs, a, r, obs2, 1.0 if done else 0.0)

                obs = obs2
                ep_reward += r
                steps = t + 1
                self.global_step += 1

                if self.global_step > self.start_learn:
                    _ = self.optimize()

                if self.global_step % self.target_update == 0:
                    self.q_t.load_state_dict(self.q.state_dict())

                if done:
                    success_flag = 1 if success else 0
                    break

            # update epsilon based on recent success
            self.recent_success.append(1 if success_flag else 0)
            self.update_auto_epsilon()

            if (ep % 25) == 0:
                self.save_ckpt(next_episode=ep + 1)

            eps_logged = self.epsilon()
            self.append_log(self.train_log, [ep, ep_reward, steps, success_flag, eps_logged])
            sr = (sum(self.recent_success)/len(self.recent_success))*100 if len(self.recent_success) > 0 else 0.0
            self.get_logger().info(f"Ep {ep} | R {ep_reward:8.2f} | steps {steps:04d} | success {success_flag} | eps {eps_logged:.3f} | roll_succ {sr:.1f}%")

        self.save_ckpt(next_episode=ep + 1)

    def run(self):
        if not self.call_reset():
            self.get_logger().error("Failed to get reset observation. Is environment node running?")
            return
        _, obs0 = self.parse_reset(self.last_reset)
        self.maybe_init_nets(obs0.shape[0])

        if self.eval_mode:
            self.run_eval()
        else:
            self.run_train()


def main():
    rclpy.init()
    node = DQNTrainer()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
