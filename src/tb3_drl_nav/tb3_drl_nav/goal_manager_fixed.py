#!/usr/bin/env python3
"""
goal_manager_dynamic.py  Phase 3  v8
==========================================================
Curriculum goal manager with 5 difficulty phases.

  Phase 1 — EASY    goals 1.0–2.0 m,  inner obstacles FROZEN
  Phase 2 — NEAR    goals 2.5–3.5 m,  inner obstacles FROZEN
  Phase 3 — ALL     goals 1.0–5.0 m,  inner obstacles FROZEN
  Phase 4 — HARD    goals 2.5–5.0 m,  6 long-centre obstacles ACTIVE
  Phase 5 — HARD    goals 2.5–5.0 m,  ALL 9 inner obstacles ACTIVE (final)

Advancement:
  Phase 1 → 2: rolling SR >= PHASE1_SR_THRESH over last PHASE1_WINDOW episodes
  Phase 2 → 3: rolling SR >= PHASE2_SR_THRESH over last PHASE2_WINDOW episodes
  Phase 3 → 4: rolling SR >= PHASE3_SR_THRESH over last PHASE3_WINDOW episodes
  Phase 4 → 5: rolling SR >= PHASE4_SR_THRESH over last PHASE4_WINDOW episodes
  (Thresholds loaded from config/phase3_ppo.yaml — default 65% / 60% / 70% / 65%)

Gazebo visualisation:
  A green pole + yellow sphere is spawned at the goal position and moved
  (not deleted/respawned) each episode for efficiency.

Publishes:  /tb3_drl/goal              (String "x.xxxx,y.yyyy")
            /tb3_drl/curriculum_phase  (Int32 — current phase 1/2/3/4/5)
Listens:    /tb3_drl/need_goal     (Bool)
            /tb3_drl/step_result   (Float32MultiArray)
"""
import collections
import os
import random
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool, Float32MultiArray, Int32
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState as SvcSetState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose

# Match the QoS used by environment_ppo for reset_obs (TRANSIENT_LOCAL)
# so we receive the signal even if we start slightly after the environment.
RESET_OBS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ── config loading ─────────────────────────────────────────────────────────────
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

# ── observation dimension (must match environment_ppo.py) ─────────────────────
OBS_DIM = _CFG.get("obs_dim", 54)

# =============================================================================
#  CURRICULUM GOAL POOLS
# =============================================================================

# Phase 1 — EASY: close cardinal/diagonal goals, no inner obstacles
GOALS_EASY = [
    ( 1.0,  0.0), (-1.0,  0.0),    # E / W  1.0m
    ( 0.0,  1.0), ( 0.0, -1.0),    # N / S  1.0m
    ( 1.2,  1.2), (-1.2,  1.2),    # NE/NW  1.7m
    ( 1.2, -0.8), (-1.2, -0.8),    # SE/SW  (was ±1.2,-1.2 — uc_clone/uc_clone_clone blocked them)
    ( 0.0,  1.5), ( 0.0, -2.0),    # N 1.5m (was 2.0 — uc_clone_1 at (0.57,2.05) blocked it), S 2.0m
    ( 1.5,  1.3), (-1.5,  1.3),    # far diag 2.0m  (was ±2.0,0 — 0.50m from dyn_obs_5/6)
]

# Phase 2 — NEAR: medium distance, obstacles still off (one change at a time)
# FIX: (+2.5,0.0) and (-2.5,0.0) were exactly on dyn_obs_6 and dyn_obs_5 spawn
#      positions — robot approached goal, hit obstacle, guaranteed collision.
#      Moved to (+2.5,+1.5) and (-2.5,+1.5) — same distance, clear of obstacle.
# FIX: (0.0,+3.5) was exactly on s10 spawn position (frozen in Phase 2/3) —
#      moved to (0.0,+2.8) — clear of s10, still a valid 2.8m N goal.
GOALS_NEAR = [
    ( 2.5,  1.5), (-2.5,  1.5),    # E/W   2.9m  (was 2.5,0 — on dyn_obs_6/5)
    ( 0.8,  2.2), (-0.8, -2.2),    # offset N/S 2.3m  (was 1.5,2.0 - blocked by dyn_obs_10/11)
    ( 2.2,  2.2), (-2.2, -2.2),    # NE/SW 3.1m  (was 1.8,1.8 - blocked by dyn_obs_10/11)
    ( 2.2, -2.2), ( 2.5, -1.5),    # SE 3.1m + S offset 2.9m (was 1.8,-1.8 - blocked by dyn_obs_12)
    ( 2.5,  2.0), (-2.5, -2.0),    # diag 3.2m  (was ±2.1,±2.1 — dyn_obs_10/11 at ±2.0,±2.5 blocked them)
    ( 0.8,  2.8), ( 0.0, -3.5),    # N 2.9m (was 1.5,2.5 — dyn_obs_10 too close; was 0,2.8 — near s8), S 3.5m
    ( 2.8,  1.8), (-2.8,  1.8),    # upper lane  3.3m
    ( 2.8, -1.8), (-2.8, -1.8),    # lower lane  3.3m
]

# Phase 3 — FAR: long corridors + corners (inner obstacles still FROZEN)
# FIX: (+3.5,-2.5) was within 0.11m of dyn_obs_4 at max travel (reaches y=-2.61)
#      moved to (+3.5,-1.8) — clear of obstacle's sweep range.
GOALS_FAR = [
    # N/S corridors — shifted x to avoid s10 at (0,3.5) sweeping x ±0.51m
    (-1.5,  4.0), ( 1.5,  4.0),               # 4.3m  (was ±1.0/0.0,4.0 — within s10 sweep)
    (-1.5, -4.0), ( 0.0, -4.0), ( 1.5, -4.0), # S corridor safe (no obstacle)
    # E/W corridors — shifted y to avoid s6(-3.5,0)/s7(3.5,0) sweeping y ±0.43m
    (-4.2,  1.5), ( 4.2,  1.5),               # 4.5m  (was ±4.2,0 — 0.70m from s6/s7)
    # Corner approaches — shifted inward to avoid corner obstacle sweep zones
    # dyn_obs_1(-3.5,3.5) x±0.76m, dyn_obs_2(3.5,3.5) x±0.83m
    # dyn_obs_3(-3.5,-3.5) y±0.70m, dyn_obs_4(3.5,-3.5) y±0.89m
    (-1.8,  3.5), ( 1.8,  3.5),               # 3.9m  (was ±2.2,3.5 — dyn_obs_1/2 sweep to within 0.25m)
    (-2.2, -3.5), ( 2.2, -3.5),               # 4.1m  (was ±3.0,-3.0 — 0.71m from corner obs)
    (-3.5,  1.8), ( 3.5,  1.8),               # 3.9m  (was ±3.5,2.5 — 1.0m from corner obs)
    (-3.5, -1.8), ( 3.5, -1.8),               # 3.9m  (was ±3.5,-2.5 — 1.0m from corner obs)
]

# ── phase advancement thresholds ──────────────────────────────────────────────
PHASE1_WINDOW    = _CFG.get("phase1_window",     100)
PHASE1_SR_THRESH = _CFG.get("phase1_sr_thresh", 0.65)  # 65% → Phase 2  (NEAR goals)
PHASE2_WINDOW    = _CFG.get("phase2_window",     100)
PHASE2_SR_THRESH = _CFG.get("phase2_sr_thresh", 0.65)  # 65% → Phase 3  (ALL goals, obstacles frozen)
PHASE3_WINDOW    = _CFG.get("phase3_window",     100)
PHASE3_SR_THRESH = _CFG.get("phase3_sr_thresh", 0.70)  # 70% → Phase 4  (6 long-centre obstacles)
PHASE4_WINDOW    = _CFG.get("phase4_window",     100)
PHASE4_SR_THRESH = _CFG.get("phase4_sr_thresh", 0.70)  # 70% → Phase 5  (all 9 inner obstacles)

# Phase 3: all distances with obstacles frozen — robot gets easy wins to build confidence
GOALS_ALL  = GOALS_EASY + GOALS_NEAR + GOALS_FAR
# Phase 4: only hard goals — removes easy SR inflation, forces genuine improvement
GOALS_HARD = GOALS_NEAR + GOALS_FAR

# ── module-level lookup tables (defined once, used everywhere) ─────────────────
PHASE_WINDOWS = {1: PHASE1_WINDOW, 2: PHASE2_WINDOW, 3: PHASE3_WINDOW,
                 4: PHASE4_WINDOW,  5: PHASE4_WINDOW}
PHASE_THRESH  = {1: PHASE1_SR_THRESH, 2: PHASE2_SR_THRESH,
                 3: PHASE3_SR_THRESH, 4: PHASE4_SR_THRESH}
PHASE_POOLS   = {1: GOALS_EASY, 2: GOALS_NEAR, 3: GOALS_ALL,
                 4: GOALS_HARD,  5: GOALS_HARD}   # Phase 4+5 use same hard goal pool

# =============================================================================
#  GAZEBO GOAL MARKER SDF  (v2 — phase-colored filled cylinder)
# =============================================================================
# Phase → colour mapping:
#   1: green (easy)  2: yellow (near)  3: orange (all)  4: red (hard)  5: purple (final)
PHASE_COLORS = {
    1: {"r": "0.0",  "g": "1.0",  "b": "0.2"},   # bright green
    2: {"r": "1.0",  "g": "0.9",  "b": "0.0"},   # yellow
    3: {"r": "1.0",  "g": "0.5",  "b": "0.0"},   # orange
    4: {"r": "1.0",  "g": "0.1",  "b": "0.1"},   # red
    5: {"r": "0.6",  "g": "0.0",  "b": "1.0"},   # purple
}

def _make_goal_sdf(phase: int = 1) -> str:
    """
    Goal marker v3 — highly visible landmark:
      • Tall glowing pole  : 0.05 m radius, 1.5 m tall
      • Ground ring        : 0.45 m radius (= GOAL_RADIUS), 0.02 m tall
      • Sphere on top      : 0.12 m radius at height 1.56 m
    All parts use full emissive so they glow even without lighting.
    Phase colour changes pole + sphere; ground ring stays white for clarity.
    """
    c = PHASE_COLORS.get(phase, PHASE_COLORS[1])
    r, g, b = c["r"], c["g"], c["b"]
    col      = f"{r} {g} {b} 1"
    col_dim  = f"{float(r)*0.6:.2f} {float(g)*0.6:.2f} {float(b)*0.6:.2f} 1"
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="goal_pole">
    <static>true</static>
    <link name="link">

      <!-- ── Tall glowing pole ── -->
      <visual name="pole">
        <pose>0 0 0.75 0 0 0</pose>
        <geometry><cylinder><radius>0.05</radius><length>1.5</length></cylinder></geometry>
        <material>
          <ambient>{col}</ambient>
          <diffuse>{col}</diffuse>
          <emissive>{col}</emissive>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>

      <!-- ── Sphere on top ── -->
      <visual name="sphere">
        <pose>0 0 1.62 0 0 0</pose>
        <geometry><sphere><radius>0.12</radius></sphere></geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>{col}</diffuse>
          <emissive>{col}</emissive>
        </material>
      </visual>

      <!-- ── Ground ring = goal success radius (0.45 m) ── -->
      <visual name="ground_ring">
        <pose>0 0 0.01 0 0 0</pose>
        <geometry><cylinder><radius>0.45</radius><length>0.02</length></cylinder></geometry>
        <material>
          <ambient>{col_dim}</ambient>
          <diffuse>{col_dim}</diffuse>
          <emissive>{col_dim}</emissive>
          <specular>0 0 0 1</specular>
        </material>
      </visual>

      <!-- ── Inner filled disc (solid floor marker) ── -->
      <visual name="disc">
        <pose>0 0 0.005 0 0 0</pose>
        <geometry><cylinder><radius>0.20</radius><length>0.01</length></cylinder></geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <emissive>{col}</emissive>
        </material>
      </visual>

    </link>
  </model>
</sdf>"""


class GoalManagerDynamic(Node):
    def __init__(self):
        super().__init__("goal_manager_dynamic")

        # ── curriculum state ──────────────────────────────────────────────────
        self.declare_parameter("start_phase", 0)   # 0 = auto from CSV
        force_phase = int(self.get_parameter("start_phase").value)

        self._phase         = 1
        self._pool          = list(GOALS_EASY)
        self._prev_idx      = -1
        self._result_window = collections.deque(maxlen=PHASE1_WINDOW)
        self._total_eps     = 0
        self._total_succs   = 0
        self._ep_open       = False

        # ── restore phase from saved file first (survives Gazebo crashes) ───────
        saved_phase = self._load_saved_phase()
        if saved_phase >= 1:
            self._phase = saved_phase
            self._pool  = list(PHASE_POOLS[self._phase])
            self._result_window = collections.deque(maxlen=PHASE_WINDOWS[self._phase])
            self.get_logger().info(
                f"[Goals] *** Restored Phase {self._phase} from saved file ***")

        # ── seed window from training CSV so restarts don't lose history ──────
        # Skip CSV seeding if start_phase is forced — prevents stale data corruption
        if force_phase < 1:
            self._seed_from_csv()

        # start_phase overrides everything if explicitly provided
        if force_phase >= 1:
            self._phase = max(1, min(5, force_phase))
            self._pool  = list(PHASE_POOLS[self._phase])
            self._result_window = collections.deque(maxlen=PHASE_WINDOWS[self._phase])
            self.get_logger().info(
                f"[Goals] start_phase={force_phase} → Phase {self._phase} forced")

        # ── Gazebo marker ─────────────────────────────────────────────────────
        # goal_pole is NOT in the world file — we spawn it fresh each run.
        self._marker_spawned      = False
        self._marker_phase        = self._phase   # track which phase colour is active
        self._pending_marker      = None   # (x, y) to place once services are ready
        self._confirmed_marker_pos = None  # last position confirmed by a successful move
        self._spawn_cli    = self.create_client(SpawnEntity, "/spawn_entity")
        self._delete_cli   = self.create_client(DeleteEntity, "/delete_entity")
        self._set_cli      = self.create_client(SvcSetState, "/set_entity_state")
        self._current_goal  = None   # always holds latest (gx, gy)
        self._marker_busy   = False  # True while an async move call is in flight
        # Force marker to current goal every 0.3s — fast recovery after resets
        self.create_timer(0.3, self._enforce_marker)

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self._goal_pub  = self.create_publisher(String, "/tb3_drl/goal", 10)
        # TRANSIENT_LOCAL = latched: any new subscriber immediately gets the last value
        _latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._phase_pub = self.create_publisher(Int32, "/tb3_drl/curriculum_phase", _latched)
        self.create_subscription(
            Bool,              "/tb3_drl/need_goal",   self._on_need_goal,    10)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/step_result", self._on_step_result,  10)
        # Listen to reset_obs: fires the moment each reset completes.
        # Use this to immediately re-place the goal marker after /reset_simulation
        # moves the pole back to its world-file spawn position.
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/reset_obs",   self._on_reset_obs,    RESET_OBS_QOS)

        # Keep-alive: republish goal every 1.5s and refresh Gazebo marker after resets
        self._last_goal_msg = None
        self.create_timer(1.5, self._republish)

        # Startup: send first goal 0.8s after boot (gives Gazebo time to start)
        self._startup_done = False
        self.create_timer(0.8, self._startup_goal)

        self._print_phase_banner()
        self.get_logger().info("[Goals] Waiting 0.8s before first goal…")
        # Publish current phase immediately so obstacle controller gets it on start
        self.create_timer(0.5, self._publish_phase_once)

    def _publish_phase_once(self):
        msg = Int32(); msg.data = self._phase
        self._phase_pub.publish(msg)

    # =========================================================================
    #  CSV SEED  (restores SR history on restart)
    # =========================================================================

    def _seed_from_csv(self):
        """Read the training CSV and pre-populate the result window + phase."""
        import csv, glob
        pattern = os.path.expanduser("~/tb3_drl_logs/phase3/*.csv")
        # Only consider episode logs (exclude *_updates.csv and other non-episode files)
        all_files = sorted(glob.glob(pattern))
        files = [f for f in all_files
                 if not os.path.basename(f).endswith("_updates.csv")
                 and "failure" not in os.path.basename(f)]
        if not files:
            return
        # Pick the largest episode log (most training data)
        csv_path = max(files, key=os.path.getsize)
        try:
            rows = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                if "goal_reached" not in (reader.fieldnames or []):
                    return  # not an episode log
                for row in reader:
                    rows.append(row)
            if not rows:
                return

            self._total_eps   = len(rows)
            self._total_succs = sum(int(r["goal_reached"]) for r in rows)

            # NOTE: phase is restored from saved file (above) — CSV seed only
            # fills the result_window history, it no longer sets the phase.
            # This prevents phase regression on Gazebo restart.
            phase = self._phase  # use already-restored phase

            # Fill result window with last N episodes
            req_window = PHASE_WINDOWS[self._phase]
            self._result_window = collections.deque(maxlen=req_window)
            for row in rows[-req_window:]:
                self._result_window.append(int(row["goal_reached"]))

            sr_now = (sum(self._result_window) / len(self._result_window)
                      if self._result_window else 0.0)
            self.get_logger().info(
                f"[Goals] Seeded from CSV: ep={self._total_eps}"
                f"  phase={self._phase}"
                f"  SR(last {len(self._result_window)})={sr_now*100:.0f}%")
        except Exception as e:
            self.get_logger().warn(f"[Goals] CSV seed failed: {e}")

    # =========================================================================
    #  CURRICULUM
    # =========================================================================

    def _print_phase_banner(self):
        names = {
            1: "EASY  (1–2 m,     no inner obs)",
            2: "NEAR  (2.5–3.5 m, no inner obs)",
            3: "ALL   (1–5 m,     no inner obs)",
            4: "HARD  (2.5–5 m,   6 long-centre obs ACTIVE)",
            5: "HARD  (2.5–5 m,   ALL 9 inner obs ACTIVE — final)",
        }
        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"  Goal Manager  Phase {self._phase} — {names[self._phase]}")
        self.get_logger().info(
            f"  {len(PHASE_POOLS[self._phase])} goal positions active")
        adv = {
            1: f"Advances → Phase 2 when SR >= {int(PHASE1_SR_THRESH*100)}% over {PHASE1_WINDOW} eps",
            2: f"Advances → Phase 3 when SR >= {int(PHASE2_SR_THRESH*100)}% over {PHASE2_WINDOW} eps",
            3: f"Advances → Phase 4 when SR >= {int(PHASE3_SR_THRESH*100)}% over {PHASE3_WINDOW} eps",
            4: f"Advances → Phase 5 when SR >= {int(PHASE4_SR_THRESH*100)}% over {PHASE4_WINDOW} eps",
            5: "Final phase — no further advancement.",
        }
        self.get_logger().info(f"  {adv[self._phase]}")
        self.get_logger().info("=" * 60)

    def _on_step_result(self, msg: Float32MultiArray):
        if len(msg.data) < OBS_DIM + 3:
            return
        done    = bool(msg.data[OBS_DIM + 1])
        info    = int(msg.data[OBS_DIM + 2])
        success = (info == 1)

        if done and self._ep_open:
            self._ep_open     = False
            self._total_eps   += 1
            self._total_succs += int(success)

            # Resize deque if phase changed since last episode
            req_window = PHASE_WINDOWS[self._phase]
            if self._result_window.maxlen != req_window:
                self._result_window = collections.deque(
                    self._result_window, maxlen=req_window)
            self._result_window.append(1 if success else 0)

            sr = (sum(self._result_window) / len(self._result_window)
                  if self._result_window else 0.0)

            if self._total_eps % 5 == 0:
                self.get_logger().info(
                    f"  [Phase {self._phase}] ep={self._total_eps}"
                    f"  sr={sr*100:.0f}%"
                    f"  succs={self._total_succs}/{self._total_eps}")

            thresh = PHASE_THRESH.get(self._phase, 1.0)
            req_w  = PHASE_WINDOWS.get(self._phase, PHASE3_WINDOW)
            if (self._phase < 5
                    and len(self._result_window) >= req_w
                    and sr >= thresh):
                self._advance_phase(sr)

    def _save_phase(self):
        """Persist current phase to file so restarts restore correct phase."""
        phase_file = os.path.expanduser("~/tb3_drl_logs/phase3/current_phase.txt")
        os.makedirs(os.path.dirname(phase_file), exist_ok=True)
        with open(phase_file, "w") as f:
            f.write(str(self._phase))

    def _load_saved_phase(self) -> int:
        """Return saved phase (1-5) or 0 if no file exists."""
        phase_file = os.path.expanduser("~/tb3_drl_logs/phase3/current_phase.txt")
        try:
            with open(phase_file) as f:
                return max(1, min(5, int(f.read().strip())))
        except Exception:
            return 0

    def _advance_phase(self, sr: float):
        self._phase += 1
        self._pool     = list(PHASE_POOLS[self._phase])
        self._prev_idx = -1
        self._result_window.clear()
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  PHASE {self._phase} UNLOCKED!")
        self.get_logger().info(
            f"  SR={sr*100:.0f}%  ep={self._total_eps}  succ={self._total_succs}")
        self._print_phase_banner()
        # Persist phase so Gazebo restarts resume from correct phase
        self._save_phase()
        # Notify obstacle controller to activate inner obstacles
        msg = Int32(); msg.data = self._phase
        self._phase_pub.publish(msg)
        self.get_logger().info(
            f"  Published curriculum_phase={self._phase} → obstacle controller will activate inner obstacles")
        # NOTE: marker respawn on phase transition intentionally SKIPPED.
        # DeleteEntity+SpawnEntity during a live episode corrupts Gazebo physics → SIGSEGV.
        # Marker colour is cosmetic only — training is unaffected.

    # =========================================================================
    #  GOAL PUBLISHING
    # =========================================================================

    def _on_need_goal(self, msg: Bool):
        if msg.data:
            self._ep_open = True
            self._publish_next_goal()

    def _publish_next_goal(self):
        pool  = self._pool
        cands = [i for i in range(len(pool)) if i != self._prev_idx]
        idx   = random.choice(cands)
        self._prev_idx = idx
        gx, gy = pool[idx]

        out      = String()
        out.data = f"{gx:.4f},{gy:.4f}"
        self._goal_pub.publish(out)
        self._last_goal_msg = out

        dist   = math.hypot(gx, gy)
        sr_str = ""
        if self._result_window:
            sr = sum(self._result_window) / len(self._result_window)
            sr_str = f"  [SR={sr*100:.0f}% w={len(self._result_window)}]"
        self.get_logger().info(
            f"[Goals Ph{self._phase}] Goal {idx+1}: "
            f"({gx:+.2f},{gy:+.2f}) {dist:.1f}m{sr_str}")

        self._pending_marker = (gx, gy)
        self._place_goal_marker(gx, gy)

    def _startup_goal(self):
        """Send first goal once on startup, 1.5s after node starts."""
        if not self._startup_done:
            self._startup_done = True
            self.get_logger().info("[Goals] Publishing startup goal…")
            self._ep_open = True
            self._publish_next_goal()

    def _on_reset_obs(self, msg: Float32MultiArray):
        """Fires the moment the environment finishes each reset (reset_obs published).
        /reset_simulation moves the robot back, but the goal_pole may also get
        reset to origin by Gazebo — so we force an immediate re-place.
        We also clear _marker_busy so we don't block on a stale in-flight call."""
        # Unblock any stale in-flight async call from before the reset
        self._marker_busy = False
        # Force a re-confirm — next enforce tick will move it even if same position
        self._confirmed_marker_pos = None
        if self._last_goal_msg is not None:
            try:
                gx, gy = [float(v) for v in self._last_goal_msg.data.split(",")]
                self._pending_marker = (gx, gy)
                # Try immediately — if service not ready, enforce timer will catch it
                if self._marker_spawned and self._set_cli.service_is_ready():
                    self._move_marker(gx, gy)
                elif not self._marker_spawned and self._spawn_cli.service_is_ready():
                    self._spawn_marker(gx, gy)
            except Exception:
                pass

    def _republish(self):
        """Keep-alive: re-send last goal string every 1.5s as a fallback."""
        if self._last_goal_msg is not None:
            self._goal_pub.publish(self._last_goal_msg)

    # =========================================================================
    #  GAZEBO MARKER
    # =========================================================================

    def _enforce_marker(self):
        """Called every 0.5s — spawns or moves marker to current goal.
        Handles both cases: marker not yet spawned (retry spawn) and marker
        moved back by /reset_simulation (retry move)."""
        if self._current_goal is None or self._marker_busy:
            return
        x, y = self._current_goal
        if not self._marker_spawned:
            if self._spawn_cli.service_is_ready():
                self._spawn_marker(x, y)
        else:
            # skip RPC if marker is already confirmed at this position
            if self._confirmed_marker_pos == (x, y):
                return
            if self._set_cli.service_is_ready():
                self._move_marker(x, y)

    def _place_goal_marker(self, x: float, y: float):
        """Immediate move — called when goal changes."""
        self._current_goal = (x, y)
        if not self._marker_spawned:
            if self._spawn_cli.service_is_ready():
                self._spawn_marker(x, y)
        else:
            if self._set_cli.service_is_ready() and not self._marker_busy:
                self._move_marker(x, y)

    def _spawn_marker(self, x: float, y: float):
        req                        = SpawnEntity.Request()
        req.name                   = "goal_pole"
        req.xml                    = _make_goal_sdf(self._phase)
        req.initial_pose           = Pose()
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = 0.0
        req.reference_frame        = "world"
        fut = self._spawn_cli.call_async(req)
        fut.add_done_callback(lambda f: self._on_spawn_done(f))

    def _on_spawn_done(self, future):
        try:
            res = future.result()
            already = (not res.success and "already exists" in res.status_message)
            if res.success or already:
                self._marker_spawned = True
                self._marker_phase  = self._phase
                if already:
                    self.get_logger().info(
                        "[Goals] Goal pole already in Gazebo — adopting it.")
                else:
                    self.get_logger().info(
                        f"[Goals] Goal pole spawned (phase {self._phase} colour).")
                # Move to the pending position
                if self._pending_marker is not None:
                    x, y = self._pending_marker
                    self._move_marker(x, y)
                    self._pending_marker = None
                elif self._current_goal is not None:
                    self._move_marker(*self._current_goal)
            else:
                self.get_logger().warn(
                    f"[Goals] Marker spawn failed: {res.status_message}")
        except Exception as e:
            self.get_logger().warn(f"[Goals] Marker spawn exception: {e}")

    def _respawn_marker_for_phase(self):
        """Delete and respawn the goal marker with the new phase colour.
        Called when phase advances — Gazebo SDF materials can't be changed at runtime."""
        if not self._marker_spawned or not self._delete_cli.service_is_ready():
            return
        self._marker_busy = True
        req = DeleteEntity.Request()
        req.name = "goal_pole"
        fut = self._delete_cli.call_async(req)
        fut.add_done_callback(lambda f: self._on_delete_done(f))

    def _on_delete_done(self, future):
        self._marker_busy   = False
        self._marker_spawned = False
        self._confirmed_marker_pos = None
        self.get_logger().info(
            f"[Goals] Old marker deleted — respawning with phase {self._phase} colour")
        # Respawn at the current goal position with new colour
        if self._current_goal is not None:
            x, y = self._current_goal
            if self._spawn_cli.service_is_ready():
                self._spawn_marker(x, y)

    def _move_marker(self, x: float, y: float):
        self._marker_busy = True
        req            = SvcSetState.Request()
        req.state      = EntityState()
        req.state.name = "goal_pole"
        p              = Pose()
        p.position.x   = float(x)
        p.position.y   = float(y)
        p.position.z   = 0.0
        p.orientation.w = 1.0
        req.state.pose  = p
        fut = self._set_cli.call_async(req)
        fut.add_done_callback(lambda f: self._on_move_done(f))

    def _on_move_done(self, future):
        self._marker_busy = False
        try:
            future.result()
            self._confirmed_marker_pos = self._current_goal  # record success
        except Exception as e:
            self._confirmed_marker_pos = None  # force retry on next enforce tick
            self.get_logger().warn(f"[Goals] Marker move failed ({e}) — will retry")


def main(args=None):
    rclpy.init(args=args)
    node = GoalManagerDynamic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
