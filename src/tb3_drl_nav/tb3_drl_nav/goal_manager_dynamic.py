#!/usr/bin/env python3
"""
goal_manager_dynamic.py  Phase 3  v9 (REDESIGNED)
=====================================================================
Curriculum goal manager with 5 difficulty phases - GRADUAL INTRODUCTION.

DESIGN PRINCIPLES (from failed runs analysis):
  1. Never add more than 4 new goals per phase
  2. Heavy weighting keeps easy goals >50% of samples
  3. High thresholds ensure mastery before advancement
  4. Split FAR goals into MID (3.8m) and DEEP (4.5m)

PHASE STRUCTURE:
  Phase 1 — CORE   : 8 goals,  1.2–2.1m,  center hub only
  Phase 2 — EXPAND : +8 goals, 2.5–3.0m,  doorway navigation
  Phase 3 — REACH  : +4 goals, 3.8m,      corner room ENTRANCES
  Phase 4 — FULL   : +4 goals, 4.5m,      corner room DEPTHS
  Phase 5 — DYNAMIC: obstacles active (future)

ADVANCEMENT THRESHOLDS (from config/phase3_ppo.yaml):
  Phase 1 → 2: 70% SR over 50 episodes
  Phase 2 → 3: 70% SR over 50 episodes
  Phase 3 → 4: 65% SR over 50 episodes
  Phase 4 → 5: 75% SR over 100 episodes (must master ALL goals before dynamics!)
  Phase 5 → 6: 60% SR over 50 episodes
  Phase 6 → 7: 55% SR over 50 episodes

Publishes:  /tb3_drl/goal              (String "x.xxxx,y.yyyy")
            /tb3_drl/curriculum_phase  (Int32 — current phase 1/2/3/4/5)
Listens:    /tb3_drl/need_goal     (Bool)
            /tb3_drl/step_result   (Float32MultiArray)
"""
import collections
import os
import random
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool, Float32MultiArray, Int32
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState as SvcSetState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose

RESET_OBS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Robot spawn parameters (must match environment_ppo.py)
HOME_X = -2.0
HOME_Y = -0.5
SPAWN_MARGIN = 0.4    # Restored: ensure goal isn't too close to spawn point

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


def _cfg_get(path, default, legacy_key=None):
    cur = _CFG
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            cur = None
            break
        cur = cur[key]
    if cur is not None:
        return cur
    if legacy_key is not None:
        return _CFG.get(legacy_key, default)
    return default

OBS_DIM = _CFG.get("obs_dim", 54)
RESET_SETTLE_S = float(_cfg_get(("goal_manager", "terminal_filters", "reset_settle_s"), 2.0))   # 2s: filter post-reset phantom collisions
MIN_STUCK_SECONDS = float(_cfg_get(("goal_manager", "terminal_filters", "min_stuck_seconds"), 8.5))
MIN_TIMEOUT_SECONDS = float(_cfg_get(("goal_manager", "terminal_filters", "min_timeout_seconds"), 70.0))


def _run_key():
    raw = os.environ.get("RUN_ID", "").strip()
    if not raw:
        return ""
    safe = []
    for ch in raw:
        safe.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(safe)


def _phase_state_paths():
    log_dir = os.path.expanduser("~/tb3_drl_logs/phase3")
    run_key = _run_key()
    if run_key:
        return (
            os.path.join(log_dir, f"current_phase_{run_key}.txt"),
            os.path.join(log_dir, f"curriculum_state_{run_key}.json"),
        )
    return (
        os.path.join(log_dir, "current_phase.txt"),
        os.path.join(log_dir, "curriculum_state.json"),
    )


def _run_log_path():
    run_key = _run_key()
    if not run_key:
        return ""
    return os.path.expanduser(f"~/tb3_drl_logs/phase3/{run_key}.csv")

GOAL_RADIUS = float(_cfg_get(("goal_manager", "safety", "goal_radius"), 0.30))
SAFE_MARGIN = 0.10    # Restored: moderate safety buffer to prevent spawning inside objects
V_PEAK = float(_cfg_get(("goal_manager", "safety", "v_peak"), 0.18))

def _amp(period_s: float) -> float:
    return V_PEAK * period_s / (2.0 * math.pi)

def _dist_point_to_segment(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 == 0.0:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)

# ═══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE DEFINITIONS (for safety filtering)
# ═══════════════════════════════════════════════════════════════════════════════

_OBS_PERIMETER = [
    ("dyn_obs_1", -3.5,  3.5, "x", 12.0, 0.15),
    ("dyn_obs_2",  3.5,  3.5, "x", 13.0, 0.15),
    ("dyn_obs_3", -3.5, -3.5, "y", 11.0, 0.15),
    ("dyn_obs_4",  3.5, -3.5, "y", 14.0, 0.15),
    ("dyn_obs_5", -2.5,  0.0, "y", 10.0, 0.15),
    ("dyn_obs_6",  2.5,  0.0, "y", 10.0, 0.15),
]

_OBS_INNER = [
    ("s1",        -1.5,  0.0, "y", 20.0, 0.13),
    ("s2",         1.5,  0.0, "y", 20.0, 0.13),
    ("s6",        -3.5,  0.0, "y", 15.0, 0.13),
    ("s7",         3.5,  0.0, "y", 15.0, 0.13),
    ("s8",        -0.8,  1.8, "x", 15.0, 0.11),
    ("s10",        0.0,  3.5, "x", 18.0, 0.13),
    ("uc_clone",       -0.8, -1.6, "y",  8.0, 0.188),
    ("uc_clone_1",      0.6,  2.0, "x",  9.0, 0.188),
    ("uc_clone_clone",  0.9, -1.4, "y",  8.0, 0.188),
]

_OBS_STATIC = [
    ("dyn_obs_9",  -2.0,  2.5, 0.15),
    ("dyn_obs_10",  2.0,  2.5, 0.15),
    ("dyn_obs_11", -2.0, -2.5, 0.15),
    ("dyn_obs_12",  2.0, -2.5, 0.15),
    ("stA",        -2.5,  0.7, 0.20),
    ("stB",         2.5, -0.7, 0.20),
    ("stC",        -1.0, -0.8, 0.20),
    ("stD",         1.0,  0.8, 0.20),
]

def _check_moving_path(x, y, obs_list):
    for _, ox, oy, axis, period, r in obs_list:
        a = _amp(period)
        if axis == "x":
            ax, ay, bx, by = ox - a, oy, ox + a, oy
        else:
            ax, ay, bx, by = ox, oy - a, ox, oy + a
        d = _dist_point_to_segment(x, y, ax, ay, bx, by)
        if d < (GOAL_RADIUS + r + SAFE_MARGIN):
            return False
    return True

def _check_static(x, y):
    for _, ox, oy, r in _OBS_STATIC:
        if math.hypot(x - ox, y - oy) < (GOAL_RADIUS + r + SAFE_MARGIN + 0.05):
            return False
    # Spawn Protection: ensure goal is not too close to the robot's starting position
    if math.hypot(x - HOME_X, y - HOME_Y) < SPAWN_MARGIN:
        return False
    return True

def _is_safe_early(x, y):
    """Phase 1-4: perimeter moving, inner obstacles FROZEN."""
    if not _check_moving_path(x, y, _OBS_PERIMETER):
        return False
    if not _check_static(x, y):
        return False
    for _, ox, oy, _, _, r in _OBS_INNER:
        if math.hypot(x - ox, y - oy) < (GOAL_RADIUS + r + 0.10):
            return False
    return True

def _is_safe_late(x, y):
    """Phase 5: all obstacles active with full sweep paths."""
    if not _check_moving_path(x, y, _OBS_PERIMETER + _OBS_INNER):
        return False
    if not _check_static(x, y):
        return False
    return True

# =============================================================================
#  CURRICULUM GOAL POOLS — REDESIGNED FOR GRADUAL INTRODUCTION
# =============================================================================
# Key insight: Previous failures came from adding too many hard goals at once.
# Solution: Split FAR goals into MID (entrances) and DEEP (depths).
# =============================================================================

# ── Phase 1: CORE — Center hub (1.2–2.1m) ────────────────────────────────────
#    8 goals, all within clear center area. Robot learns basic navigation.
_GOALS_CORE_RAW = [
    ( 0.0,  1.2),  ( 0.0, -1.2),   # N / S close
    ( 0.0,  1.9),  ( 0.0, -1.9),   # N / S mid (Clear of s8/s10/etc)
    ( 1.4,  1.4),  (-1.4,  1.4),   # NE / NW (Clears stD(1, 0.8))
    ( 1.4, -1.4),                  # SE (SW G08 (-1.4, -1.4) deleted - structural block)
]

# ── Phase 2: EXPAND — Doorway navigation (2.5–3.0m) ──────────────────────────
_GOALS_EXPAND_RAW = [
    ( 0.0,  2.5),  ( 0.0, -2.5),   # N / S corridors start
    ( 2.5,  0.0),  (-2.5,  0.0),   # E / W corridors start
    ( 3.2,  1.5),  (-3.2,  1.5),   # Safe from stA/stB pillars
                   (-3.2, -1.5),   # (G15 (3.2, -1.5) deleted - structural block)
]

# ── Phase 3: REACH — Corner room entrances (3.8m diagonal) ───────────────────
_GOALS_MID_RAW = [
    (-2.8,  2.8),   # NW entrance (Clears dyn_obs_9(-2, 2.5))
    ( 2.8,  2.8),   # NE entrance (Clears dyn_obs_10(2, 2.5))
    (-2.5, -2.5),   # SW entrance (G19 relocated from -2.8, -2.8 to clear geometry constraint)
    ( 2.8, -2.8),   # SE entrance (Clears dyn_obs_12(2, -2.5))
]

# ── Phase 4 bridge goals: corridor vs selective corners ────────────────────
_GOALS_INTER_CORNER_RAW = [
    # Near-room bridge goals: inside the room approach, but not at the deep corners.
    (-3.3,  2.2),
    ( 3.3,  2.2),
    (-3.3, -2.2),
    ( 3.3, -2.2),
]

_GOALS_INTER_CORRIDOR_RAW = [
    # Corridor bridge goals that survive the current safety filter.
    (-0.5,  3.0),
    ( 0.5,  3.0),
    ( 0.0, -3.1),
    (-0.5, -3.0),
    ( 0.5, -3.0),
]

# ── Deep goals split by geometry so long corridors are learned before deep corners ──
_GOALS_DEEP_CORNER_RAW = [
    (-4.4,  4.4),   # NW extreme deep
    ( 4.4,  4.4),   # NE extreme deep
    (-4.4, -4.4),   # SW extreme deep
    ( 4.4, -4.4),   # SE extreme deep
]

_GOALS_DEEP_CORRIDOR_RAW = [
    # Extreme corridor ends are far, but much more learnable than the deep diagonals.
    ( 0.0,  4.4),
    ( 0.0, -4.4),
    ( 4.4,  0.0),
    (-4.4,  0.0),
]

_GOALS_DEEP_RAW = _GOALS_DEEP_CORRIDOR_RAW + _GOALS_DEEP_CORNER_RAW

# ── Phase 7 mastery subset: keep the hardest learnable goals in training ───
# The diagonal deep-corner stress goals remain useful for evaluation, but they
# are too destructive as bulk curriculum samples under full dynamics.
_GOALS_PHASE7_EXPAND_RAW = [
    ( 0.0,  2.5),
    ( 0.0, -2.5),
    ( 3.2,  1.5),
]

_GOALS_PHASE7_INTER_CORNER_RAW = [
    (-3.3,  2.2),
    ( 3.3,  2.2),
    (-3.3, -2.2),
    ( 3.3, -2.2),
]

_GOALS_PHASE7_DEEP_MASTER_RAW = [
    (-4.4,  0.0),
    ( 0.0, -4.4),
    ( 4.4,  0.0),
]

_GOALS_PHASE7_DEEP_STRETCH_RAW = [
    ( 0.0,  4.4),
]

# ── Phase advancement thresholds (from config or defaults) ───────────────────
# NEW CURRICULUM v2.0 - Two stages: STATIC (1-4) and DYNAMIC (5-7)
#
# STATIC STAGE: All obstacles FROZEN, learn navigation
PHASE1_WINDOW    = _CFG.get("phase1_window",      50)
PHASE1_SR_THRESH = _CFG.get("phase1_sr_thresh", 0.70)   # 70% → Phase 2
PHASE2_WINDOW    = _CFG.get("phase2_window",      50)
PHASE2_SR_THRESH = _CFG.get("phase2_sr_thresh", 0.70)   # 70% → Phase 3
PHASE3_WINDOW    = _CFG.get("phase3_window",      50)
PHASE3_SR_THRESH = _CFG.get("phase3_sr_thresh", 0.65)   # 65% → Phase 4
PHASE4_WINDOW    = _CFG.get("phase4_window",     100)   # LONGER window
PHASE4_SR_THRESH = _CFG.get("phase4_sr_thresh", 0.60)   # 60% → Phase 5 (realistic for all-goal pool)

# DYNAMIC STAGE: Obstacles start moving, gradual difficulty
PHASE5_WINDOW    = _CFG.get("phase5_window",      50)
PHASE5_SR_THRESH = _CFG.get("phase5_sr_thresh", 0.60)   # 60% → Phase 6
PHASE6_WINDOW    = _CFG.get("phase6_window",      50)
PHASE6_SR_THRESH = _CFG.get("phase6_sr_thresh", 0.55)   # 55% → Phase 7
PHASE7_WINDOW = _CFG.get("phase7_window", PHASE6_WINDOW)
PHASE7_REGRESS_SR_THRESH = _CFG.get("phase7_regress_sr_thresh", 0.50)
PHASE7_MIN_EPISODES      = _CFG.get("phase7_min_episodes", 40)
PHASE7_COVERAGE_EVERY = max(0, int(_CFG.get("phase7_coverage_every", 6)))
# Phase 7 is final - no threshold

# ── Apply safety filtering ───────────────────────────────────────────────────
GOALS_CORE   = [g for g in _GOALS_CORE_RAW   if _is_safe_early(*g)]
GOALS_EXPAND = [g for g in _GOALS_EXPAND_RAW if _is_safe_early(*g)]
GOALS_MID    = [g for g in _GOALS_MID_RAW    if _is_safe_early(*g)]
GOALS_INTER_CORNER = [g for g in _GOALS_INTER_CORNER_RAW if _is_safe_early(*g)]
GOALS_INTER_CORRIDOR = [g for g in _GOALS_INTER_CORRIDOR_RAW if _is_safe_early(*g)]
GOALS_INTER  = GOALS_INTER_CORRIDOR + GOALS_INTER_CORNER
GOALS_DEEP_CORRIDOR = [g for g in _GOALS_DEEP_CORRIDOR_RAW if _is_safe_early(*g)]
GOALS_DEEP_CORNER = [g for g in _GOALS_DEEP_CORNER_RAW if _is_safe_early(*g)]
GOALS_DEEP   = GOALS_DEEP_CORRIDOR + GOALS_DEEP_CORNER
GOALS_PHASE7_EXPAND = [g for g in _GOALS_PHASE7_EXPAND_RAW if _is_safe_early(*g)]
GOALS_PHASE7_INTER_CORNER = [g for g in _GOALS_PHASE7_INTER_CORNER_RAW if _is_safe_early(*g)]
GOALS_PHASE7_DEEP_MASTER = [g for g in _GOALS_PHASE7_DEEP_MASTER_RAW if _is_safe_early(*g)]
GOALS_PHASE7_DEEP_STRETCH = [g for g in _GOALS_PHASE7_DEEP_STRETCH_RAW if _is_safe_early(*g)]

# Combined pools for convenience
GOALS_ALL  = GOALS_CORE + GOALS_EXPAND + GOALS_MID + GOALS_INTER + GOALS_DEEP
GOALS_HARD = [g for g in (GOALS_EXPAND + GOALS_MID + GOALS_INTER + GOALS_DEEP) if _is_safe_late(*g)]

# goal-position → category label (for eval logging)
_GOAL_CATEGORY: dict = {}
for _g in GOALS_CORE:          _GOAL_CATEGORY[_g] = "CORE"
for _g in GOALS_EXPAND:        _GOAL_CATEGORY[_g] = "EXPAND"
for _g in GOALS_MID:           _GOAL_CATEGORY[_g] = "MID"
for _g in GOALS_INTER:         _GOAL_CATEGORY[_g] = "INTER"
for _g in GOALS_DEEP:          _GOAL_CATEGORY[_g] = "DEEP"
for _g in GOALS_PHASE7_EXPAND:       _GOAL_CATEGORY[_g] = "EXPAND"
for _g in GOALS_PHASE7_INTER_CORNER: _GOAL_CATEGORY[_g] = "INTER"
for _g in GOALS_PHASE7_DEEP_MASTER:  _GOAL_CATEGORY[_g] = "DEEP"
for _g in GOALS_PHASE7_DEEP_STRETCH: _GOAL_CATEGORY[_g] = "DEEP"

# Startup verification
print(f"[GoalManager] CURRICULUM v3.0 - Step-by-step corner learning:")
print(f"  CORE={len(GOALS_CORE)}/{len(_GOALS_CORE_RAW)}  "
      f"EXPAND={len(GOALS_EXPAND)}/{len(_GOALS_EXPAND_RAW)}  "
      f"MID={len(GOALS_MID)}/{len(_GOALS_MID_RAW)}  "
    f"INTER_CORR={len(GOALS_INTER_CORRIDOR)}/{len(_GOALS_INTER_CORRIDOR_RAW)}  "
    f"INTER_CORNER={len(GOALS_INTER_CORNER)}/{len(_GOALS_INTER_CORNER_RAW)}  "
    f"DEEP_CORR={len(GOALS_DEEP_CORRIDOR)}/{len(_GOALS_DEEP_CORRIDOR_RAW)}  "
    f"DEEP_CORNER={len(GOALS_DEEP_CORNER)}/{len(_GOALS_DEEP_CORNER_RAW)}")
print(f"  Phase7 mastery subset: EXPAND={len(GOALS_PHASE7_EXPAND)}/{len(_GOALS_PHASE7_EXPAND_RAW)}  "
    f"INTER_CORNER={len(GOALS_PHASE7_INTER_CORNER)}/{len(_GOALS_PHASE7_INTER_CORNER_RAW)}  "
    f"DEEP_MASTER={len(GOALS_PHASE7_DEEP_MASTER)}/{len(_GOALS_PHASE7_DEEP_MASTER_RAW)}  "
    f"DEEP_STRETCH={len(GOALS_PHASE7_DEEP_STRETCH)}/{len(_GOALS_PHASE7_DEEP_STRETCH_RAW)}")
print(f"  Total: {len(GOALS_ALL)} goals available")

# ── PHASE_POOLS — NEW v2.0 CURRICULUM ─────────────────────────────────────────
#
# STATIC STAGE (Phases 1-4): Learn to navigate to ALL distances
# - Phase 1: CORE only (100% easy) - learn basics
# - Phase 2: 60% CORE, 40% EXPAND - transition to medium goals
# - Phase 3: 40% CORE, 30% EXPAND, 30% MID - learn corner entrances
# - Phase 4: 25% each (EQUAL WEIGHTING!) - MUST learn hard goals!
#
# DYNAMIC STAGE (Phases 5-7): Learn to handle moving obstacles
# - Goal mixes stay confidence-preserving while obstacle difficulty rises
# - Phase 5: slow dynamics with mostly solved static goals + corridor bridge
# - Phase 6: medium dynamics with more corner pressure and first deep goals
# - Phase 7: full dynamics with the hardest mixed pool
#
PHASE_WINDOWS = {
    1: PHASE1_WINDOW, 2: PHASE2_WINDOW, 3: PHASE3_WINDOW, 4: PHASE4_WINDOW,
    5: PHASE5_WINDOW, 6: PHASE6_WINDOW, 7: PHASE7_WINDOW
}
PHASE_THRESH = {
    1: PHASE1_SR_THRESH, 2: PHASE2_SR_THRESH, 3: PHASE3_SR_THRESH, 4: PHASE4_SR_THRESH,
    5: PHASE5_SR_THRESH, 6: PHASE6_SR_THRESH  # Phase 7 is final
}


def _curriculum_weight(phase_key: str, weight_key: str, default: int) -> int:
    weights = _CFG.get("curriculum", {}).get(phase_key, {})
    return max(0, int(weights.get(weight_key, default)))


def _weighted_pool(goal_list, weight: int):
    return goal_list * max(0, int(weight))


PHASE4_CORE_W = _curriculum_weight("phase4_weights", "core", 1)
PHASE4_EXPAND_W = _curriculum_weight("phase4_weights", "expand", 1)
PHASE4_MID_W = _curriculum_weight("phase4_weights", "mid", 2)
PHASE4_INTER_CORRIDOR_W = _curriculum_weight("phase4_weights", "inter_corridor", 3)
PHASE4_INTER_CORNER_W = _curriculum_weight("phase4_weights", "inter_corner", 1)
PHASE4_DEEP_CORRIDOR_W = _curriculum_weight("phase4_weights", "deep_corridor", 0)
PHASE4_DEEP_CORNER_W = _curriculum_weight("phase4_weights", "deep_corner", 0)

PHASE5_CORE_W = _curriculum_weight("phase5_weights", "core", 2)
PHASE5_EXPAND_W = _curriculum_weight("phase5_weights", "expand", 2)
PHASE5_MID_W = _curriculum_weight("phase5_weights", "mid", 2)
PHASE5_INTER_CORRIDOR_W = _curriculum_weight("phase5_weights", "inter_corridor", 2)
PHASE5_INTER_CORNER_W = _curriculum_weight("phase5_weights", "inter_corner", 1)
PHASE5_DEEP_CORRIDOR_W = _curriculum_weight("phase5_weights", "deep_corridor", _curriculum_weight("phase5_weights", "deep", 0))
PHASE5_DEEP_CORNER_W = _curriculum_weight("phase5_weights", "deep_corner", 0)

PHASE6_CORE_W = _curriculum_weight("phase6_weights", "core", 1)
PHASE6_EXPAND_W = _curriculum_weight("phase6_weights", "expand", 2)
PHASE6_MID_W = _curriculum_weight("phase6_weights", "mid", 2)
PHASE6_INTER_CORRIDOR_W = _curriculum_weight("phase6_weights", "inter_corridor", 2)
PHASE6_INTER_CORNER_W = _curriculum_weight("phase6_weights", "inter_corner", 2)
PHASE6_DEEP_CORRIDOR_W = _curriculum_weight("phase6_weights", "deep_corridor", _curriculum_weight("phase6_weights", "deep", 1))
PHASE6_DEEP_CORNER_W = _curriculum_weight("phase6_weights", "deep_corner", 0)

PHASE7_CORE_W = _curriculum_weight("phase7_weights", "core", 2)
PHASE7_EXPAND_W = _curriculum_weight("phase7_weights", "expand", 2)
PHASE7_INTER_CORRIDOR_W = _curriculum_weight("phase7_weights", "inter_corridor", 3)
PHASE7_INTER_CORNER_W = _curriculum_weight("phase7_weights", "inter_corner", 2)
PHASE7_DEEP_MASTER_W = _curriculum_weight("phase7_weights", "deep_master", _curriculum_weight("phase7_weights", "deep_corridor", 1))
PHASE7_DEEP_STRETCH_W = _curriculum_weight("phase7_weights", "deep_stretch", 1)

# Goal pools with smoother late-stage progression.
PHASE_POOLS = {
    # STATIC STAGE - learn navigation step-by-step to corners
    1: GOALS_CORE,                                                      # 8 goals (basics)
    2: GOALS_CORE * 3 + GOALS_EXPAND * 2,                               # 60% easy, 40% medium
    3: GOALS_CORE * 2 + GOALS_EXPAND * 2 + GOALS_MID * 2,               # + corner entrances
    4: (
        _weighted_pool(GOALS_CORE, PHASE4_CORE_W)
        + _weighted_pool(GOALS_EXPAND, PHASE4_EXPAND_W)
        + _weighted_pool(GOALS_MID, PHASE4_MID_W)
        + _weighted_pool(GOALS_INTER_CORRIDOR, PHASE4_INTER_CORRIDOR_W)
        + _weighted_pool(GOALS_INTER_CORNER, PHASE4_INTER_CORNER_W)
        + _weighted_pool(GOALS_DEEP_CORRIDOR, PHASE4_DEEP_CORRIDOR_W)
        + _weighted_pool(GOALS_DEEP_CORNER, PHASE4_DEEP_CORNER_W)
    ),

    # DYNAMIC STAGE - moving obstacles
    5: (
        _weighted_pool(GOALS_CORE, PHASE5_CORE_W)
        + _weighted_pool(GOALS_EXPAND, PHASE5_EXPAND_W)
        + _weighted_pool(GOALS_MID, PHASE5_MID_W)
        + _weighted_pool(GOALS_INTER_CORRIDOR, PHASE5_INTER_CORRIDOR_W)
        + _weighted_pool(GOALS_INTER_CORNER, PHASE5_INTER_CORNER_W)
        + _weighted_pool(GOALS_DEEP_CORRIDOR, PHASE5_DEEP_CORRIDOR_W)
        + _weighted_pool(GOALS_DEEP_CORNER, PHASE5_DEEP_CORNER_W)
    ),
    6: (
        _weighted_pool(GOALS_CORE, PHASE6_CORE_W)
        + _weighted_pool(GOALS_EXPAND, PHASE6_EXPAND_W)
        + _weighted_pool(GOALS_MID, PHASE6_MID_W)
        + _weighted_pool(GOALS_INTER_CORRIDOR, PHASE6_INTER_CORRIDOR_W)
        + _weighted_pool(GOALS_INTER_CORNER, PHASE6_INTER_CORNER_W)
        + _weighted_pool(GOALS_DEEP_CORRIDOR, PHASE6_DEEP_CORRIDOR_W)
        + _weighted_pool(GOALS_DEEP_CORNER, PHASE6_DEEP_CORNER_W)
    ),
    7: (
        _weighted_pool(GOALS_CORE, PHASE7_CORE_W)
        + _weighted_pool(GOALS_PHASE7_EXPAND, PHASE7_EXPAND_W)
        + _weighted_pool(GOALS_INTER_CORRIDOR, PHASE7_INTER_CORRIDOR_W)
        + _weighted_pool(GOALS_PHASE7_INTER_CORNER, PHASE7_INTER_CORNER_W)
        + _weighted_pool(GOALS_PHASE7_DEEP_MASTER, PHASE7_DEEP_MASTER_W)
        + _weighted_pool(GOALS_PHASE7_DEEP_STRETCH, PHASE7_DEEP_STRETCH_W)
    ),
}

PHASE7_BASE_UNIQUE = list(dict.fromkeys(PHASE_POOLS[7]))
PHASE7_COVERAGE_POOL = [g for g in GOALS_HARD if g not in PHASE7_BASE_UNIQUE]

# Print pool compositions
print(f"[GoalManager] Phase pool compositions:")
print(f"  --- STATIC STAGE (obstacles frozen) ---")
for p in range(1, 6):
    pool = PHASE_POOLS[p]
    print(f"  Phase {p}: {len(pool)} samples")
print(f"  --- DYNAMIC STAGE (obstacles moving) ---")
for p in range(5, 8):
    pool = PHASE_POOLS[p]
    print(f"  Phase {p}: {len(pool)} samples")
print(f"  Phase 7 coverage sweep: every {PHASE7_COVERAGE_EVERY} eps over {len(PHASE7_COVERAGE_POOL)} extra hard goals")

# =============================================================================
#  GAZEBO GOAL MARKER SDF
# =============================================================================
PHASE_COLORS = {
    1: {"r": "0.0",  "g": "1.0",  "b": "0.2"},   # green  - CORE (Phase 1)
    2: {"r": "1.0",  "g": "0.9",  "b": "0.0"},   # yellow - EXPAND (Phase 2)
    3: {"r": "1.0",  "g": "0.5",  "b": "0.0"},   # orange - MID (Phase 3)
    4: {"r": "0.8",  "g": "0.3",  "b": "0.0"},   # dark orange - Phase 4 bridge
    5: {"r": "0.5",  "g": "0.0",  "b": "1.0"},   # purple - slow dynamics
    6: {"r": "0.8",  "g": "0.0",  "b": "0.8"},   # magenta - medium dynamics
    7: {"r": "1.0",  "g": "0.0",  "b": "0.5"},   # pink   - full dynamics
}

def _make_goal_sdf(phase: int = 1) -> str:
    c = PHASE_COLORS.get(phase, PHASE_COLORS[1])
    r, g, b = c["r"], c["g"], c["b"]
    col      = f"{r} {g} {b} 0.8"
    col_em   = f"{r} {g} {b} 1"
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="goal_pole">
    <static>true</static>
    <link name="link">
      <visual name="ground_disc">
        <pose>0 0 0.01 0 0 0</pose>
        <geometry><cylinder><radius>0.4</radius><length>0.02</length></cylinder></geometry>
        <material>
          <ambient>{col}</ambient><diffuse>{col}</diffuse>
          <emissive>{col_em}</emissive><specular>0.5 0.5 0.5 1</specular>
        </material>
      </visual>
      <visual name="sphere">
        <pose>0 0 1.2 0 0 0</pose>
        <geometry><sphere><radius>0.15</radius></sphere></geometry>
        <material>
          <ambient>{col}</ambient><diffuse>{col}</diffuse><emissive>{col_em}</emissive>
        </material>
      </visual>
      <visual name="beam">
        <pose>0 0 0.6 0 0 0</pose>
        <geometry><cylinder><radius>0.02</radius><length>1.2</length></cylinder></geometry>
        <material>
          <ambient>{col}</ambient><diffuse>{col}</diffuse><emissive>{col_em}</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


class GoalManagerDynamic(Node):
    def __init__(self):
        super().__init__("goal_manager_dynamic")

        self.declare_parameter("start_phase", 0)
        self.declare_parameter("eval_mode", False)
        force_phase = int(self.get_parameter("start_phase").value)
        self._eval_mode = bool(self.get_parameter("eval_mode").value)

        self._phase         = 1
        self._pool          = list(GOALS_CORE)
        self._prev_idx      = -1
        self._result_window = collections.deque(maxlen=PHASE1_WINDOW)
        self._total_eps     = 0
        self._total_succs   = 0
        self._ep_open       = False
        self._phase7_cov_idx = -1
        # per-goal-category stats (eval mode)
        self._eval_cat_stats: dict = {}   # category -> [attempts, successes]
        self._eval_current_goal: tuple = (0.0, 0.0)
        self._last_was_spawn_kill = False
        self._spawn_kill_retries  = 0

        # Eval mode: deterministic round-robin over unique goals
        self._eval_cycle_idx = -1
        self._eval_unique_goals: list = []

        if self._eval_mode:
            self.get_logger().info(
                "[Goals] *** EVAL MODE: phase locked, no advancement, no state save ***")
            self._total_eps   = 0   # always start from ep 0 in eval
            self._total_succs = 0

        # Handle phase initialization based on start_phase parameter
        if force_phase >= 1:
            # Force specific phase (for fresh training or testing)
            self._phase = max(1, min(7, force_phase))
            self._pool  = list(PHASE_POOLS[self._phase])
            self._result_window = collections.deque(maxlen=PHASE_WINDOWS[self._phase])
            self.get_logger().info(
                f"[Goals] start_phase={force_phase} → Phase {self._phase} forced (fresh start)")

        # Build deterministic eval cycle: sorted unique goals from pool
        if self._eval_mode:
            seen = set()
            for g in self._pool:
                key = (round(g[0], 4), round(g[1], 4))
                if key not in seen:
                    seen.add(key)
                    self._eval_unique_goals.append(g)
            self._eval_unique_goals.sort(key=lambda g: (g[0], g[1]))
            self.get_logger().info(
                f"[Goals] EVAL deterministic cycle: {len(self._eval_unique_goals)} unique goals")
        else:
            # Auto-resume: try to load saved state
            saved_phase = self._load_saved_phase()
            if saved_phase >= 1:
                # Window was already restored in _load_saved_phase
                self._phase = saved_phase
                self._pool  = list(PHASE_POOLS[self._phase])
                # DON'T clear the window - it was loaded!
                self.get_logger().info(
                    f"[Goals] *** Resumed Phase {self._phase}, "
                    f"window={len(self._result_window)}/{PHASE_WINDOWS[self._phase]} ***")
            else:
                # No saved state, start fresh at Phase 1
                self.get_logger().info(
                    f"[Goals] No saved state found, starting fresh at Phase 1")
            # Seed from CSV for any missing episode data
            self._seed_from_csv()

        self._marker_spawned      = False
        self._marker_phase        = self._phase
        self._pending_marker      = None
        self._confirmed_marker_pos = None
        self._spawn_cli    = self.create_client(SpawnEntity, "/spawn_entity")
        self._delete_cli   = self.create_client(DeleteEntity, "/delete_entity")
        self._set_cli      = self.create_client(SvcSetState, "/set_entity_state")
        self._current_goal  = None
        self._marker_busy   = False
        self.create_timer(0.1, self._enforce_marker)
        self._last_goal_time = 0.0

        # TRANSIENT_LOCAL = latched: any new subscriber immediately gets the last value
        _latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._goal_pub  = self.create_publisher(String, "/tb3_drl/goal", _latched)
        self._phase_pub = self.create_publisher(Int32, "/tb3_drl/curriculum_phase", _latched)
        self.create_subscription(
            Bool,              "/tb3_drl/need_goal",   self._on_need_goal,    10)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/step_result", self._on_step_result,  10)
        self.create_subscription(
            Float32MultiArray, "/tb3_drl/reset_obs",   self._on_reset_obs,    RESET_OBS_QOS)

        self._last_goal_msg = None
        self.create_timer(1.5, self._republish)

        # Track episodes spent in the current phase to prevent jittery transitions
        self._phase_ep_count = 0
        self._startup_done   = False
        self.create_timer(0.8, self._startup_goal)

        self._print_phase_banner()
        self.get_logger().info("[Goals] Waiting 0.8s before first goal…")
        self.create_timer(0.5, self._publish_phase_once)

    def _publish_phase_once(self):
        msg = Int32(); msg.data = self._phase
        self._phase_pub.publish(msg)

    def _seed_from_csv(self):
        import csv
        csv_path = _run_log_path()
        if not csv_path or not os.path.exists(csv_path):
            return
        
        try:
            rows = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                if "goal_reached" not in (reader.fieldnames or []):
                    return
                for row in reader:
                    rows.append(row)
            if not rows:
                return

            self._total_eps   = len(rows)
            self._total_succs = sum(int(r["goal_reached"]) for r in rows)

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

    def _print_phase_banner(self):
        names = {
            1: "CORE    (1.2–2.1m, center hub) — STATIC",
            2: "EXPAND  (2.5–3.0m, doorways) — STATIC",
            3: "REACH   (3.8m, corner entrances) — STATIC",
            4: "BRIDGE  (3.1–4.4m, corridors + selective corners) — STATIC",
            5: "SLOW DYNAMICS (bridge + far corridors, 6 obstacles)",
            6: "MEDIUM DYNAMICS (hard corners + far corridors, 9 obstacles)",
            7: "FULL DYNAMICS (mastery pool, all obstacles)",
        }
        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"  Goal Manager  Phase {self._phase} — {names[self._phase]}")
        self.get_logger().info(
            f"  {len(PHASE_POOLS[self._phase])} samples in pool")
        adv = {
            1: f"Advances → Phase 2 when SR >= {int(PHASE1_SR_THRESH*100)}% over {PHASE1_WINDOW} eps",
            2: f"Advances → Phase 3 when SR >= {int(PHASE2_SR_THRESH*100)}% over {PHASE2_WINDOW} eps",
            3: f"Advances → Phase 4 when SR >= {int(PHASE3_SR_THRESH*100)}% over {PHASE3_WINDOW} eps",
            4: f"Advances → Phase 5 (slow dynamics) when SR >= {int(PHASE4_SR_THRESH*100)}% over {PHASE4_WINDOW} eps",
            5: f"Advances → Phase 6 when SR >= {int(PHASE5_SR_THRESH*100)}% over {PHASE5_WINDOW} eps",
            6: f"Advances → Phase 7 when SR >= {int(PHASE6_SR_THRESH*100)}% over {PHASE6_WINDOW} eps",
            7: f"Final phase — falls back to Phase 6 if SR < {int(PHASE7_REGRESS_SR_THRESH*100)}% over {PHASE7_WINDOW} eps",
        }
        self.get_logger().info(f"  {adv[self._phase]}")
        self.get_logger().info("=" * 60)

    def _regress_phase(self, sr: float):
        prev_phase = self._phase
        self._phase = max(1, self._phase - 1)
        self._phase_ep_count = 0
        self._pool = list(PHASE_POOLS[self._phase])
        self._prev_idx = -1
        self._result_window = collections.deque(maxlen=PHASE_WINDOWS[self._phase])
        self.get_logger().warn("=" * 60)
        self.get_logger().warn(
            f"  PHASE {prev_phase} COLLAPSED → returning to Phase {self._phase}")
        self.get_logger().warn(
            f"  SR={sr*100:.0f}%  ep={self._total_eps}  succ={self._total_succs}")
        self._print_phase_banner()
        self._save_phase()
        msg = Int32(); msg.data = self._phase
        self._phase_pub.publish(msg)
        self.get_logger().warn(
            f"  Published curriculum_phase={self._phase}")

    def _on_step_result(self, msg: Float32MultiArray):
        # Handle both old Environment-style (len 57) and new Trainer-style (len 6) results
        if len(msg.data) == 6:
            # Training report: [ep, reward, steps, coll, goal, info]
            done    = True
            info    = int(msg.data[5])
        elif len(msg.data) >= 57:
            # Environment update: [...] + [rew, done, info]
            done    = bool(msg.data[55])
            info    = int(msg.data[56])
        else:
            return  # Ignore incomplete messages

        success = (info == 1)

        if done and self._ep_open:
            if self._is_impossible_terminal(info):
                self.get_logger().info(f"[Goals] Ignored impossible terminal info={info} (too fast)")
                self._last_was_spawn_kill = True
                return
            
            self._ep_open     = False
            self.get_logger().info(f"[Goals] Episode CLOSED (info={info}). Ready for next goal.")
            
            if info == 0:
                self._last_was_spawn_kill = True
                return
            self._last_was_spawn_kill = False
            self._spawn_kill_retries  = 0
            self._total_eps   += 1
            self._total_succs += int(success)
            self._phase_ep_count += 1

            # ── Per-category tracking (eval mode) ──
            if self._eval_mode:
                cat = _GOAL_CATEGORY.get(self._eval_current_goal, "OTHER")
                if cat not in self._eval_cat_stats:
                    self._eval_cat_stats[cat] = [0, 0]
                self._eval_cat_stats[cat][0] += 1
                self._eval_cat_stats[cat][1] += int(success)

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

            # Eval mode: per-category breakdown every 50 eps
            if self._eval_mode and self._total_eps % 50 == 0:
                self._log_eval_category_summary()

            # Save state every 10 episodes for seamless resume
            if self._total_eps % 10 == 0:
                self._save_phase()

            if not self._eval_mode:
                thresh = PHASE_THRESH.get(self._phase, 1.0)
                req_w  = PHASE_WINDOWS.get(self._phase, PHASE3_WINDOW)
                if (self._phase < 7
                        and len(self._result_window) >= req_w
                        and sr >= thresh):
                    self._advance_phase(sr)
                elif (self._phase == 7
                        and len(self._result_window) >= req_w
                        and self._phase_ep_count >= PHASE7_MIN_EPISODES
                        and sr < PHASE7_REGRESS_SR_THRESH):
                    self._regress_phase(sr)

    def _log_eval_category_summary(self):
        """Print per-category SR breakdown during eval."""
        CAT_ORDER = ["CORE", "EXPAND", "MID", "INTER", "DEEP", "OTHER"]
        lines = [f"  [EVAL Ph{self._phase}] ep={self._total_eps}  "
                 f"overall SR={self._total_succs/max(1,self._total_eps)*100:.1f}%  "
                 f"({self._total_succs}/{self._total_eps})"]
        for cat in CAT_ORDER:
            if cat in self._eval_cat_stats:
                n, s = self._eval_cat_stats[cat]
                lines.append(f"    {cat:<8} {s:>3}/{n:<3}  SR={s/n*100:.0f}%")
        self.get_logger().info("\n".join(lines))

    def _save_phase(self):
        """Save complete curriculum state for seamless resume. No-op in eval mode."""
        if self._eval_mode:
            return
        import json
        phase_file, state_file = _phase_state_paths()
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        state = {
            "phase": self._phase,
            "result_window": list(self._result_window),
            "total_eps": self._total_eps,
            "total_succs": self._total_succs,
        }
        with open(state_file, "w") as f:
            json.dump(state, f)
        with open(phase_file, "w") as f:
            f.write(str(self._phase))

    def _load_saved_phase(self) -> int:
        """Load complete curriculum state if available."""
        import json
        phase_file, state_file = _phase_state_paths()
        try:
            with open(state_file) as f:
                state = json.load(f)
            phase = max(1, min(7, int(state.get("phase", 1))))
            # Restore window history for seamless resume
            window_data = state.get("result_window", [])
            if window_data:
                req_window = PHASE_WINDOWS.get(phase, 50)
                self._result_window = collections.deque(window_data, maxlen=req_window)
            # Restore counters
            self._total_eps = state.get("total_eps", 0)
            self._total_succs = state.get("total_succs", 0)
            self.get_logger().info(
                f"[Resume] Phase {phase}, window={len(self._result_window)}, "
                f"eps={self._total_eps}, succs={self._total_succs}")
            return phase
        except Exception:
            try:
                with open(phase_file) as f:
                    return max(1, min(7, int(f.read().strip())))
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
        self._save_phase()
        msg = Int32(); msg.data = self._phase
        self._phase_pub.publish(msg)
        self.get_logger().info(
            f"  Published curriculum_phase={self._phase}")

    def _on_need_goal(self, msg: Bool):
        if not msg.data:
            return
        # ── Spawn-kill protection: if the previous episode was a spawn
        #    artifact (info=0 or impossible terminal), re-use the SAME
        #    goal instead of picking a new one.  Gives up after 5
        #    consecutive retries to avoid infinite loops.
        if (self._last_was_spawn_kill
                and self._last_goal_msg is not None
                and self._spawn_kill_retries < 3):
            self._spawn_kill_retries += 1
            self._goal_pub.publish(self._last_goal_msg)
            self._last_goal_time = time.time()
            self._ep_open = True
            self.get_logger().info(
                f"[Goals] Spawn-kill retry "
                f"{self._spawn_kill_retries}/3 — same goal")
            return
        # ── Guard: ignore duplicate need_goal if an episode is already open
        if self._ep_open:
            return
        self._spawn_kill_retries = 0
        self._ep_open = True
        self._publish_next_goal()

    def _publish_next_goal(self):
        use_phase7_coverage = (
            not self._eval_mode       # ← never inject coverage goals during eval
            and self._phase == 7
            and PHASE7_COVERAGE_EVERY > 0
            and len(PHASE7_COVERAGE_POOL) > 0
            and self._total_eps > 0
            and self._total_eps % PHASE7_COVERAGE_EVERY == 0
        )

        if use_phase7_coverage:
            self._phase7_cov_idx = (self._phase7_cov_idx + 1) % len(PHASE7_COVERAGE_POOL)
            gx, gy = PHASE7_COVERAGE_POOL[self._phase7_cov_idx]
            self.get_logger().info(
                f"[Goals Ph7] Coverage sweep goal {self._phase7_cov_idx + 1}/{len(PHASE7_COVERAGE_POOL)}")
        elif self._eval_mode and self._eval_unique_goals:
            # Deterministic round-robin over unique goals
            self._eval_cycle_idx = (self._eval_cycle_idx + 1) % len(self._eval_unique_goals)
            gx, gy = self._eval_unique_goals[self._eval_cycle_idx]
        else:
            pool  = self._pool
            cands = [i for i in range(len(pool)) if i != self._prev_idx]
            
            # Filter pool for spawn safety
            safe_cands = [i for i in cands if _check_static(pool[i][0], pool[i][1])]
            if not safe_cands:
                # Fallback: if all goals are blocked, allow previous but warn
                self.get_logger().warn("[Goals] No safe goals in current pool! Relaxing safety...")
                safe_cands = cands

            idx   = random.choice(safe_cands)
            self._prev_idx = idx
            gx, gy = pool[idx]
        self._last_goal_time = time.time()

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
            f"[Goals Ph{self._phase}] Goal: "
            f"({gx:+.2f},{gy:+.2f}) {dist:.1f}m{sr_str}")

        self._pending_marker = (gx, gy)
        self._place_goal_marker(gx, gy)
        self._eval_current_goal = (round(gx, 4), round(gy, 4))

    def _startup_goal(self):
        if not self._startup_done:
            self._startup_done = True
            self.get_logger().info("[Goals] Publishing startup goal…")
            self._ep_open = True
            self._publish_next_goal()

    def _is_impossible_terminal(self, info: int) -> bool:
        if info == 0:
            return False
        elapsed = time.time() - self._last_goal_time
        if elapsed < RESET_SETTLE_S:
            return True
        if info == 8 and elapsed < MIN_STUCK_SECONDS:
            return True
        if info == 4 and elapsed < MIN_TIMEOUT_SECONDS:
            return True
        return False

    def _on_reset_obs(self, msg: Float32MultiArray):
        # Clear busy flag so the upcoming new-goal placement can proceed immediately.
        # Do NOT re-place the old goal here — it just delays showing the real new goal.
        self._marker_busy = False
        self._confirmed_marker_pos = None
        # If marker was never spawned, spawn it now at the current goal position.
        if not self._marker_spawned and self._current_goal is not None:
            if self._spawn_cli.service_is_ready():
                self._spawn_marker(*self._current_goal)

    def _republish(self):
        if self._last_goal_msg is not None:
            self._goal_pub.publish(self._last_goal_msg)

    def _enforce_marker(self):
        if self._current_goal is None or self._marker_busy:
            return
        x, y = self._current_goal
        if not self._marker_spawned:
            if self._spawn_cli.service_is_ready():
                self._spawn_marker(x, y)
        else:
            if self._confirmed_marker_pos == (x, y):
                return
            if self._set_cli.service_is_ready():
                self._move_marker(x, y)

    def _place_goal_marker(self, x: float, y: float):
        self._current_goal = (x, y)
        if not self._marker_spawned:
            if self._spawn_cli.service_is_ready():
                self._spawn_marker(x, y)
        else:
            if self._set_cli.service_is_ready():
                # Force immediate update for new goals — clear any in-flight notion
                # so the new position wins without waiting up to 0.3s for _enforce_marker.
                self._marker_busy = False
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
            self._confirmed_marker_pos = self._current_goal
        except Exception as e:
            self._confirmed_marker_pos = None
            self.get_logger().warn(f"[Goals] Marker move failed ({e}) — will retry")


def main(args=None):
    rclpy.init(args=args)
    node = GoalManagerDynamic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
