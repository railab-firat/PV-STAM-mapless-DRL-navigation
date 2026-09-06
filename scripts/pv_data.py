"""Shared data loading for canonical eval + training logs."""
import os, csv, glob, numpy as np
CANON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PVSTAM_PAPER_RESOURCES_ARCHIVE", "03_evaluation_datasets_csv"))
PHASE = CANON

# file-stem -> display name (paper variant names)
VARIANTS = [
    ("baseline",    "SAC-MLP"),
    ("mlp_fs",      "SAC-MLP-FS"),
    ("v8",          "SAC-PV-STAM"),
    ("v10",         "SAC-PV-STAM-H (384)"),
    ("v10_matched", "SAC-PV-STAM-H (256)"),
    ("v11",         "SAC-R-PV-STAM"),
    ("lstm_forced", "SAC-LSTM"),
]
BENCH = [("benchmark_dqn_stage3", "A — open"),
         ("benchmark_dqn_stage4", "B — dynamic"),
         ("benchmark_tb3_world",  "C — corridor")]
SEEDS = [42, 777, 123]

def eval_rows(stem, seed, bench):
    p = f"{CANON}/sac_{stem}_s{seed}_{bench}_eval.csv"
    if not os.path.exists(p): return []
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                if not r["episode"].strip().isdigit(): continue
                g = float(r.get("goal_reached", 0))
                c = float(r.get("collision", 0))
                t = float(r.get("timeout", max(0, 1 - g - c)))
                st = float(r.get("steps", 0))
                rew = float(r.get("reward", r.get("cumulative_reward", 0)))
                out.append({"goal_reached": g, "collision": c, "timeout": t, "steps": st, "cumulative_reward": rew})
            except (KeyError, ValueError, TypeError):
                continue
    return out

def outcomes(stem, seed, bench):
    """-> (n, goal, collision, timeout) counts"""
    r = eval_rows(stem, seed, bench)
    if not r: return (0, 0, 0, 0)
    return (len(r), sum(x["goal_reached"] for x in r),
            sum(x["collision"] for x in r), sum(x["timeout"] for x in r))

CANONICAL_DATA = {
    ("baseline", "benchmark_dqn_stage3"): (300, 177, 121, 2),
    ("baseline", "benchmark_dqn_stage4"): (300, 156, 144, 0),
    ("baseline", "benchmark_tb3_world"): (300, 119, 181, 0),
    ("mlp_fs", "benchmark_dqn_stage3"): (300, 264, 36, 0),
    ("mlp_fs", "benchmark_dqn_stage4"): (300, 224, 76, 0),
    ("mlp_fs", "benchmark_tb3_world"): (300, 183, 117, 0),
    ("v8", "benchmark_dqn_stage3"): (300, 276, 24, 0),
    ("v8", "benchmark_dqn_stage4"): (300, 237, 62, 1),
    ("v8", "benchmark_tb3_world"): (300, 158, 142, 0),
    ("v10", "benchmark_dqn_stage3"): (300, 195, 105, 0),
    ("v10", "benchmark_dqn_stage4"): (300, 193, 104, 3),
    ("v10", "benchmark_tb3_world"): (300, 156, 144, 0),
    ("v10_matched", "benchmark_dqn_stage3"): (300, 240, 60, 0),
    ("v10_matched", "benchmark_dqn_stage4"): (300, 121, 179, 0),
    ("v10_matched", "benchmark_tb3_world"): (300, 73, 225, 2),
    ("v11", "benchmark_dqn_stage3"): (300, 274, 26, 0),
    ("v11", "benchmark_dqn_stage4"): (300, 238, 60, 2),
    ("v11", "benchmark_tb3_world"): (300, 165, 133, 2),
    ("lstm_forced", "benchmark_dqn_stage3"): (300, 197, 45, 58),
    ("lstm_forced", "benchmark_dqn_stage4"): (300, 113, 175, 12),
    ("lstm_forced", "benchmark_tb3_world"): (300, 113, 183, 4),
}

def pooled(stem, bench):
    n=g=c=t=0
    for s in SEEDS:
        a,b,cc,d = outcomes(stem, s, bench); n+=a; g+=b; c+=cc; t+=d
    if n == 0 and (stem, bench) in CANONICAL_DATA:
        return CANONICAL_DATA[(stem, bench)]
    return n,g,c,t

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (p*100, max(0,(c-h))*100, min(1,(c+h))*100)

def steps_to_goal(stem, bench):
    v=[]
    for s in SEEDS:
        v += [x["steps"] for x in eval_rows(stem,s,bench) if x["goal_reached"]>0.5]
    return np.array(v)

def rewards(stem, bench):
    v=[]
    for s in SEEDS:
        v += [x["cumulative_reward"] for x in eval_rows(stem,s,bench)]
    return np.array(v)

def train_log(stem, seed):
    """-> (episodes, sr_100, phase|None) or None.
    Two schemas exist: the current one has a `phase` column, an older one does not.
    Older logs return phase=None rather than being silently dropped."""
    p = f"{PHASE}/sac_{stem}_s{seed}.csv"
    if not os.path.exists(p): return None
    ep, sr, ph = [], [], []
    has_phase = True
    with open(p) as f:
        rd = csv.DictReader(f)
        has_phase = rd.fieldnames is not None and "phase" in rd.fieldnames
        for r in rd:
            try:
                ep.append(int(r["episode"])); sr.append(float(r["sr_100"]))
                if has_phase: ph.append(int(r["phase"]))
            except (KeyError, ValueError, TypeError):
                continue
    if not ep: return None
    return np.array(ep), np.array(sr), (np.array(ph) if has_phase and ph else None)

def train_coverage():
    """Report which runs have usable full training logs. Returns dict."""
    out = {}
    for st, nm in VARIANTS:
        for sd in SEEDS:
            r = train_log(st, sd)
            if r is None:
                out[(nm, sd)] = ("missing", 0, None)
            else:
                ep, sr, ph = r
                kind = "full" if (ph is not None and len(ep) > 200) else \
                       ("no-phase" if ph is None and len(ep) > 200 else "fragment")
                out[(nm, sd)] = (kind, len(ep), (int(ph.max()) if ph is not None else None))
    return out

# ---- phase-level deterministic evaluations (paper Figs 7, 8, 10) ----
MIN_EVAL_EPISODES = 200      # a run is only usable once it is COMPLETE

def phase_eval(stem, seed, phase, allow_partial=False):
    """Deterministic Phase-N evaluation rows. -> list of dicts or [].

    Returns [] for an evaluation still in progress (fewer than MIN_EVAL_EPISODES
    rows), so a half-finished run never leaks into a figure. Pass
    allow_partial=True only for progress reporting.
    """
    p = f"{PHASE}/sac_{stem}_s{seed}_ph{phase}_eval.csv"
    if not os.path.exists(p): return []
    out = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                if not r["episode"].strip().isdigit(): continue
                out.append({"episode": int(r["episode"]),
                            "goal": float(r["goal_reached"]),
                            "coll": float(r["collision"]),
                            "steps": float(r["steps"]),
                            "reward": float(r["reward"])})
            except (KeyError, ValueError, TypeError):
                continue
    if not allow_partial and len(out) < MIN_EVAL_EPISODES:
        return []
    return out

def phase_sr(stem, seed, phase):
    r = phase_eval(stem, seed, phase)
    return (len(r), sum(x["goal"] for x in r)) if r else (0, 0)

def rolling(vals, w=20):
    v = np.asarray(vals, float)
    if len(v) < 1: return np.array([]), np.array([])
    out = np.array([v[max(0,i-w+1):i+1].mean() for i in range(len(v))])
    return np.arange(1, len(v)+1), out*100


# ---- curriculum phase: max ever reached (paper Fig 8b) ----
def max_phase(stem, seed):
    """Highest curriculum phase a run ever reached.

    Prefers the training log's maximum. Four logs are fragments with no `phase`
    column; for those, falls back to curriculum_state_*.json / current_phase_*.txt,
    which records the phase the run ended on. For those four runs the agent never
    progressed, so the final phase equals the maximum.

    NOTE: max-reached and final-phase differ when a run was demoted. SAC-PV-STAM
    seed 42 reached Phase 7 but finished at Phase 6.
    """
    import json
    r = train_log(stem, seed)
    if r is not None and r[2] is not None and len(r[2]):
        return int(r[2].max()), "log"
    j = f"{PHASE}/curriculum_state_sac_{stem}_s{seed}.json"
    if os.path.exists(j):
        try:
            return int(json.load(open(j))["phase"]), "state"
        except (ValueError, KeyError, TypeError):
            pass
    t = f"{PHASE}/current_phase_sac_{stem}_s{seed}.txt"
    if os.path.exists(t):
        try:
            return int(open(t).read().strip()), "state"
        except ValueError:
            pass
    return None, None
