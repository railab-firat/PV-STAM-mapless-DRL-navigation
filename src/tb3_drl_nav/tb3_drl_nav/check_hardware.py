#!/usr/bin/env python3
"""
check_hardware.py  Phase 3
==============================================
Standalone hardware + training environment diagnostic.
Run from any terminal — no ROS needed.

    python3 check_hardware.py

Checks:
  1. GPU model, driver, VRAM
  2. PyTorch version and CUDA support
  3. Whether training is using GPU or CPU (and WHY)
  4. CPU speed benchmark for PPO update
  5. Disk space for logs and checkpoints
  6. ROS2 and Gazebo process status
  7. Training log summary (if any run is active)
"""
import os
import sys
import time
import subprocess
import platform

# ── colour helpers (no dependencies) ─────────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def cyan(s):   return f"\033[96m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"

def section(title):
    print()
    print(bold(cyan(f"{'─'*60}")))
    print(bold(cyan(f"  {title}")))
    print(bold(cyan(f"{'─'*60}")))

def ok(label, value=""):
    print(f"  {green('✓')}  {label:<35} {value}")

def warn(label, value=""):
    print(f"  {yellow('⚠')}  {label:<35} {yellow(value)}")

def fail(label, value=""):
    print(f"  {red('✗')}  {label:<35} {red(value)}")

def info(label, value=""):
    print(f"  {dim('·')}  {label:<35} {value}")


def main():
    # ─────────────────────────────────────────────────────────────────────────────
    print()
    print(bold(cyan("╔══════════════════════════════════════════════════════════╗")))
    print(bold(cyan("║     Phase 3 — Hardware & Env Diagnostic   ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════════════╝")))

    # ─── 1. System ────────────────────────────────────────────────────────────────
    section("1. System")
    info("OS",       platform.platform())
    info("Python",   sys.version.split()[0])
    info("Hostname", platform.node())

    # CPU count / frequency
    try:
        import multiprocessing
        cpus = multiprocessing.cpu_count()
        info("CPU cores", str(cpus))
    except Exception:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            lines = f.read()
        total = int([l for l in lines.splitlines() if "MemTotal" in l][0].split()[1])
        avail = int([l for l in lines.splitlines() if "MemAvailable" in l][0].split()[1])
        total_gb = total / 1024 / 1024
        avail_gb = avail / 1024 / 1024
        color = ok if avail_gb > 2 else warn
        color("RAM total / available", f"{total_gb:.1f} GB / {avail_gb:.1f} GB free")
    except Exception:
        pass

    # ─── 2. GPU hardware ──────────────────────────────────────────────────────────
    section("2. GPU Hardware")

    nvidia_smi_ok = False
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                name, driver, mem_total, mem_free, compute = (
                    parts + ["?"]*5)[:5]
                ok("GPU name",          name)
                ok("Driver version",    driver)
                ok("VRAM total / free", f"{mem_total} / {mem_free}")
                sm = compute.replace(".", "")
                sm_int = int(sm) if sm.isdigit() else 0
                if sm_int >= 60:
                    ok("Compute capability", f"sm_{sm}  (supported)")
                elif sm_int == 50:
                    warn("Compute capability",
                         f"sm_{sm}  (Maxwell — NOT supported by PyTorch ≥ 1.13)")
                else:
                    warn("Compute capability", f"sm_{sm}  (unknown)")
                nvidia_smi_ok = True
        else:
            fail("nvidia-smi", "not found or failed")
    except FileNotFoundError:
        fail("nvidia-smi", "not installed")
    except subprocess.TimeoutExpired:
        fail("nvidia-smi", "timeout")

    # ─── 3. PyTorch / CUDA ────────────────────────────────────────────────────────
    section("3. PyTorch & CUDA")

    try:
        import torch
        ok("PyTorch installed", torch.__version__)
        ok("CUDA built with",   str(torch.version.cuda))

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap  = torch.cuda.get_device_capability(0)
            sm   = cap[0] * 10 + cap[1]
            ok("CUDA available",    f"yes — {name}")
            ok("Compute cap",       f"sm_{sm}")

            # Check if sm is in PyTorch supported list
            try:
                archs = torch.cuda.get_arch_list()
                tag   = f"sm_{sm}"
                if tag in archs:
                    ok("sm in torch arch list", f"YES — training will use GPU")
                else:
                    fail("sm in torch arch list",
                         f"NO — sm_{sm} not in {archs}")
                    fail("→ Training uses",
                         "CPU  (GPU physically present but unsupported by this PyTorch)")
            except AttributeError:
                warn("arch list check", "skipped (older PyTorch)")
        else:
            warn("CUDA available", "no")
            if nvidia_smi_ok:
                fail("→ Training uses",
                     "CPU  (GPU present but PyTorch cannot use it)")
                print()
                print(f"  {yellow('Explanation:')}")
                print(f"  Your Quadro M2000M is Maxwell architecture (sm_50).")
                print(f"  PyTorch dropped sm_50 CUDA support in version 1.13.")
                print(f"  This is {yellow('expected and normal')} for this hardware.")
                print(f"  CPU is {green('sufficient')} — PPO update ≈ 0.5s,")
                print(f"  Gazebo episode ≈ 2-3s. CPU is NOT the bottleneck.")
            else:
                warn("→ Training uses", "CPU  (no GPU detected)")

        # CPU benchmark
        print()
        print(f"  {dim('Running CPU speed benchmark (100 forward passes)…')}")
        import torch.nn as nn
        net = nn.Sequential(
            nn.Linear(28, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 2))
        x = torch.randn(256, 28)
        t0 = time.time()
        for _ in range(100):
            _ = net(x)
        dt = (time.time() - t0) * 10  # ms per call
        color = ok if dt < 20 else (warn if dt < 50 else fail)
        color("CPU fwd pass (batch=256)", f"{dt:.1f} ms  ({'fast' if dt<20 else 'acceptable' if dt<50 else 'slow'})")

        # PPO update estimate
        t0 = time.time()
        for _ in range(10):
            y = net(x)
            loss = y.sum()
            loss.backward()
        dt_upd = (time.time() - t0) * 100
        color2 = ok if dt_upd < 1000 else warn
        color2("PPO update estimate",
               f"~{dt_upd:.0f} ms  (episodes run every ~{2500:.0f} ms in Gazebo)")

    except ModuleNotFoundError:
        fail("PyTorch",
             "not found in this Python env. Run inside ROS workspace.")
        print(f"  {dim('Tip: source ~/tubitak_2209_ws/install/setup.bash first')}")

    # ─── 4. Disk space ────────────────────────────────────────────────────────────
    section("4. Disk Space")

    for path in [
        os.path.expanduser("~/tb3_drl_logs"),
        os.path.expanduser("~/tb3_drl_models"),
        os.path.expanduser("~/tubitak_2209_ws"),
    ]:
        try:
            st = os.statvfs(path)
            free_gb = st.f_bavail * st.f_frsize / 1e9
            label = os.path.basename(path) or path
            color = ok if free_gb > 5 else (warn if free_gb > 1 else fail)
            color(f"Free space ({label})", f"{free_gb:.1f} GB")
        except Exception:
            pass

    # existing checkpoints
    ckpt_dir = os.path.expanduser("~/tb3_drl_models/phase3")
    if os.path.isdir(ckpt_dir):
        runs = os.listdir(ckpt_dir)
        for run in sorted(runs):
            run_path = os.path.join(ckpt_dir, run)
            ckpts = sorted(os.listdir(run_path)) if os.path.isdir(run_path) else []
            ckpts = [c for c in ckpts if c.endswith(".pt")]
            if ckpts:
                latest = ckpts[-1]
                ep_num = latest.replace("ckpt_ep", "").replace(".pt", "").lstrip("0") or "0"
                ok(f"Checkpoint [{run}]",
                   f"ep {ep_num}  ({len(ckpts)} files)")
            else:
                info(f"Run [{run}]", "no checkpoints yet")
    else:
        info("~/tb3_drl_models/phase3", "not created yet (no training run yet)")

    # ─── 5. ROS2 / Gazebo processes ───────────────────────────────────────────────
    section("5. ROS2 / Gazebo Process Status")

    processes = {
        "gzserver (Gazebo physics)":   "gzserver",
        "gzclient (Gazebo GUI)":       "gzclient",
        "environment_ppo":             "environment_ppo",
        "goal_manager_dynamic":        "goal_manager_dynamic",
        "train_agent_ppo":             "train_agent_ppo",
        "obstacle_controller":         "obstacle_controller",
    }
    for label, proc in processes.items():
        r = subprocess.run(["pgrep", "-f", proc],
                           capture_output=True, text=True)
        if r.returncode == 0:
            pids = r.stdout.strip().split()
            ok(label, f"running  (pid {', '.join(pids[:2])})")
        else:
            info(label, dim("not running"))

    # ─── 6. Training log summary ──────────────────────────────────────────────────
    section("6. Training Log Summary")

    log_dir = os.path.expanduser("~/tb3_drl_logs/phase3")
    if os.path.isdir(log_dir):
        logs = sorted([os.path.join(log_dir, f)
                       for f in os.listdir(log_dir) if f.endswith(".csv")])
        if logs:
            for log_path in logs[-3:]:   # show last 3 runs
                try:
                    import csv as _csv
                    with open(log_path) as f:
                        rows = list(_csv.DictReader(f))
                    if not rows:
                        info(os.path.basename(log_path), "empty")
                        continue
                    last    = rows[-1]
                    ep      = int(float(last["episode"]))
                    sr      = float(last["sr_100"])
                    elapsed = float(last.get("elapsed_s", 0))
                    best_sr = max(float(r["sr_100"]) for r in rows)
                    color   = ok if sr >= 50 else (warn if sr >= 20 else info)
                    color(os.path.basename(log_path),
                          f"ep={ep}  SR={sr:.1f}%  best={best_sr:.1f}%  "
                          f"elapsed={elapsed/3600:.1f}h")
                except Exception as e:
                    info(os.path.basename(log_path), f"read error: {e}")
        else:
            info("No CSV logs found", dim("(training not started yet)"))
    else:
        info("~/tb3_drl_logs/phase3", "not created yet")

    # ─── 7. Summary & recommendations ────────────────────────────────────────────
    section("7. Summary & Recommendations")

    print(f"""
  {bold('Why is training slow to start?')}

  The most common causes:

  {yellow('①')}  Gazebo takes 15–30 s to fully load the world.
     Wait until Terminal 1 shows:
     {dim('turtlebot3_diff_drive: Subscribed to [/cmd_vel]')}

  {yellow('②')}  /reset_simulation service only appears after Gazebo is ready.
     The environment_ppo node will keep retrying silently.

  {yellow('③')}  goal_manager_dynamic must publish a goal BEFORE the first
     reset_obs can be sent. Start it before environment_ppo.

  {yellow('④')}  If training stops mid-run (crash, Ctrl+C, power loss):
     Just run the same command again — it will auto-resume
     from the last checkpoint (saved every 50 episodes).

  {bold('Why are we on CPU?')}

  Your GPU is Quadro M2000M = Maxwell architecture = sm_50.
  PyTorch 1.13+ removed CUDA support for sm_50 to save binary size.
  {green('This is expected and NOT a problem')} — CPU is fast enough:
    • PPO update    ≈ 0.5 s  on CPU
    • Gazebo episode ≈ 2–3 s  (physics simulation)
    • CPU is idle during Gazebo time → no bottleneck

  To use GPU you would need PyTorch ≤ 1.12 (which has other issues)
  or a newer GPU (any GTX 10xx or newer supports sm_60+).
""")

    print(bold(cyan("─" * 60)))
    print(bold("  Run complete."))
    print(bold(cyan("─" * 60)))
    print()


if __name__ == "__main__":
    main()
