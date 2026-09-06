# PV-STAM — Velocity-Aware Attention for Mapless DRL Navigation

Code, trained policies and evaluation data for the paper:

> **PV-STAM: Velocity-Aware Attention for Mapless Deep Reinforcement Learning Navigation in Dynamic Environments**
> Anas Mahyoub Naji Saeed Alqadhi, Munef El Muhammed, Mohammed Ali M. S. Bajhaw, Aysegul Ucar
> *Applied Sciences* (MDPI), 2026 · RAI Laboratory, Firat University

---

## What this is

Mapless navigation with a 2D LiDAR is hard because a single scan reports **where** an obstacle is, not **whether it is approaching**. PV-STAM is a compact perception block (**19,968 trainable parameters**, under 3% of network capacity) that combines a per-sector scan-difference channel with two-head self-attention over 24 LiDAR sectors and a 384 → 48 compression bottleneck.

**The paper's principal result is a dissociation.** Across three zero-shot benchmark arenas the three temporally-informed variants are statistically indistinguishable; across 130 physical trials on a TurtleBot3 Waffle Pi they separate with large margins. Simulation benchmarks of this kind did not have the resolution to rank policies that differ in how they handle sparse, asynchronous observations.

| Variant | Bench A | Bench B | Bench C | Hardware (standardised) |
|---|---|---|---|---|
| SAC-PV-STAM | 92.0% | 79.0% | 52.7% | 32.5% |
| SAC-R-PV-STAM | 91.3% | 79.3% | 55.0% | **97.5%** |
| SAC-MLP-FS | 88.0% | 74.7% | 61.0% | 65.0% |

Simulation figures pool three seeds at 100 episodes each (300 per cell). Hardware figures are directly standardised over two matched corridor scenarios.

---

## Repository layout

```
src/tb3_drl_nav/     ROS 2 package: agents, environments, launch files
  tb3_drl_nav/       SAC variants, environment and goal managers, eval nodes
  launch/            training, evaluation and benchmark launch files
  config/
models/              deployed actor weights used on the Jetson AGX Orin
data/
  evaluation/        per-episode CSVs: 45 benchmark runs + 63 curriculum-phase runs
  hardware/          trials_index.csv, bags.npz, Figure 2 arrays, attention weights
scripts/             figure-rendering scripts (Figures 1–12) and the shared data loader
```

Full training checkpoints (423 MB) are **not** in the repository — see [Trained models](#trained-models).

---

## Variant naming

Filenames throughout the code and data use short stems:

| Stem | Paper name | Temporal mechanism | Per-sector Δs |
|---|---|---|---|
| `baseline` | SAC-MLP | none (single frame) | yes |
| `mlp_fs` | SAC-MLP-FS | frame stack (k = 3) | no |
| `v8` | SAC-PV-STAM | frame stack (k = 3) | no |
| `v10` | SAC-PV-STAM-H | frame stack, Huber critic, width 384 | no |
| `v10_matched` | SAC-PV-STAM-H (256) | frame stack, Huber critic, width-matched | no |
| `v11` | SAC-R-PV-STAM | GRU (h = 64, 8-step burn-in) | yes |
| `lstm_forced` | SAC-LSTM | LSTM (h = 64), fixed step-budget advancement | no |

Benchmark arenas: `dqn_stage3` = **A** (open, static obstacles) · `dqn_stage4` = **B** (dynamic) · `tb3_world` = **C** (corridor). Seeds are 42, 777 and 123 throughout.

---

## Requirements

- Ubuntu 22.04 with **ROS 2 Humble**
- Gazebo Classic 11
- Python 3.10+, PyTorch 2.0+
- TurtleBot3 packages (`turtlebot3`, `turtlebot3_simulations`)

```bash
pip install -r requirements.txt
```

## Build

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/railab-firat/PV-STAM-mapless-DRL-navigation.git
cp -r PV-STAM-mapless-DRL-navigation/src/tb3_drl_nav .
cd ~/ros2_ws && colcon build --packages-select tb3_drl_nav
source install/setup.bash
```

## Train

```bash
ros2 launch tb3_drl_nav train_sac_stam.launch.py        # SAC-PV-STAM
ros2 launch tb3_drl_nav train_sac_v11.launch.py         # SAC-R-PV-STAM
ros2 launch tb3_drl_nav train_sac_v10_matched.launch.py # width-matched Huber
ros2 launch tb3_drl_nav train_sac_lstm_forced.launch.py # LSTM, forced advancement
```

Training follows a seven-phase curriculum with rolling-episode-success-rate gates and bidirectional demotion, scaling from static goal-seeking to fifteen simultaneously moving obstacles at 0.18 m/s. Phase 7 additionally activates a Hardware-Calibrated Training Mode (65 ms latency buffer, σ = 0.02 m/s velocity noise).

## Evaluate

```bash
ros2 launch tb3_drl_nav eval_canonical.launch.py   # three benchmark arenas
ros2 launch tb3_drl_nav eval_benchmark.launch.py
```

## Reproduce the paper's numbers

Every table in the paper can be recomputed from `data/`. For example, Table 6:

```python
import csv, glob, os, re, collections
SUF = {'dqn_stage3': 'A', 'dqn_stage4': 'B', 'tb3_world': 'C'}
agg = collections.defaultdict(lambda: [0, 0])
for f in glob.glob('data/evaluation/*benchmark*eval.csv'):
    m = re.match(r'sac_(.+?)_s(\d+)_benchmark_(\w+?)_eval\.csv', os.path.basename(f))
    rows = list(csv.DictReader(open(f)))
    a = agg[(m.group(1), SUF[m.group(3)])]
    a[0] += len(rows)
    a[1] += sum(1 for r in rows if r['goal_reached'] == '1')
for (variant, bench), (n, goals) in sorted(agg.items()):
    print(f'{variant:<12} {bench}  {goals}/{n} = {100*goals/n:.1f}%')
```

Figures are regenerated with the scripts in `scripts/` — `pv_data.py` is the shared loader.

**One caveat on the hardware behavioural metrics (Table 13):** each quantity is computed *per trial* and then averaged across trials with equal weight. Pooling all control samples instead gives different values.

---

## Trained models

`models/` contains the **deployed actor weights** (4.7 MB) actually used for the physical trials on the Jetson AGX Orin — enough to reproduce the hardware behaviour.

Full training checkpoints, including critic networks and optimiser state (423 MB), are attached to the [Releases](../../releases) page rather than tracked in git, since one exceeds GitHub's 100 MB file limit.

---

## Data

`data/evaluation/` — one row per episode. Success is `goal_reached == 1`; `collision` marks a threshold violation; `terminal_type` gives the termination reason.

`data/hardware/` — `trials_index.csv` indexes all 143 recorded physical trials (130 analysed, 13 excluded with documented cause). `bags.npz` holds six arrays per trial keyed `<trial_id>|<array>`: `ox, oy, ot` (odometry), `ct` (timestamps), `cl` (commanded linear velocity), `ca` (commanded angular velocity).

Note that `min_clearance_m` is the minimum LiDAR range to **any** surface, walls included — not only to the obstacle.

---

## Terminology

Following the paper: a **threshold violation** is a LiDAR reading below the applicable threshold that terminated the trial; a **safety-zone intrusion** is entry into the danger zone without termination. Neither implies measured physical contact, which was not instrumented in either setting.

The scan-difference channel Δsᵢ is a **frame-to-frame range difference**, not an obstacle velocity. It is not divided by the sampling interval and is affected by the robot's own translation and rotation, by sector discretisation, by occlusion and by changes in the reflecting surface.

---

## Citing

See [`CITATION.cff`](CITATION.cff), or use the "Cite this repository" button on GitHub.

## Licence

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

Supported by The Scientific and Technological Research Council of Türkiye (TÜBİTAK), grant 123E406, and by Firat University Scientific Research Projects Unit (FÜBAP), grants MF.24.80, MF.25.154 and MF.25.155. Part of this work was supported within the TÜBİTAK 2209-A programme.
