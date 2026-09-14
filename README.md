# PV-STAM — Velocity-Aware Attention for Mapless DRL Navigation

Official implementation, trained policy checkpoints, ROS 2 packages, and raw evaluation datasets for the paper:

> **PV-STAM: Velocity-Aware Attention for Mapless Deep Reinforcement Learning Navigation in Dynamic Environments**  
> **Anas Alqadhi**  
> **RAI Laboratory, Firat University**, 2026

---

## 📌 Executive Summary

Mapless navigation using 2D LiDAR is fundamentally challenging because a single scan reports **where** an obstacle is, but cannot distinguish whether an obstacle is stationary or rapidly approaching. **PV-STAM** (Positional-Velocity Spatio-Temporal Attention Module) is an ultra-compact perception block (**19,968 trainable parameters**, under 3% of policy capacity) that pairs a per-sector scan-difference channel ($\Delta s_i$) with multi-head self-attention over 24 LiDAR sectors and a linear compression bottleneck ($384 \rightarrow 48$).

<p align="center">
  <img src="docs/figures/Figure_01_System_Pipeline_Overview.png" alt="System Pipeline Overview" width="95%"/>
  <br/>
  <em>Figure 1: End-to-end system architecture pipeline showing 2D LiDAR sectorization, dual-channel feature extraction, PV-STAM attention processing, and SAC actor-critic policy execution.</em>
</p>

### 🔑 Key Results & Findings

* **Simulation/Hardware Dissociation:** Across 3 zero-shot benchmark simulation arenas (Open, Dynamic, Corridor), the three leading temporally-informed variants are statistically indistinguishable ($92.0\%$ vs $91.3\%$ vs $88.0\%$). However, across **130 physical TurtleBot3 trials**, they separate with large margins (**$97.5\%$** for SAC-R-PV-STAM vs $65.0\%$ for SAC-MLP-FS and $32.5\%$ for SAC-PV-STAM).
* **Rotation Contamination Mitigation:** Evaluated across 27,461 physical scan frames, unmitigated frame-stacking experiences a **$12.0\times$ rotation-contamination ratio** during turns ($\omega \ge 0.1\text{ rad/s}$), creating false motion signals. Sector attention and per-sector scan-difference eliminate this artifact without requiring scan registration.

---

## 🏗️ Module Architecture

<p align="center">
  <img src="docs/figures/Figure_03_PVSTAM_Module_Architecture.png" alt="PV-STAM Module Architecture" width="98%"/>
  <br/>
  <em>Figure 3: PV-STAM module parameter breakdown (19,968 trainable parameters; 19,952 for C = 2 recurrent variant). Over 92.5% of parameters reside in the linear compression bottleneck.</em>
</p>

### Parameter Breakdown by Layer

| Component / Layer Name | Mathematical Symbol | Tensor Shape | Parameters | Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Learned Positional Encoding** | $\mathbf{E}_{\mathrm{pos}}$ | $24 \times 16$ | 384 | 1.9% |
| **Input Projection Weight** | $\mathbf{W}_{\mathrm{in}}$ | $16 \times 3$ | 48 | 0.2% |
| **Input Projection Bias** | $\mathbf{b}_{\mathrm{in}}$ | $16$ | 16 | 0.1% |
| **QKV Self-Attention Projection** | $\mathbf{W}_{QKV}$ | $48 \times 16$ | 768 | 3.8% |
| **Attention Output Weight** | $\mathbf{W}_{\mathrm{out}}$ | $16 \times 16$ | 256 | 1.3% |
| **Attention Output Bias** | $\mathbf{b}_{\mathrm{out}}$ | $16$ | 16 | 0.1% |
| **Linear Compression Weight** | $\mathbf{W}_{\mathrm{comp}}$ | $48 \times 384$ | 18,432 | 92.3% |
| **Linear Compression Bias** | $\mathbf{b}_{\mathrm{comp}}$ | $48$ | 48 | 0.2% |
| **Total PV-STAM Module** | | | **19,968** | **100.0%** |

---

## 🔄 Rotation Contamination Analysis

<p align="center">
  <img src="docs/figures/Figure_02_Scan_Difference_Contamination.png" alt="Rotation Contamination Analysis" width="95%"/>
  <br/>
  <em>Figure 2: Empirical rotation contamination analysis across 130 physical TurtleBot3 trials, comparing frame-difference signal ratios during pure translation vs rotation.</em>
</p>

---

## 🎓 Progressive 7-Phase Curriculum

<p align="center">
  <img src="docs/figures/Figure_04_Seven_Phase_Progressive_Curriculum.png" alt="Seven Phase Curriculum Flowchart" width="95%"/>
  <br/>
  <em>Figure 4: Flowchart of the progressive 7-phase curriculum advancing agents through static obstacles, dynamic SFM pedestrians, sensor noise, and hardware control latency.</em>
</p>

---

## 🤖 Real-Robot Physical Evaluation

<p align="center">
  <img src="docs/figures/Figure_11_Physical_Evaluation_Corridor_Setup.png" alt="Physical Evaluation Corridor Setup" width="95%"/>
  <br/>
  <em>Figure 11: Real-world experimental corridor (3.3 × 7.6 m) setup with TurtleBot3 Waffle Pi evaluating Scenario 2 (static obstacles) and Scenario 3 (moving obstacles).</em>
</p>

### Hardware Evaluation Outcomes (130 Trials)

| Variant | Scenario 2 (Static) | Scenario 3 (Moving) | Overall Standardized Rate | Threshold Violations | Timeouts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SAC-R-PV-STAM** | **19 / 20 (95.0%)** | **20 / 20 (100.0%)** | **97.5%** | **0 / 40** | **1** |
| **SAC-MLP-FS** | 14 / 20 (70.0%) | 12 / 20 (60.0%) | 65.0% | 3 / 40 | 11 |
| **SAC-PV-STAM** | 1 / 10 (10.0%) | 11 / 20 (55.0%) | 32.5% | 9 / 30 | 10 |
| **SAC-MLP** | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0.0% | 0 / 20 | 20 |

---

## 🛠️ Repository Structure

```
PV-STAM-mapless-DRL-navigation/
├── data/
│   ├── evaluation/         # 45 canonical benchmark evaluation CSV files (300 episodes/cell)
│   └── hardware/           # 130 physical TurtleBot3 trial arrays (bags.npz) & attention files
├── docs/
│   └── figures/            # High-resolution publication figures & diagrams (Figures 1–12)
├── models/                 # Deployed policy checkpoints for hardware execution
├── scripts/                # Python verification & plotting scripts for Figures 1–12
└── src/
    └── tb3_drl_nav/        # ROS 2 package (environments, controllers, SAC agents)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# ROS 2 Humble / Foxy & Gazebo Simulation
sudo apt install ros-${ROS_DISTRO}-turtlebot3* ros-${ROS_DISTRO}-gazebo-ros-pkgs

# Install Python Dependencies
pip install -r requirements.txt
```

### Evaluation & Reproduction

To reproduce all publication figures from the canonical data:

```bash
# Recompute hardware statistics & Table 13 signatures
python data/hardware/show_results.py

# Re-render Figures 1–12
python scripts/f01_benchmarks.py
python scripts/f08_hardware.py
python scripts/s01_pvstam_module.py
```

---

## 📜 Citation

If you find PV-STAM useful for your research, please cite:

```bibtex
@article{alqadhi2026pvstam,
  title={PV-STAM: Velocity-Aware Attention for Mapless Deep Reinforcement Learning Navigation in Dynamic Environments},
  author={Alqadhi, Anas},
  journal={RAI Laboratory, Firat University},
  year={2026}
}
```

---

## 📄 License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
