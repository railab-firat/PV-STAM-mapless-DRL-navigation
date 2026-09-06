HARDWARE TRIALS DATASET README
==============================

trials_index.csv:
-----------------
Index of all 143 physical hardware trials conducted on the TurtleBot3 Waffle Pi platform
in the 3.3 x 7.6 m corridor setup.
- decision: "saved" (130 valid analyzed trials), or one of the exclusion categories
  "excluded_dualnode" (5), "excluded_envchange" (4) and "archived" (4), 13 exclusions in total.
- model: baseline (SAC-MLP), mlp_fs (SAC-MLP-FS), v8 (SAC-PV-STAM),
         v10 (SAC-PV-STAM-H, critic width 384), v11 (SAC-R-PV-STAM)
  Two further variants appear in the simulation datasets (folders 02 and 03) but were not
  deployed on hardware:
         v10_matched (SAC-PV-STAM-H, width-matched, critic width 256)
         lstm_forced (SAC-LSTM retrained under fixed step-budget phase advancement)
- scenario: 2 (static obstacles) or 3 (moving obstacles)
- outcome: goal, collision, or timeout
- min_clearance_m: minimum LiDAR range recorded during the trial, to ANY surface, walls
  included. It is not restricted to the obstacle. A reading below 0.20 m that terminated the
  trial is a threshold violation; a reading below 0.20 m in a trial that continued is a
  non-terminating safety-zone intrusion. Two SAC-R-PV-STAM trials record 0.160 m and 0.120 m
  and both reached the goal, so they are intrusions and not violations.

bags.npz:
---------
Numpy archive with six arrays per trial, keyed "<trial_id>|<array>":
  ox, oy, ot  - odometry x (m), y (m) and heading (rad)
  ct          - timestamp of each control step (s)
  cl          - commanded linear velocity (m/s)
  ca          - commanded angular velocity (rad/s)
130 valid trials x 6 arrays = 780 entries.

Note on reproducing Table 13: each quantity is computed per trial and then averaged across
trials with equal weight on each trial. Pooling all control samples instead gives different
values (for example SAC-MLP-FS forward-command-active reads 45.3% pooled against 52.0%
per-trial).

fig2abcd.npz / fig2ef.npz:
--------------------------
Source arrays for Figure 2. fig2abcd.npz holds the simulation analysis (lo_/hi_ prev, cur and
ds vectors, plus lo_rate 1.9865 and hi_rate 4.5117 spurious spikes per timestep over
lo_n = 20,723 and hi_n = 41,590 transitions, a ratio of 2.27). fig2ef.npz holds the same
analysis computed from the 130 physical trials under identical thresholds (lo_rate 0.1492,
hi_rate 1.7908 over lo_n = 27,461 and hi_n = 3,672 control cycles, a ratio of 12.0).

attention.npz:
--------------
PV-STAM sector-attention weights captured in simulation for four representative situations:
Open_space, Obstacle_ahead, Obstacle_approaching_(left) and Corridor. Each entry is a
24-element vector over the LiDAR sectors, normalized to sum to 1.0.

attention_real.npz:
-------------------
Deployment attention weights for four representative situations, named in _order:
Clear Path, Head-on Obstacle, Left Approach and Multiple Threats. Each situation has three
24-element vectors, keyed "<situation>|<array>":
  attn  - sector attention weights (sum to 1.0)
  scan  - proximity-amplified sector distances
  ds    - per-sector scan-difference channel
Also contains _align and _ratio summary series over 2,864 deployment control steps.

Angular velocity was NOT logged alongside these attention weights. The rotation
down-weighting mitigation discussed in Section 3.1 of the manuscript therefore cannot be
tested from these files, which is why the manuscript reports it as an untested design
argument rather than a result.

show_results.py:
----------------
Convenience script for loading and printing the arrays above.
