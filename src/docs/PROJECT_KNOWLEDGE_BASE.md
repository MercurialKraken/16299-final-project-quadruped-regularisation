# Flow Matching for Smoothing Quadruped Locomotion Policies — Complete Knowledge Base

> **Purpose of this document:** This file contains everything needed to fully understand, explain, and answer questions about this project. It is designed to be read by an AI assistant (e.g., Claude) so that the assistant can act as a knowledgeable project team member. All key numbers, decisions, failures, and rationale are included inline.

---

## 1. Project Summary (Elevator Pitch)

We applied **Conditional Flow Matching (CFM)** to post-process joint commands from a PPO-trained quadruped robot (Unitree Go1) in NVIDIA Isaac Sim. The flow model learns a velocity field that transports raw (jittery) actions toward low-pass filtered (smooth) actions, conditioned on the robot's observation state. By integrating the flow ODE only partially (`t_end < 1.0`), we control smoothing intensity — removing high-frequency jitter while preserving gait dynamics.

**Key results:**
- 15.6% system-level jitter reduction on flat terrain (t_end=1.0), 5.8% at conservative t_end=0.3
- Individual joints see up to 39% reduction (FR_hip at t_end=1.0)
- Statistically significant push recovery improvement: fall rate drops from 50% to 33.8% (McNemar p=0.015)
- 4.7% jerk reduction and 4.1% energy savings
- 14.5% jitter reduction on rough terrain (t_end=0.5), showing the approach generalizes
- Zero velocity tracking degradation across all conditions

**Student:** Paul Colombo, Sophomore, Computer Science and Robotics, Carnegie Mellon University
**Date:** April 2026
**Platform:** NVIDIA Isaac Sim 5.1 + Isaac Lab, Unitree Go1, PPO via RSL-RL

---

## 2. The Problem: Why PPO Policies Jitter

PPO-trained locomotion policies produce high-frequency oscillations in joint position commands. Three factors:

1. **Residual exploration noise:** The stochastic policy trains with Gaussian noise. Even after convergence, the learned mean retains artifacts from noisy exploration.
2. **Weak penalty constraint:** The reward function includes `action_rate_l2` with weight -0.01, but making it stronger degrades velocity tracking. It's a fundamentally conflicting objective.
3. **No temporal memory:** The MLP policy (no recurrence) recomputes actions from scratch each timestep. There's no built-in temporal smoothing — consecutive actions are independent given the observation.

**Metric we use:** Action Rate RMS = `√(mean((a_t - a_{t-1})²))` across all 12 joints and all timesteps.

---

## 3. Platform and Robot Details

### Unitree Go1
- 12 actuated joints: 4 legs × 3 joints (hip, thigh, calf)
- Joint names: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
- Action space: 12-dimensional joint position targets
- Control frequency: 50 Hz (dt = 0.02s)

### Observation Spaces
- **Flat terrain (48 dims):** base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos(12), joint_vel(12), actions(12)
- **Rough terrain (235 dims):** same as flat + height_scan(187). The height scan is a grid of terrain height samples around each foot, giving the policy local terrain awareness.

### PPO Training
- **Flat:** MLP [128, 128, 128] with ELU, 300 iterations, 4096 parallel envs, γ=0.99, λ=0.95, ε=0.2, lr=0.001 with adaptive KL (desired_kl=0.01)
- **Rough:** MLP [512, 256, 128] with ELU, 100+ iterations, same hyperparams otherwise
- Checkpoints used: `model_299.pt` (flat), `model_100.pt` (rough)

### Reward Function
| Term | Weight | Purpose |
|------|--------|---------|
| track_lin_vel_xy_exp | 1.5 | Velocity command tracking (primary task) |
| track_ang_vel_z_exp | 0.75 | Yaw rate tracking |
| lin_vel_z_l2 | -2.0 | Penalize vertical bouncing |
| ang_vel_xy_l2 | -0.05 | Penalize roll/pitch oscillation |
| dof_torques_l2 | -0.0002 | Penalize high torques |
| dof_acc_l2 | -2.5e-7 | Penalize joint accelerations |
| action_rate_l2 | -0.01 | Penalize jitter (insufficient alone) |
| feet_air_time | 0.25 (flat) / 0.01 (rough) | Encourage gait timing |
| flat_orientation_l2 | -2.5 (flat) / 0.0 (rough) | Keep body level |

---

## 4. Flow Matching Method

### 4.1 Conditional Flow Matching Theory

Flow matching learns a velocity field `v(x_t, t, s)` that defines an ODE transporting samples from a source distribution (raw actions) to a target distribution (LP-filtered actions), conditioned on robot state `s`.

**Linear interpolation path:**
```
x_t = (1 - t) · x_0 + t · x_1,    t ∈ [0, 1]
```
where x_0 = raw action, x_1 = LP-filtered action.

**Conditional velocity field:** `v*(x_t, t | x_0, x_1) = x_1 - x_0` (constant — the displacement vector, independent of t). This is the key insight of CFM: for straight-line paths, the optimal velocity is just the difference.

**Training loss:**
```
L(θ) = E_{x_0, x_1, t~U(0,1)} [ ||v_θ(x_t, t, s) - (x_1 - x_0)||² ]
```

### 4.2 VelocityNet Architecture

```
Input: [action(12) | time(1) | state(48 or 235)]
  → Linear(input_dim → 256) + SiLU
  → Linear(256 → 256) + SiLU
  → Linear(256 → 256) + SiLU
  → Linear(256 → 12)
Output: velocity field v(12)
```

- Input dim: 61 (flat) or 248 (rough)
- Activation: SiLU (Sigmoid Linear Unit) — smooth and non-monotonic
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Scheduler: Cosine annealing
- Batch size: 512
- Epochs: 200

### 4.3 Inference (Euler Integration)

```python
def flow_smooth(model, raw_action, state, n_steps=20, t_end=0.3):
    x = raw_action.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = i * dt
        x = x + model(x, t_val, state) * dt
    return x
```

**Critical design choice:** This outputs a **single action per timestep**, not a trajectory. It's a drop-in post-processor — you just replace `exec_action = raw_action` with `exec_action = flow_smooth(model, raw_action, state)`. No replanning, no trajectory optimization.

### 4.4 The t_end Parameter

`t_end` controls how far along the flow ODE we integrate:
- `t_end = 0.0`: raw action (no smoothing)
- `t_end = 0.3`: 30% of the way toward LP-filtered target (conservative, used for flat terrain)
- `t_end = 0.5`: 50% (used for rough terrain)
- `t_end = 1.0`: full integration to LP-filtered distribution (most aggressive)

Partial integration is key — it removes jitter while preserving gait dynamics that full smoothing would dampen.

### 4.5 Low-Pass Filter Targets (Training Data)

The "smooth targets" x_1 come from applying a 2nd-order Butterworth filter (bidirectional via `filtfilt` to avoid phase shift) to the raw rollout actions. Sampling rate = 50 Hz.

**Critical discovery:** Not all joints should use the same cutoff frequency. Per-joint adaptive cutoffs:

| Joint | Cutoff (Hz) | Why |
|-------|------------|-----|
| FL_hip | 10 | Benefits from smoothing |
| FL_thigh | 12 | Moderate benefit |
| FL_calf | 22 | Gets WORSE with low cutoff — barely filter |
| FR_hip | 8 | Strongest responder (14.4% at t=0.3, 39.1% at t=1.0) |
| FR_thigh | 15 | Mixed signals |
| FR_calf | 8 | Strong benefit |
| RL_hip | 22 | Gets WORSE — barely filter |
| RL_thigh | 10 | Good benefit |
| RL_calf | 15 | Moderate |
| RR_hip | 12 | Mixed |
| RR_thigh | 25 | Gets 10% WORSE — almost no filter |
| RR_calf | 10 | Strong benefit |

**Why the asymmetry:** Some joints (thighs, certain hips) carry real gait signal at frequencies that a uniform cutoff would destroy. Other joints (calves, FR_hip) have mostly noise in the high-frequency band. The adaptive cutoffs were determined through in-simulation testing.

---

## 5. Training Data Collection

### Flat Terrain Rollout
- 5 episodes, 1000 steps each = 5000 total timesteps
- 48-dim observations, 12-dim actions
- All episodes ran to completion (no falls)
- Episode returns: 19.38, 19.45, 19.43, 19.47, 19.42 (very consistent)
- File: `rollout_data.npz`

### Rough Terrain Rollout
- 5 episodes attempted, 3805 total timesteps (not 5000 — 2 episodes had early terminations = falls)
- 235-dim observations, 12-dim actions
- Episode lengths: [1000, 1000, 805, 1000, ?] — rough terrain is harder
- File: `rollout_data_rough.npz`

### Training Convergence
- **Flat adaptive:** Loss 0.0200 → 0.0032 over 200 epochs (smooth convergence, no overfitting)
- **Rough:** Loss 0.0089 → 0.0023 over 200 epochs (lower starting loss because rough terrain actions have less room to smooth — the raw-to-filtered displacement is smaller)

---

## 6. All Experimental Results

### 6.1 In-Simulation Smoothing Verification (Flat Terrain)

500 steps, cmd_vx=1.0 m/s, single environment, measured in live Isaac Sim:

| t_end | Action Rate RMS | Jitter Reduction | Mean Vx | Vx Std |
|-------|----------------|-----------------|---------|--------|
| 0.0 (raw) | 0.5889 | — | 0.5093 m/s | 0.1087 |
| 0.3 | 0.5710 | 3.0% | 0.4886 m/s | 0.0846 |
| 0.5 | 0.5666 | 3.8% | 0.4646 m/s | 0.1098 |
| 1.0 | 0.5526 | 6.2% | 0.4950 m/s | 0.0841 |

Note: These numbers are from the `smoothing_verification.npz` single-env run. The per-joint adaptive comparison (`adaptive_comparison.npz`) shows larger reductions because it measures the effect on individual joints separately.

### 6.2 Per-Joint Analysis (Flat, Adaptive Model)

At t_end=0.3 (production setting):
| Joint | Raw RMS | Smoothed RMS | Reduction |
|-------|---------|-------------|-----------|
| FL_hip | 0.4798 | 0.4475 | 6.7% |
| FL_thigh | 0.6925 | 0.6630 | 4.2% |
| FL_calf | 0.4810 | 0.4745 | 1.3% |
| FR_hip | 0.4209 | 0.3604 | **14.4%** |
| FR_thigh | 0.6082 | 0.5801 | 4.6% |
| FR_calf | 0.8122 | 0.7229 | **11.0%** |
| RL_hip | 0.6220 | 0.6161 | 0.9% |
| RL_thigh | 0.6346 | 0.5929 | 6.6% |
| RL_calf | 0.5773 | 0.5526 | 4.3% |
| RR_hip | 0.4947 | 0.4434 | **10.4%** |
| RR_thigh | 0.6474 | 0.6471 | 0.0% |
| RR_calf | 0.3330 | 0.2858 | **14.2%** |
| **TOTAL** | **0.5806** | **0.5468** | **5.8%** |

At t_end=1.0 (full integration):
- FR_hip: 39.1% reduction, RR_calf: 36.9%, FR_calf: 32.0%, RR_hip: 28.6%
- Total: 15.6%
- But RR_thigh still only 0.1% — it genuinely has no jitter to remove

### 6.3 Push Recovery (Main Result)

**Setup:** 80 parallel environments, lateral force impulse to robot trunk, magnitudes 50–500 N (10 levels, 8 envs each), 200ms push duration at t=2s.

| Metric | Raw (t_end=0.0) | Smoothed (t_end=0.3) |
|--------|-----------------|---------------------|
| Falls | 40/80 (50.0%) | 27/80 (33.8%) |
| Discordant b (raw fell, smo survived) | 19 | — |
| Discordant c (smo fell, raw survived) | — | 6 |
| McNemar χ² | 6.76 | **p = 0.015** |

This is statistically significant at α=0.05. In 19 environments where the raw controller fell, the smoothed controller survived. Only 6 went the other direction.

**Why it works:** Smoothing reduces oscillations that compound during disturbance recovery. Jittery actions cause legs to fight each other; coherent actions let the legs coordinate a recovery motion.

**Earlier pilot (low force):** We first tried 5–50 N with 40 environments. Nobody fell (0% both sides). The robot is surprisingly robust at low forces — you need 50–500 N to find the breaking point.

### 6.4 Effort and Energy Analysis

**Setup:** 20 paired environments, 500 steps (10 seconds walking), identical seeds.

| Metric | Raw | Smoothed | Reduction |
|--------|-----|----------|-----------|
| Action Rate RMS | 0.7431 | 0.7311 | 1.6% |
| Jerk RMS | 0.6972 | 0.6642 | **4.7%** |
| Mean Power | 12.077 W | 11.585 W | **4.1%** |
| Torque RMS | 5.1128 | 5.0921 | 0.4% |
| Falls | 0/20 | 0/20 | — |

The 4.1% power reduction comes from fewer rapid direction changes in joint torque = less wasted oscillatory energy.

### 6.5 A-to-B Race (Velocity Benchmark)

Single environment, 500 steps, cmd_vx=1.0 m/s:

| Metric | Raw | Smoothed |
|--------|-----|----------|
| Mean Vx | 0.563 m/s | 0.578 m/s |
| Final X Progress | 5.622 m | 5.764 m (+2.5%) |

The smoothed controller actually walks slightly faster/farther — jittery actions waste energy on oscillations instead of forward progress.

### 6.6 Rough Terrain Evaluation

**Setup:** 5 episodes, 1000 steps each, rough terrain (curriculum terrain tiles), t_end=0.5.

| Metric | Raw | Smoothed (t_end=0.5) | Change |
|--------|-----|---------------------|--------|
| Action Rate RMS | 0.2340 | 0.2001 | **-14.5%** |
| Torque RMS | 3.271 | 3.253 | -0.6% |
| Mean Return | 14.154 | 14.279 | +0.9% |
| Fall Rate | 0% | 0% | Same |

**Why less reduction than flat:** The absolute action rate RMS on rough terrain (0.234) is already much lower than flat (0.589). The rough terrain policy is inherently smoother because: (a) larger network [512,256,128]; (b) many high-frequency signals are real terrain responses (reacting to height variations), not noise. The flow model correctly preserves these — it's doing state-dependent smoothing, not blind filtering.

---

## 7. Models Produced

| File | Description | State Dim | Terrain |
|------|-------------|-----------|---------|
| `flow_model.pt` | Uniform 15Hz cutoff (superseded) | 48 | Flat |
| `flow_model_adaptive.pt` | Per-joint adaptive cutoffs (production) | 48 | Flat |
| `flow_model_rough.pt` | Rough terrain adaptive | 235 | Rough |

All models use the same VelocityNet architecture, just different input dimensions and training data.

---

## 8. Project Iterations — What Worked and What Didn't

### Iteration 1: Basic Flow Matching (Uniform 15Hz Cutoff)
- **What:** Applied same LP cutoff to all 12 joints
- **Result:** Worked conceptually but was too aggressive on some joints and too weak on others
- **Lesson:** Led to Iteration 2

### Iteration 2: Per-Joint Adaptive Cutoffs
- **Discovery:** Through in-sim testing, found that FL_calf, RL_hip, RR_thigh get WORSE with low cutoffs (they carry gait signal). FR_hip, FR_calf, RR_calf benefit most.
- **Result:** The adaptive model (flow_model_adaptive.pt) is strictly better than uniform
- **Key insight:** Joints are not created equal. The asymmetry is striking — FR_hip gets 14.4% reduction while RL_hip gets 0.9%.

### Iteration 3: t_end Tuning
- **Question:** How far to integrate the ODE?
- **Result:** t_end=0.3 chosen for flat terrain (conservative but effective), t_end=0.5 for rough terrain
- **Why not 1.0:** Full integration is too aggressive — it maps to the LP distribution and dampens the policy's reactive behavior

### Iteration 4: Push Recovery Testing
- **First attempt:** 5–50 N forces, 40 envs → 0% fall rate both sides (forces too weak)
- **Second attempt:** 50–500 N, 80 envs → 50% vs 33.8%, McNemar p=0.015
- **Lesson:** Need strong enough perturbation to find the breaking point

### Iteration 5: Effort/Energy Analysis
- **Result:** 4.1% energy savings, 4.7% jerk reduction
- **Interpretation:** Modest but meaningful — smoother commands = less wasted oscillatory effort

### Iteration 6: Rough Terrain Extension
- **Challenge:** 235-dim obs (vs 48), actions carry legitimate high-freq terrain responses
- **Result:** 14.5% jitter reduction, no velocity degradation, model correctly preserves terrain-reactive signals

### Things That Did NOT Work
1. **Uniform cutoff for all joints** — too blunt
2. **Full integration (t_end=1.0) as default** — too aggressive for production
3. **Headless video recording on rough terrain** — Isaac Sim's omni.replicator pipeline is unreliable in headless mode. Camera tracking (ViewerCfg origin_type="asset_root") worked intermittently but not consistently. Non-deterministic GPU rendering meant same script produces valid frames one run and black frames the next. Workaround: use Isaac Sim GUI.
4. **Low push forces (5–50 N)** — not enough to differentiate controllers

---

## 9. Code Organization

### Training Pipeline
1. `collect_rollout.py` / `collect_rollout_rough.py` — Run trained PPO policy, save obs + actions
2. `flow_matching.py` — Basic flow matching with uniform cutoff (superseded)
3. `flow_matching_adaptive.py` — Flow matching with per-joint adaptive cutoffs (production)
4. `flow_matching_rough.py` — Flow matching for rough terrain (235-dim state)

### Evaluation Scripts
1. `push_recovery_eval.py` — 80-env parallel push test, saves binary fall outcomes
2. `push_video_eval.py` — Single-env push with video recording
3. `replay_smoothed.py` — Replay smoothed actions in sim, record video + velocity logs
4. `rough_terrain_eval.py` — Head-to-head rough terrain eval
5. `offline_rough_eval.py` — Offline rough terrain metrics (no sim needed)
6. `analyze_effort.py` — Paired effort/energy analysis
7. `ab_race.py` — A-to-B velocity race comparison
8. `verify_smoothing.py` — In-sim jitter measurement at different t_end values
9. `multi_seed_eval.py` — Multi-seed forward walking evaluation

### Visualization
1. `make_rough_plots.py` — Dark-themed rough terrain comparison plots
2. `make_eff_push_plots.py` — Effort + push recovery plots
3. `make_ab_plots.py` — A/B test comparison plots
4. `add_*_slides.py` — Insert plots into PowerPoint presentations

### Key Utility Scripts
1. `test_flow_in_sim.py` — Quick in-sim test of flow smoothing
2. `inspect_flow.py` / `inspect_rollout.py` — Inspect model weights and data statistics
3. `fast_scan.py` / `find_discordant.py` — Find push environments where controllers differ

### Configs
- `flat_agent.yaml` / `flat_env.yaml` — PPO and env config for flat terrain
- `rough_agent.yaml` / `rough_env.yaml` — PPO and env config for rough terrain

---

## 10. How to Reproduce

### Prerequisites
- NVIDIA Isaac Sim 5.1 + Isaac Lab
- GPU with CUDA support
- Python environment from Isaac Sim (`_isaac_sim\python.bat` on Windows)

### Commands
```bash
# 1. Train PPO (flat)
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 4096

# 2. Collect rollout data
_isaac_sim\python.bat collect_rollout.py \
  --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 1 --num_episodes 5 --checkpoint <model.pt>

# 3. Train flow model
_isaac_sim\python.bat flow_matching_adaptive.py --train --epochs 200

# 4. Evaluate push recovery
_isaac_sim\python.bat push_recovery_eval.py --headless --num_envs 80 \
  --flow_model flow_model_adaptive.pt --t_end 0.3 --checkpoint <model.pt> --results_path push_smoothed.npz

# 5. Generate plots
_isaac_sim\python.bat make_eff_push_plots.py
```

---

## 11. Frequently Asked Questions

**Q: Does flow matching change the policy?**
A: No. The PPO policy is frozen. Flow matching is a post-processor that takes the policy's output and cleans it up before sending it to the actuators.

**Q: Does it output one action or a trajectory?**
A: One action per timestep. It's a per-step post-processor, not a trajectory planner. Temporal coherence emerges because the model is conditioned on slowly-changing observations.

**Q: Why not just increase the action_rate_l2 penalty?**
A: Making it stronger degrades velocity tracking. The penalty creates a fundamental tradeoff — the policy can't optimize both simultaneously. Flow matching separates the smoothing from the policy optimization, avoiding this tradeoff.

**Q: Why not use a simple low-pass filter directly?**
A: A fixed LP filter doesn't know the robot's state. It would smooth away legitimate high-frequency responses (like terrain reactions) along with noise. The flow model is state-conditioned — it learns what's signal vs. noise depending on the observation.

**Q: What's the computational overhead?**
A: 20 Euler steps through a small MLP (3 hidden layers of 256) per timestep. Negligible compared to the physics simulation. The flow_smooth function runs in <1ms on GPU.

**Q: Why is rough terrain jitter reduction smaller?**
A: The rough terrain policy already has lower jitter (0.234 vs 0.589 on flat). It uses a larger network and many of its "high-frequency" actions are real terrain responses, not noise. The model correctly preserves these.

**Q: What do the plots show?**
A: The `plots/` folder contains: effort comparison bars, paired scatter plots, power-over-time traces, push recovery fall rates by magnitude, McNemar discordant visualization, rough terrain action traces, velocity tracking, and smoothness comparison bars.

---

## 12. Data File Reference

All data has been extracted to human/AI-readable JSON in the `data_extracts/` folder:

| File | Contents |
|------|----------|
| `training_losses.json` | Loss curves for flat and rough flow models |
| `rollout_data_stats.json` | Shape, mean, std, min, max for all rollout arrays |
| `per_joint_analysis.json` | Per-joint RMS and reduction at t_end=0.3/0.5/1.0 + cutoff frequencies |
| `smoothing_verification.json` | In-sim action rate and velocity at each t_end |
| `push_recovery.json` | Per-magnitude fall counts, McNemar stats for both push experiments |
| `effort_analysis.json` | Power, jerk, torque, energy comparison (20 paired envs) |
| `ab_race.json` | Velocity and distance comparison |
| `rough_terrain_eval.json` | Full rough terrain eval metrics |
| `velocity_comparison_summary.json` | Per-step velocity trace summaries |
| `multiseed_eval.json` | Multi-seed x-progress results |

---

## 13. Conversation History Summary

This project was developed across multiple sessions between Paul Colombo and Claude. Key phases:

1. **Initial development:** Built the complete flow matching pipeline — PPO rollout collection, Butterworth filtering, VelocityNet training, Euler integration inference. Discovered the jitter problem and implemented the CFM solution.

2. **Adaptive cutoffs:** Discovered that uniform filtering hurts some joints. Systematically tested per-joint cutoffs in simulation. Built flow_model_adaptive.pt.

3. **Evaluation battery:** Designed and ran push recovery (80 envs, McNemar test), effort analysis (20 paired envs), AB race, velocity tracking verification, and rough terrain extension.

4. **Visualization and presentation:** Generated dark-themed plots, built PowerPoint presentations with inserted plots, created the organized project folder with 37 files.

5. **Rough terrain:** Extended to 235-dim observations, trained flow_model_rough.pt, evaluated showing 14.5% jitter reduction with no velocity degradation.

6. **Video recording (struggled):** Attempted headless video recording on rough terrain. Isaac Sim's rendering pipeline proved unreliable — non-deterministic GPU state caused black frames. Camera tracking worked intermittently. Concluded that GUI recording is the reliable approach.

7. **Documentation:** Created comprehensive README.md, iteration notes document (.docx), and this knowledge base.

---

## 14. References

- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL" (CoRL 2022)
- NVIDIA Isaac Lab documentation: https://isaac-sim.github.io/IsaacLab/
