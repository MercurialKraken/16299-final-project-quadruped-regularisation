# Flow Matching for Smoothing Quadruped Locomotion — Ablation Package

## Overview

This package documents the **complete 9-variant ablation study** crossing PPO reward-side
action regularization (no-reg / some-reg / extreme-reg) with post-hoc smoothing methods
(none / Bal-LP flow trained per-policy / causal IIR LP / cascade), evaluated on the
Unitree Go1 in Isaac Sim with 80-env push recovery (50–500 N lateral impulse).

It supersedes (and supplements) the original Iter-1..5 deck by isolating
the contribution of (a) reward-side regularization, (b) flow matching, and (c) low-pass
filtering — variables that were entangled in the original presentation.

**Headline:**

| PPO reward `λ_action_rate_l2` | Raw fall rate | + Bal-LP Flow |
|---|---|---|
| 0.0  (no-reg)        | 31.25% | 23.75% |
| **−0.01 (some-reg, default)** | 50.00% | **3.75%** ← winner, matches deck Iter-5 |
| −0.5  (extreme-reg)  | 46.25% | 100% (catastrophic collapse) |

The action-rate penalty has a **sharp sweet spot at −0.01**: removing it weakens flow's
benefit, pushing it extreme breaks the underlying policy so badly that flow makes
things worse. The existing default weight is already the optimum.

---

## Mathematical Foundation

### 1. PPO base policy

We train a Gaussian policy `π_θ(a | s)` to maximize the clipped surrogate objective:

```
L^CLIP(θ) = E_t [ min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
r_t(θ)    = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

with GAE(λ=0.95) advantages `Â_t`, discount γ=0.99, clip ε=0.2. Actor and critic
are MLP[128,128,128] ELU. Adaptive KL learning-rate schedule with desired KL=0.01.

The Go1 has 48-D proprioceptive observations: `[base_lin_vel(3), base_ang_vel(3),
projected_gravity(3), velocity_commands(3), joint_pos(12), joint_vel(12), prev_action(12)]`.
Action space is 12-D joint position offsets from default stance, scale 0.25, default
hip ±0.1, thigh 0.8/1.0, calf −1.5. Sim runs at 200 Hz physics, decimation 4 → 50 Hz
control.

The reward sums:
```
r = 1.5  · track_lin_vel_xy_exp   + 0.75 · track_ang_vel_z_exp
  − 2.0  · lin_vel_z_l2           − 0.05 · ang_vel_xy_l2
  − 2e-4 · dof_torques_l2         − 2.5e-7 · dof_acc_l2
  + λ    · action_rate_l2
  + 0.25 · feet_air_time          − 2.5  · flat_orientation_l2
```

The ablation varies **only** `λ_action_rate_l2 ∈ {0.0, −0.01, −0.5}`. Everything else
is identical (same seed=42, 4096 envs, 300 iterations).

### 2. Conditional Flow Matching with Optimal Transport paths

We learn a velocity field `v_θ(x_t, t, s)` that transports samples from the source
(raw PPO action) to the target (optimal-then-LP-smoothed action), conditioned on
observation `s`.

**Path:** linear interpolation between `x_0` (raw) and `x_1` (target) gives
the OT path under the squared-Euclidean cost:
```
x_t = (1 − t) · x_0 + t · x_1,   t ∈ [0, 1]
```

**Velocity along the path** is the displacement (independent of `t` for OT paths):
```
v*(x_t, t | x_0, x_1) = x_1 − x_0
```

**Training objective** (MSE regression of the conditional velocity, marginalizing
over couplings as in Lipman et al. 2023):
```
L(θ) = E_{(x_0, x_1, s) ∼ D, t ∼ U(0,1)} ||v_θ(x_t, t, s) − (x_1 − x_0)||²
```

**Network:** VelocityNet, MLP[61→256→256→256→12] with SiLU activations. Input is the
concatenation `[x_t (12), t (1), s (48)]`. AdamW, lr=1e-3, weight_decay=1e-5, cosine
annealing over 200 epochs, batch 512.

**Inference** by Euler-integrating the ODE from `x_0`:
```
x ← x_0
dt = t_end / N_steps     # we use N_steps = 20
for i in 0..N_steps−1:
    t = i · dt
    x ← x + v_θ(x, t, s) · dt
return x
```
`t_end ∈ [0, 1]` is the smoothing intensity dial. We use `t_end = 1.0`.

### 3. Optimal target generation (the Bal-LP recipe)

For each recorded timestep `(s_t, x_0_t, sim_state_t)` from a PPO rollout, we run
short-horizon random shooting in K=16 parallel envs and pick the candidate with the
lowest multi-objective cost:

```
Cost = w_track · ||v_x − v_cmd||²      (per-step)
     + w_jerk  · ||a − a_prev||²       (per-step)
     + w_energy · |τ · q̇|              (per-step, summed over joints)
     + w_stab  · ||g_z + 1||²          (per-step, projected gravity)
```

Weights (Iter-5 "balanced"): `w_track=3, w_jerk=0.5, w_energy=0.3, w_stab=2`.

For each of K candidates we restore the sim to the recorded state, perturb the PPO
action with Gaussian noise (`σ=0.1`), then roll out H=10 steps with the PPO policy
in the loop. Average the per-step cost over the horizon. Pick `x_star = argmin(cost)`.

**Then** we Butterworth-LP the per-episode trajectory of `x_star` at fc=15 Hz (2nd
order, zero-phase `filtfilt`) — this is the "LP-on-targets" trick from the deck's
Iter-5. The flow is then trained against `x_1 = x_star_lp`.

**Why this works:** the optimal targets are physics-aware (better-than-current
actions for the cost), but they're not temporally coherent — random shooting picks
each candidate independently. LP-filtering the trajectory enforces temporal coherence
at the target level. The flow then learns a velocity field that points toward
"physics-aware *and* smooth" — both objectives baked in at the target distribution
level rather than as separate terms.

### 4. Causal IIR low-pass (the alternative we tested)

For variants 5 and 6 ("+ LP"), we apply a 1st-order Butterworth-style IIR low-pass
online during inference:

```
α = dt / (RC + dt),   RC = 1/(2π·f_c),   dt = 1/50 s,   f_c = 15 Hz
y_n = (1 − α) · y_{n−1} + α · x_n
```

Causal (one-sided): introduces phase lag but can run online during the push test. This
is *different* from the deck's `scipy.signal.filtfilt` (non-causal, two-sided, zero
phase) which we use only for offline spectral comparison.

---

## Pipeline / How to Reproduce

### Step 0: Prerequisites
- NVIDIA Isaac Sim 5.1 + Isaac Lab at `A:\AllIsaac\IsaacLab` (Windows)
- RTX 5070 Laptop or similar GPU, 8 GB+ VRAM, headless mode required if no display
- Python from Isaac Sim (`isaaclab.bat -p ...` or `_isaac_sim\python.bat`)

### Step 1: Train 3 PPOs at different λ
```
cd A:\AllIsaac\IsaacLab
isaaclab.bat -p A:\AllIsaac\flow_matching_project\scripts\ablation\train_ablation.py ^
  --task Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 --variant_name noreg ^
  --headless --num_envs 4096 --max_iterations 300 --seed 42
isaaclab.bat -p A:\AllIsaac\flow_matching_project\scripts\ablation\train_ablation.py ^
  --task Isaac-Velocity-Flat-Unitree-Go1-ExtremeReg-v0 --variant_name extremereg ^
  --headless --num_envs 4096 --max_iterations 300 --seed 42
```
Some-reg PPO already exists from a prior training run.
Each takes ~6 min on RTX 5070 Laptop. The new task IDs are registered by
`scripts/ablation/register_reg_variants.py` (imported from inside `train_ablation.py`).

### Step 2: Per-policy Bal-LP flow training
For each PPO checkpoint:
```
# 2a) Collect rollout with full sim state (5 episodes × 1000 steps, ~1 min)
isaaclab.bat -p A:\AllIsaac\IsaacLab\collect_rollout_states.py ^
  --task Isaac-Velocity-Flat-Unitree-Go1-v0 ^
  --num_envs 1 --num_episodes 5 --max_steps 1000 ^
  --checkpoint <path> --output rollout_states_<variant>.npz --headless

# 2b) Generate optimal targets via random shooting (~19 min for noreg)
isaaclab.bat -p A:\AllIsaac\IsaacLab\generate_optimal_targets.py ^
  --task Isaac-Velocity-Flat-Unitree-Go1-v0 ^
  --num_candidates 16 --horizon 10 ^
  --rollout_data rollout_states_<variant>.npz --checkpoint <path> ^
  --output optimal_targets_<variant>.npz ^
  --w_tracking 3.0 --w_jerk 0.5 --w_energy 0.3 --w_stability 2.0 --headless

# 2c) LP-filter the targets (instant)
isaaclab.bat -p A:\AllIsaac\flow_matching_project\scripts\ablation\lp_filter_targets.py ^
  --in_npz optimal_targets_<variant>.npz ^
  --out_npz optimal_targets_<variant>_lp.npz --fc 15.0

# 2d) Train flow model (200 epochs, ~30 s)
isaaclab.bat -p A:\AllIsaac\IsaacLab\flow_matching_optimal.py ^
  --train --data optimal_targets_<variant>_lp.npz ^
  --model flow_model_balanced_lp_<variant>.pt --epochs 200
```

### Step 3: Run all 9 push-recovery evals (~15 s each at 80 envs)
```
isaaclab.bat -p A:\AllIsaac\flow_matching_project\scripts\ablation\push_recovery_runner.py ^
  --task <task> --variant <name> --mode <raw|flow|lp|flow_lp> ^
  --checkpoint <path> [--flow_model <path>] --t_end 1.0 --headless
```
Run 9 invocations (one per variant). Outputs `data\ablation\push_<variant>.npz`.

### Step 4: Spectral analysis + report build
```
isaaclab.bat -p scripts\ablation\collect_rollout_fixed.py    # fixed-cmd vx=1.0 rollout per PPO
isaaclab.bat -p scripts\ablation\synth_smooth_rollouts.py    # offline-smoothed rollouts for spectral
isaaclab.bat -p scripts\ablation\stitch_timings.py
isaaclab.bat -p scripts\ablation\analyze_ablation.py         # writes results.json/csv
isaaclab.bat -p scripts\ablation\make_comparison_plot.py     # 4-panel dark plot
isaaclab.bat -p scripts\ablation\build_report.py             # final docx
```

---

## Results Summary

Sorted by fall rate (lower = more push-robust):

| Rank | Variant | λ | HF >10Hz | Action rate RMS | Fall rate | Inference |
|---|---|---|---|---|---|---|
| 🥇 | Some-reg + Bal-LP Flow | −0.01 | 6.46% | 0.723 | **3.75%** | 6.0 ms |
| 2 | No-reg + Bal-LP Flow | 0 | 4.17% | 0.970 | 23.75% | 5.6 ms |
| 3 | No-reg PPO (raw) | 0 | 11.90% | 1.010 | 31.25% | 0.013 ms |
| 4 | No-reg + Causal IIR LP | 0 | 3.32% | 0.888 | 33.75% | 0.069 ms |
| 5 | No-reg + Flow + LP | 0 | **1.23%** | 0.915 | 37.50% | 5.6 ms |
| 6 | Extreme-reg PPO (frozen) | −0.5 | 3.74% | 0.049 | 46.25% | 0.013 ms |
| 7 | Some-reg PPO (raw) | −0.01 | 7.97% | 0.717 | 50.00% | 0.013 ms |
| 8 | Some-reg + Causal IIR LP | −0.01 | 4.92% | 0.671 | 56.25% | 0.065 ms |
| ❌ | Extreme-reg + Bal-LP Flow | −0.5 | 6.35% | 0.077 | **100%** | 5.7 ms |

### Per-policy Bal-LP flow gain (apples-to-apples)

| Reg level | Raw | + Bal-LP Flow | Δ |
|---|---|---|---|
| λ = 0 (no-reg) | 31.25% | 23.75% | −7.5 pp |
| λ = −0.01 (some-reg) | 50.00% | 3.75% | **−46.25 pp** |
| λ = −0.5 (extreme-reg) | 46.25% | 100.00% | +53.75 pp |

The action-rate penalty makes the raw policy fragile but trains a flow that recovers
spectacularly. Removing the penalty entirely yields a more push-robust raw policy but
a less effective flow. Pushing it extreme breaks the policy and the flow can't rescue it.

---

## What's in this package

| Path | Description |
|---|---|
| `README.md` | This file. |
| `PROJECT_KNOWLEDGE_BASE.md` | Comprehensive deep-dive (FAQ-style, all numbers, all rationales). |
| `Ablation_Report.docx` | Polished landscape report (5 pages, 9-row table, embedded plot). |
| `ablation_comparison.png` | 4-panel dark-themed comparison chart. |
| `configs/` | PPO env/agent yamls + a description of the reward-variant changes. |
| `data_extracts/` | Per-topic JSON summaries (push results, training timing, etc.). |
| `code/ablation/` | All scripts written for this ablation study. |
| `code/core/` | Existing scripts we depend on (collect_rollout_states, generate_optimal_targets, flow_matching_optimal). |
| `flow_models/` | The 3 trained Bal-LP flow models (one per λ). |
| `results/` | `results.csv`, `results.json`, individual `push_*.npz`. |

---

## References

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023).
  *Flow Matching for Generative Modeling.* ICLR.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022).
  *Learning to Walk in Minutes Using Massively Parallel Deep RL.* CoRL.
- NVIDIA Isaac Lab — https://isaac-sim.github.io/IsaacLab/
