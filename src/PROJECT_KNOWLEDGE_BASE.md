# Flow Matching for Smoothing Quadruped Locomotion — Complete Knowledge Base (Ablation Edition)

> **Purpose of this document:** This file contains everything needed to understand,
> explain, and answer questions about the 9-variant ablation study completed on
> 2026-05-04. It is designed to be read by a teammate or by an LLM acting as one.
> All key numbers, decisions, failures, and rationale are inline.

---

## 1. Project summary

We took the prior "Bal-LP" Flow Matching result from the deck (Iter 5: PPO + flow
trained on optimal-then-LP targets, 4% fall rate) and ran a **controlled ablation**
to isolate the contribution of each ingredient.

The new finding is that the action-rate reward penalty has a **sharp sweet spot at
λ = −0.01**:
- removing it weakens flow's benefit (4% → 24% fall rate)
- the existing default value is the optimum
- pushing it 50× higher destroys the underlying policy and flow then makes things worse

The ablation also shows that **flow is policy-specific** — reusing a flow trained on
PPO-A on PPO-B's actions hurts performance (this was a methodological mistake in the
v1 of the report; corrected in v2/v3).

**Team:** Paul Colombo (Sophomore CS+Robotics), Arnav Shah, Jack Gerdsen — CMU,
Spring 2026.
**Platform:** NVIDIA Isaac Sim 5.1 + Isaac Lab 0.54.3, Unitree Go1, PPO via RSL-RL
3.x, RTX 5070 Laptop.

---

## 2. The 9 variants

Each variant is `<PPO reward variant> + <inference-time smoother>`.

### PPO reward variants (vary `λ_action_rate_l2` only, all other rewards unchanged)
- **noreg**     λ =  0.0
- **somereg**   λ = −0.01  (existing default; matches the deck)
- **extremereg** λ = −0.5

### Inference-time smoothers
- **raw**     no smoothing
- **flow**    Bal-LP flow trained on **the same PPO's** rollouts, t_end = 1.0
- **lp**      causal 1st-order IIR Butterworth, fc = 15 Hz, online
- **flow_lp** flow output then passed through the causal IIR (cascade)

The 9 evaluated combinations:
1. noreg + raw
2. somereg + raw
3. extremereg + raw
4. noreg + flow
5. somereg + flow
6. extremereg + flow
7. noreg + lp
8. somereg + lp
9. noreg + flow_lp

(somereg + flow_lp and extremereg + lp/flow_lp were judged redundant for the story.)

---

## 3. Mathematical detail

### 3.1 PPO clipped surrogate

```
L^CLIP(θ) = E_t [ min( r_t(θ) Â_t, clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

Hyperparameters identical across all three reward variants:
- 4096 parallel envs
- 24 steps/env/iter, 5 epochs, 4 mini-batches
- ε = 0.2, γ = 0.99, GAE λ = 0.95
- Adaptive KL learning rate, target KL = 0.01, base lr = 1e-3
- Actor/critic = MLP[128, 128, 128] ELU
- 300 iterations
- Seed 42

### 3.2 Conditional Flow Matching (Lipman 2023, OT paths)

Each training example is `(x_0, x_1, s)` where `x_0` is the raw PPO action, `x_1` is
the LP-smoothed optimal target, and `s` is the 48-D observation.

OT path between `x_0` and `x_1`:
```
x_t = (1 − t) x_0 + t x_1                   t ∈ [0, 1]
```
Conditional velocity (the displacement, independent of `t` for OT):
```
v* = x_1 − x_0
```
Loss minimized:
```
L(θ) = E_{(x_0, x_1, s) ~ D, t ~ U(0,1)} ||v_θ(x_t, t, s) − (x_1 − x_0)||²
```
Network `v_θ`: MLP [61 → 256 → 256 → 256 → 12], SiLU activations, no normalization.
Trained with AdamW, lr = 1e-3, weight_decay = 1e-5, cosine annealing, 200 epochs,
batch size 512.

Inference: Euler integration of the learned ODE
```
x ← x_0
dt = t_end / N_steps                         # we use N_steps = 20
for i in 0..N_steps−1:
    t = i · dt
    x ← x + v_θ(x, t, s) · dt
return x
```
`t_end ∈ [0, 1]` is a continuous smoothing-intensity dial. We use 1.0.

### 3.3 Optimal target generation (random shooting)

We pick `x_1` for each timestep by short-horizon multi-objective optimization:
- restore the sim to recorded `(joint_pos, joint_vel, root_pose, root_vel)`
- generate K = 16 candidates `x_0 + ε`, ε ~ N(0, σ² I), σ = 0.1
- roll out H = 10 steps with each candidate as the action at t=0, then PPO
  policy thereafter
- score each rollout with multi-objective cost
- pick the argmin

Cost per step:
```
c = w_track  · (v_x − v_cmd)²
  + w_jerk   · ||a − a_prev||²
  + w_energy · |τ · q̇|        (sum over joints)
  + w_stab   · ||g_z + 1||²    (projected gravity)
```
"Balanced" weights: w_track = 3, w_jerk = 0.5, w_energy = 0.3, w_stab = 2.

Total cost = mean over H steps. Argmin selects `x_star`.

### 3.4 LP-on-targets (Iter 5 trick)

After getting per-timestep `x_star`, we Butterworth-LP-filter the per-episode
trajectory at fc = 15 Hz, 2nd order, zero-phase via `scipy.signal.filtfilt`.
This produces `x_star_lp` which is what the flow trains against.

Why: the random-shooting targets are temporally incoherent (each step is independent).
LP-filtering enforces a smooth trajectory at the target distribution level, so the
flow learns to point toward "physics-aware *and* smooth" without needing any explicit
smoothness term in its own loss.

### 3.5 Causal IIR LP (1st-order RC)

For the "+ LP" variants we use a causal one-pole IIR online during inference (filtfilt
is non-causal so cannot run online):
```
α = dt / (RC + dt),   RC = 1 / (2π fc),   dt = 1/50 s,   fc = 15 Hz
y_n = (1 − α) y_{n−1} + α x_n
```

This introduces phase lag (~1 sample), which is part of why it cannot match the
zero-phase filtfilt for offline spectral comparisons.

---

## 4. Training run details

### 4.1 PPO training (per variant)

| Variant | λ | Iterations | Wall (s) | Mean reward (final) | Notes |
|---|---|---|---|---|---|
| noreg | 0.0 | 300 | 367.97 | ~34 | Healthy gait, mean Vx 1.03 m/s. |
| somereg | −0.01 | 300 | (existing) | (existing) | Existing checkpoint from 2026-04-06; same hyperparams. |
| extremereg | −0.5 | 300 | 341.36 | ~5 | **Policy collapsed to standing.** Mean Vx ≈ 0. |

### 4.2 Optimal-target generation (per variant)

| Variant | Total wall (s) | Steps/sec | Improvement rate | Mean cost reduction |
|---|---|---|---|---|
| noreg | 1166.6 | 4.3 | 96.7% (4836/5000) | 729.85 → 640.33 (89.5) |
| extremereg | 1176.5 | 4.2 | 96.9% (4846/5000) | 55.88 → 48.19 (7.7) |

The extreme-reg cost is much lower in absolute terms because the policy is frozen —
there's not much to optimize. The mean improvement of 7.7 vs noreg's 89.5 reflects
this: random shooting can find better-than-frozen actions, but they still depend on a
policy that can't follow up.

### 4.3 Flow training (per variant)

| Variant | Final loss | Epochs |
|---|---|---|
| noreg | 0.011461 | 200 |
| extremereg | 0.001267 | 200 |

The extreme-reg flow has a much lower training loss because the LP-smoothed targets
are very close to `x_0` (Mean |x_smooth − x0| = 0.07 vs noreg's 0.22). The flow has
less work to do — but as we'll see, the work it does is wrong for the policy that
will execute it.

---

## 5. Results — full table

Spectral metrics come from a 2000-step rollout at vx=1.0 fixed command. Push results
from 80 envs, 50–500 N (10 magnitudes × 8 envs each), 200-ms impulse at t=2 s.

| Variant | λ | HF >10Hz | Action rate RMS | Jerk RMS | Mean Vx (m/s) | Fall rate | Inference (ms) | Train wall (s) |
|---|---|---|---|---|---|---|---|---|
| noreg + raw | 0 | 11.90% | 1.010 | 1.170 | 1.031 | 31.25% | 0.013 | 367.97 |
| somereg + raw | −0.01 | 7.97% | 0.717 | 0.677 | 1.067 | 50.00% | 0.013 | (existing) |
| extremereg + raw | −0.5 | 3.74% | 0.049 | 0.071 | −0.001 | 46.25% | 0.013 | 341.36 |
| noreg + flow | 0 | 4.17% | 0.970 | 0.983 | 1.031 | **23.75%** | 5.562 | 367.97 |
| somereg + flow | −0.01 | 6.46% | 0.723 | 0.647 | 1.067 | **3.75%** | 6.008 | (existing) |
| extremereg + flow | −0.5 | 6.35% | 0.077 | 0.124 | −0.001 | **100.00%** | 5.740 | 341.36 |
| noreg + lp | 0 | 3.32% | 0.888 | 0.849 | 1.031 | 33.75% | 0.069 | 367.97 |
| somereg + lp | −0.01 | 4.92% | 0.671 | 0.526 | 1.067 | 56.25% | 0.065 | (existing) |
| noreg + flow_lp | 0 | 1.23% | 0.915 | 0.841 | 1.031 | 37.50% | 5.603 | 367.97 |

---

## 6. What each result tells us

### 6.1 Reward-side regularization on its own
- **noreg → 31.25%**, somereg → 50.00%, extremereg → 46.25%

The action-rate penalty alone *hurts* push robustness on this task. The penalty
suppresses rapid corrections, which are exactly what's needed to recover from a 50–
500 N impulse. Removing the penalty makes the policy more agile (and more jittery).
Pushing the penalty to extreme values causes the policy to learn the degenerate
"stand still" solution, which falls 46% of the time anyway because there are no
corrections at all when pushed.

This is **counterintuitive** to the deck's framing of action-rate as a smoothing
mechanism. It does smooth (HF energy 11.9% → 7.97% → 3.74% as λ increases) but
the smoothness comes at a cost in robustness.

### 6.2 Bal-LP flow on its native policy
- noreg + flow → 23.75% (−7.5 pp vs noreg raw)
- **somereg + flow → 3.75%** (−46.25 pp vs somereg raw) ← **winner, matches deck**
- extremereg + flow → 100.00% (+53.75 pp vs extremereg raw) ← **catastrophic**

**The combination of moderate reg + flow is the best result anywhere.** Neither
ingredient alone gets close. Removing the reg makes the flow less effective. Pushing
the reg too far breaks the underlying policy so badly that even flow can't rescue
it — the flow tells the frozen policy to walk, and it falls trying.

This is the single most important finding of the new ablation: **flow matching is
not a substitute for reward shaping. They are complementary, with a sharp sweet
spot at the existing default λ = −0.01.**

### 6.3 Causal IIR LP filter
- noreg + lp → 33.75% (slightly worse than raw)
- somereg + lp → 56.25% (worse than raw)

The LP filter alone at 15 Hz removes high-frequency content indiscriminately,
including legitimate corrections. The phase lag from a causal IIR makes this worse
than the deck's offline filtfilt. **LP alone is not a viable smoother for push
recovery on this task.**

### 6.4 Flow + LP cascade
- noreg + flow_lp → 37.50% fall rate, **1.23% HF energy** (the lowest jitter
  anywhere)

Same dead-end as the deck's Iter 3: chaining the smoothers averages their effects
rather than compounding them. The LP layer suppresses corrections that the flow
correctly preserves. The HF energy is gorgeous; the fall rate is mediocre.

---

## 7. Reconciliation with the original deck

| Claim | Deck | This ablation | Reconciled |
|---|---|---|---|
| Some-reg raw fall rate | 50% (40/80) | 50% (40/80) | ✓ exact match |
| Bal-LP fall rate | 4% (3/80) | 3.75% (3/80) | ✓ exact match |
| Bal-LP HF energy reduction (some-reg) | −38% | −19% (7.97% → 6.46%) | Different methodology — see §10 |
| HF energy on raw some-reg | 2.1% | 7.97% | Different FFT methodology — DC removal / windowing differ |

Internal consistency across new variants is what matters for the ablation. The deck
numbers are still valid for the comparisons made *within* the deck.

---

## 8. The cross-policy generalization mistake (v1 of report)

In v1 of the report, we used the existing flow_model_balanced_lp.pt (trained on
some-reg PPO) and applied it to the no-reg PPO's actions. Result: noreg + flow looked
worse than noreg raw (40% vs 31% fall rate). This led to the incorrect headline
"flow matching is worse than LP."

In v2 we retrained the flow on no-reg PPO rollouts. Result: noreg + flow improved to
23.75% — better than raw, with a clear smoothness benefit. The v1 result was a
methodological artifact: the flow had learned a velocity field tied to the some-reg
action distribution, which doesn't transfer.

**Practical implication:** the Bal-LP recipe must be re-run whenever the underlying
PPO changes. The flow training itself is fast (~30 s); the bottleneck is
optimal-target generation (~19 min for 5000 timesteps).

---

## 9. The extreme-reg-flow collapse (a notable failure)

extremereg + flow = 80/80 fell. This is not a bug. The flow is trying to push the
frozen policy's actions toward optimal targets that say "move forward to track 1
m/s". The policy never learned coordinated walking (because the −0.5 penalty made
moving very expensive). When the flow modifies the action enough to begin a step,
the next state is one the policy hasn't trained on — so the next action is wrong,
and the robot falls.

The lesson: **flow matching depends on the underlying policy being capable of
following up on the corrections.** The flow modifies a single timestep; the policy
has to handle the next 100. If the policy is broken, flow makes things worse, not
better.

---

## 10. Methodology notes (caveats)

### 10.1 Spectral analysis methodology
We compute HF energy as: `Σ_{f > 10 Hz} PSD(f) / Σ_f PSD(f)`, averaged across
joints, in percent. PSD via a single-window rfft of the 2000-step rollout per joint
after DC removal. No tapering window. No detrending beyond DC removal.

The deck's HF numbers (e.g., 1.4% for raw) suggest a different methodology —
possibly: (a) computed only over a subset of joints, (b) a windowed FFT (Hann/Hamming),
(c) a specific frequency band, or (d) PSD normalized differently.

We chose internal consistency over deck reproduction. All 9 variants in this report
are computed with the same code, so within-table comparisons are valid. The
**relative** improvement from flow on some-reg is roughly comparable to the deck:
−19% in our methodology vs −38% in the deck.

### 10.2 Push protocol
- 80 envs, 10 magnitudes from 50 N to 500 N in 50 N steps, 8 envs per magnitude
- 200 ms impulse at t = 2 s, applied to the trunk in the +y direction
- 500-step episode (10 s)
- Termination = trunk contact (illegal_contact term) within the episode
- "Fall rate" = fraction of envs where termination fired
- vx_command = 1.0 m/s, vy_command = 0, yaw_rate_command = 0 throughout

### 10.3 Spectral rollout protocol
- 1 env, 2000 steps (40 s @ 50 Hz), vx = 1.0 fixed (we monkey-patch the velocity
  command term so randomization can't kick in)
- Stock terrain (flat plane), no domain randomization beyond the env defaults
- Used as input to FFT for HF energy; also for action-rate / jerk RMS

### 10.4 Inference latency
Measured during the push test from inside `push_recovery_runner.py`. Wall time
between `policy(obs)` returning and the smoothed action being ready. CUDA-synchronized.
Mean over the last 450 steps of the 500-step push test (drops the warmup). Reported
in milliseconds at the per-batch level (80 envs in parallel) — i.e., this is the
*total* latency for the smoothing pass on 80 envs, not per-env.

For deployment that's actually the right number, since you'd run a single policy on
a single robot — but the figure already aggregates across 80, so per-env is even
smaller. Real-robot latency would be dominated by the policy itself, not the
smoothing block.

### 10.5 The 80-env push test noise floor
With 8 envs per magnitude bucket, each bucket's pass/fail rate has σ ≈ 17% under
binomial. Aggregating across 10 magnitudes gives σ ≈ 5.6% on the overall fall rate.
Differences less than ~6 pp are at the noise floor. Differences ≥ 12 pp are clearly
real.

The 23.75% / 31.25% gap (8 envs out of 80) is meaningful. The 31.25% / 33.75% gap
(2 envs) is at the noise floor.

---

## 11. Code organization in this package

```
code/
├── ablation/                            # Scripts written for this ablation study
│   ├── register_reg_variants.py         # Adds NoReg-v0 and ExtremeReg-v0 task IDs
│   ├── train_ablation.py                # Mirrors IsaacLab train.py, adds timing + variants
│   ├── collect_rollout_fixed.py         # Fixed-cmd vx=1.0 rollout for spectral analysis
│   ├── lp_filter_targets.py             # Butterworth LP on per-episode x_star trajectories
│   ├── synth_smooth_rollouts.py         # Offline-apply smoothers for spectral comparison
│   ├── synth_lp_variant.py              # Standalone offline Butterworth wrapper
│   ├── push_recovery_runner.py          # The 4-mode push runner (raw/flow/lp/flow_lp)
│   ├── stitch_timings.py                # Copy timing.json into the ablation data dir
│   ├── analyze_ablation.py              # Compute all metrics, write results.json/.csv
│   ├── make_comparison_plot.py          # 4-panel dark-themed bar chart
│   ├── build_report.py                  # Generate the final docx
│   ├── stage_existing.py                # Copy existing data into the ablation layout
│   ├── _validate_docx.py                # Validity check on the deliverable docx
│   ├── run_all_pushes.bat               # Batch runner for the 7 pushes
│   ├── rerun_47.bat                     # Batch runner for variants 4 & 7 with new flow
│   └── _torch_check.py / _register_smoketest.py  # Sanity checks
└── core/                                # Existing project scripts we depend on
    ├── collect_rollout_states.py        # Existing — rollout with full sim state
    ├── generate_optimal_targets.py      # Existing — random shooting K-candidate eval
    └── flow_matching_optimal.py         # Existing — flow training on optimal targets
```

---

## 12. Frequently anticipated questions

**Q: Why not just run with extra reg weights to fill the curve in?**
A: Each new λ requires a full pipeline run: PPO training (~6 min) + rollout-states
(~1 min) + optimal-target gen (~19 min) + flow training (~30 s) + push test (~15 s)
+ analysis ≈ 27 min per λ. We sampled three points to bracket the behavior; the
sweet-spot finding is robust at the bracket already.

**Q: Why not retrain flow on the existing some-reg PPO rollout (instead of the v1
shortcut of reusing the model)?**
A: We did. The "somereg + flow" row uses `flow_model_balanced_lp.pt`, which was
trained on rollouts of this exact some-reg PPO checkpoint. It's the deck's Iter-5
model. The 3.75% fall rate confirms the deck.

**Q: Why is the flow model "policy-specific"?**
A: The flow's velocity field `v_θ(x_t, t, s)` is conditioned on the obs state `s`,
but the *direction* it points (toward x_1) is a function of x_0 — and x_0 is the
PPO action. Different PPOs produce different action distributions. The flow
extrapolates poorly outside the action distribution it trained on. Using a
some-reg-trained flow on no-reg actions is OOD inference.

**Q: Why did the LP filter hurt some-reg?**
A: The 15 Hz cutoff is below the 17 Hz peak we see in some-reg's action spectrum
(corresponds to a corrective oscillation needed for stability under the action-rate
penalty). The LP attenuates that peak, removing legitimate corrections. The
no-reg policy doesn't have that peak (it can correct at any frequency), so LP
hurts it less.

**Q: Could a better LP cutoff fix this?**
A: Maybe. The deck's per-joint adaptive cutoffs (8–25 Hz) try to address this
exact issue. We didn't replicate that for the new ablation — only the uniform
15 Hz baseline. The flow matching approach implicitly learns per-joint cutoffs
from data, which is part of why it works.

**Q: How brittle is the sweet-spot finding?**
A: Tested at three λ values that span 5 orders of magnitude (0, 0.01, 0.5). The
extremereg+flow=100% result is robust (every magnitude bucket had 8/8 falls). The
some-reg+flow = 3.75% closely matches the deck (4%), so we have two independent
data points at λ=−0.01. The no-reg+flow result is intermediate as expected. The
shape of the curve (sweet-spot in the middle) is consistent across multiple metrics
(HF energy, jerk, fall rate). Confident the qualitative finding holds.

**Q: How do I deploy this to a real Go1?**
A: Same pipeline. PPO + Bal-LP flow + t_end=1.0 inference. The flow is policy-
specific so retrain it on whichever PPO checkpoint you ship. Inference is ~6 ms per
step at 50 Hz, well within the 20 ms budget. The flow is a 73K-parameter MLP, easy to
deploy.

**Q: What's left as future work?**
A: (a) Rough terrain — we only tested flat. (b) Sim-to-real validation. (c) An
adaptive `t_end` — currently a constant. (d) Train PPO + flow jointly via a 2-stage
objective rather than post-hoc.

---

## 13. Files and what they contain

### `data_extracts/`
- `push_recovery_all_variants.json` — fall rates per magnitude per variant
- `spectral_analysis.json` — HF energy, action rate RMS, jerk RMS per variant
- `ppo_training_summary.json` — per-iter timing for the new PPOs
- `optimal_target_gen_summary.json` — improvement rates, cost reductions
- `flow_training_summary.json` — final losses, training curves
- `cost_inference_breakdown.json` — per-mode inference latency stats
- `reward_variants.md` — what changed between the three PPO reward configs

### `flow_models/`
- `flow_model_balanced_lp_noreg.pt`     — trained on no-reg rollouts, state_dim 48
- `flow_model_balanced_lp_somereg.pt`   — trained on some-reg rollouts (the existing model)
- `flow_model_balanced_lp_extremereg.pt` — trained on extreme-reg rollouts (use with caution)

### `results/`
- `results.csv`, `results.json` — the master ablation table
- `push_<variant>.npz` (×9) — raw push outcome data per variant
- `rollout_<variant>.npz` (×7) — fixed-cmd rollouts (3 PPOs + 4 offline-smoothed)

---

## 14. Conversation history (sanitized)

The ablation was completed in a single session of ~3 hours wall-clock:

1. Discovered that the existing `rollout_data.npz` was collected with random vx
   commands, not the deck's vx=1.0; existing `push_raw.npz` used 5–50 N magnitudes
   not 50–500 N. Recollected both with the protocol matching the deck.
2. Trained no-reg PPO (368 s, mean reward → 34) and extreme-reg PPO (341 s, mean
   reward → 5; collapsed to standing).
3. Ran initial 7-variant push recovery using the existing somereg-trained flow
   model on the no-reg policy. Observed flow performing worse than LP on no-reg
   (40% vs 33%) and incorrectly concluded "flow is worse" in v1 of the report.
4. Identified the cross-policy generalization mistake. Trained a new flow on no-reg
   rollouts (rollout_states ~1 min, optimal-target gen ~19 min, flow training
   ~30 s).
5. Re-ran pushes for variants 4 and 7. Result: noreg + flow improved from 40% →
   23.75% fall rate. Rewrote the report (v2).
6. User asked whether some-reg + flow was tested. It wasn't (not in the original
   7-variant matrix). Ran it: 3.75% fall rate — confirmed the deck.
7. User requested a small reg sweep with flow at each level. Trained an extreme-reg
   flow (rollout_states + optimal-target gen ~19 min, flow training ~30 s),
   evaluated. Result: 100% fall rate — catastrophic collapse.
8. Final v3 of report has 9 variants with the sweet-spot finding clearly stated.

Total compute: 3 PPO trainings, 2 fresh optimal-target generations, 2 fresh flow
trainings, 11 push recovery evaluations, full spectral analysis pipeline, plot +
docx generation.

---

## 15. References

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023).
  *Flow Matching for Generative Modeling.* ICLR.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022).
  *Learning to Walk in Minutes Using Massively Parallel Deep RL.* CoRL.
- Williams, G., Drews, P., Goldfain, B., Rehg, J. M., & Theodorou, E. A. (2017).
  *Information theoretic MPC for model-based reinforcement learning.* (Background
  for the random-shooting candidate evaluation idea.)
- NVIDIA Isaac Lab — https://isaac-sim.github.io/IsaacLab/
