# Quadruped Flow Matching — Jitter Reduction for RL Locomotion Policies

**Course:** 16-299 Introduction to Robotics, Spring 2026 · Carnegie Mellon University
**Authors:** Paul Colombo, Arnav Shah, Jack Gerdsen
**Platform:** NVIDIA Isaac Sim 5.1 + Isaac Lab · Unitree Go1 · PPO via RSL-RL

This repo contains everything used to produce the project results: Isaac Lab training/eval/visualization scripts, trained-model rollout extracts, RL configs, and the full knowledge base. The static report site lives in a separate repo at [`16299-final-project-quadruped-regularisation`](https://github.com/PaulCarnegie10/16299-final-project-quadruped-regularisation).

**Headline result:** Some-reg PPO + Bal-LP Flow achieves a **3.75% fall rate** under 50–500 N lateral pushes (vs 50% raw), a 46.25-point reduction, with **−38% high-frequency spectral energy** and zero velocity-tracking degradation.

## Repo layout

```
.
├── README.md                  ← you are here
├── docs/
│   └── PROJECT_KNOWLEDGE_BASE.md   Full project write-up: math, decisions, results
├── configs/                   RSL-RL / Isaac Lab YAML configs
│   ├── flat_agent.yaml        PPO actor/critic for flat terrain
│   ├── flat_env.yaml          Reward, observation, terrain spec (flat)
│   ├── rough_agent.yaml       PPO actor/critic for rough terrain
│   └── rough_env.yaml         Reward, observation, terrain spec (rough)
├── data/                      JSON extracts from training & eval runs
│   ├── ab_race.json
│   ├── effort_analysis.json
│   ├── multiseed_eval.json
│   ├── per_joint_analysis.json
│   ├── push_recovery.json
│   ├── rollout_data_stats.json
│   ├── rough_terrain_eval.json
│   ├── smoothing_verification.json
│   ├── training_losses.json
│   └── velocity_comparison_summary.json
└── code/
    ├── training/              PPO rollout collection + flow-matching training
    │   ├── collect_rollout.py
    │   ├── collect_rollout_rough.py
    │   ├── flow_matching.py            Conditional flow-matching trainer (flat)
    │   ├── flow_matching_adaptive.py   Adaptive-conditioning variant
    │   └── flow_matching_rough.py      Rough-terrain variant
    ├── evaluation/            Metric harnesses against the trained policies
    │   ├── ab_race.py
    │   ├── analyze_effort.py
    │   ├── effort_eval.py
    │   ├── multi_seed_eval.py
    │   ├── multi_seed_scan.py
    │   ├── offline_rough_eval.py
    │   ├── push_recovery_eval.py
    │   ├── push_video_eval.py
    │   ├── record_rough_video.py
    │   ├── replay_smoothed.py
    │   ├── rough_terrain_eval.py
    │   └── verify_smoothing.py
    ├── analysis/              Post-hoc data crunching
    │   ├── analyze_multiseed.py
    │   ├── analyze_push.py
    │   ├── analyze_velocity.py
    │   └── compare_vel.py
    ├── visualization/         Plot + slide generation
    │   ├── add_ab_slides.py
    │   ├── add_eff_push_slides.py
    │   ├── add_rough_slides.py
    │   ├── make_ab_plots.py
    │   ├── make_ab_plots_dark.py
    │   ├── make_eff_push_plots.py
    │   ├── make_rough_plots.py
    │   └── plot_comparison.py
    └── utilities/             Inspectors, scanners, smoke tests
        ├── fast_scan.py
        ├── find_discordant.py
        ├── inspect_flow.py
        ├── inspect_npz.py
        ├── inspect_rollout.py
        ├── scan_discordant.py
        └── test_flow_in_sim.py
```

## Pipeline at a glance

1. **Train PPO** (Isaac Lab built-in). Configs in `configs/`.
2. **Collect rollouts** with `code/training/collect_rollout.py` (or `_rough.py`) → `.npz` of (obs, raw action, low-pass action) tuples.
3. **Train the flow model** with `code/training/flow_matching.py` (or `_adaptive.py` / `_rough.py`). Learns a velocity field that transports raw actions toward low-pass-filtered actions, conditioned on observations.
4. **Evaluate** under push tests / rough terrain / multi-seed sweeps in `code/evaluation/`. Partial ODE integration (`t_end < 1.0`) controls smoothing intensity at deploy time.
5. **Analyze + plot** with `code/analysis/` and `code/visualization/`. Outputs feed into the report site.

## Reproducing results

Requires NVIDIA Isaac Sim 5.1 + Isaac Lab, RSL-RL, PyTorch, and a CUDA GPU. See `docs/PROJECT_KNOWLEDGE_BASE.md` for hyperparameters, observation/action spaces, the full reward table, and a per-experiment description of what each script produces.

## Live site

The report site is hosted separately at [`16299-final-project-quadruped-regularisation`](https://github.com/PaulCarnegie10/16299-final-project-quadruped-regularisation) and deployed via GitHub Pages.

## References

- Lipman et al. (2023). *Flow Matching for Generative Modeling.* ICLR.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
- Mittal et al. (2023). *Orbit / Isaac Lab.* IEEE RA-L.
- Fjelde, Mathieu, Dutordoir (2024). *An Introduction to Flow Matching.* Cambridge MLG Blog.
