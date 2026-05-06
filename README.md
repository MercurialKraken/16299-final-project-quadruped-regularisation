# 16-299 Final Project — Exploring Jitter-Reduction Methods for RL Locomotion Policies

**Authors:** Paul Colombo, Arnav Shah, Jack Gerdsen
**Course:** 16-299 — Robotics, Spring 2026
**Institution:** Carnegie Mellon University

## Overview

This repository hosts the GitHub Pages site for our final project report investigating jitter-reduction methods for reinforcement learning quadruped locomotion policies on the Unitree Go1 in NVIDIA Isaac Sim. We compare three families of techniques — reward-side action regularisation, online causal low-pass filtering, and a learned post-hoc refinement based on conditional flow matching — through a 9-variant ablation evaluated under push-recovery stress tests.

**Headline result:** Some-reg PPO + Bal-LP Flow achieves **3.75% fall rate** under 50–500 N lateral pushes (vs 50% raw), a 46.25 percentage-point reduction, with **−38% high-frequency spectral energy**.

## Live site

The report is deployed at:
`https://<username>.github.io/16299-final-project-quadruped-regularisation/`

## Repo structure

```
.
├── index.html                       # The full report (single-page site)
├── assets/
│   └── ablation_comparison.png      # 9-variant ablation comparison plot
└── README.md                        # This file
```

## Deployment instructions

1. Create a new public GitHub repo named `16299-final-project-quadruped-regularisation`.
2. Push these files to the `main` branch at the repo root.
3. Go to **Settings → Pages → Source** and set it to `Deploy from a branch`, branch `main`, folder `/ (root)`.
4. Wait ~1 minute for the deployment to complete. The site will be live at the URL above.

## References

- Lipman et al. (2023). *Flow Matching for Generative Modeling.* ICLR.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
- Mittal et al. (2023). *Orbit / Isaac Lab.* IEEE RA-L.
- Fjelde, Mathieu, Dutordoir (2024). *An Introduction to Flow Matching.* Cambridge MLG Blog.
