# Reward variants used in this ablation

Only `action_rate_l2.weight` is varied. Everything else is identical to `flat_env_cfg.py`.

| Variant | Task ID | `action_rate_l2.weight` |
|---|---|---|
| noreg | Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 | 0.0 |
| somereg | Isaac-Velocity-Flat-Unitree-Go1-v0 (stock) | -0.01 |
| extremereg | Isaac-Velocity-Flat-Unitree-Go1-ExtremeReg-v0 | -0.5 |

Registered via `scripts/ablation/register_reg_variants.py`.
