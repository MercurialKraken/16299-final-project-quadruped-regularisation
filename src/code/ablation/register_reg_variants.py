"""Register PPO reward-ablation variants for the Go1 flat task.

Imported BEFORE training scripts so two extra gym task IDs become available:
  - Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0       (action_rate_l2 weight = 0.0)
  - Isaac-Velocity-Flat-Unitree-Go1-ExtremeReg-v0  (action_rate_l2 weight = -0.5)

Everything else (env, obs, dynamics, other reward terms) matches the stock
UnitreeGo1FlatEnvCfg, so the only thing that varies between the three PPOs in
the ablation table is the weight on action_rate_l2.
"""
import gymnasium as gym
from isaaclab.utils import configclass

# ensure the stock task is registered first
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import (
    UnitreeGo1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1 import agents


@configclass
class UnitreeGo1FlatNoRegEnvCfg(UnitreeGo1FlatEnvCfg):
    """No action-rate regularization in the reward (ablation variant)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = 0.0


@configclass
class UnitreeGo1FlatExtremeRegEnvCfg(UnitreeGo1FlatEnvCfg):
    """Extreme action-rate regularization (50x stock)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.5


def _register_once(task_id: str, cfg_cls):
    if task_id in gym.envs.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": cfg_cls,
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg"
            ),
        },
    )


_register_once("Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0", UnitreeGo1FlatNoRegEnvCfg)
_register_once(
    "Isaac-Velocity-Flat-Unitree-Go1-ExtremeReg-v0", UnitreeGo1FlatExtremeRegEnvCfg
)
