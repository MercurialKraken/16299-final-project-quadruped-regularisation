"""Collect rollout with FIXED forward velocity command (vx=1.0) and a pinnable
checkpoint, so spectral analysis is comparable across variants.

Output: rollout_<variant>.npz keyed by --variant.
"""
import argparse
import os
import sys
import time

# Allow `import cli_args` (from IsaacLab's RSL-RL launcher dir)
sys.path.insert(0, r"A:\AllIsaac\IsaacLab\scripts\reinforcement_learning\rsl_rl")

from isaaclab.app import AppLauncher
import cli_args  # type: ignore

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=str, required=True)
ap.add_argument("--variant", type=str, required=True)
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--num_steps", type=int, default=2000,
                help="Total steps (single env). 2000 @ 50Hz = 40s of gait.")
ap.add_argument("--cmd_vx", type=float, default=1.0)
ap.add_argument("--cmd_vy", type=float, default=0.0)
ap.add_argument("--cmd_yaw", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out_dir", type=str,
                default=r"A:\AllIsaac\flow_matching_project\data\ablation")
cli_args.add_rsl_rl_args(ap)
AppLauncher.add_app_launcher_args(ap)
args_cli, hydra_args = ap.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import register_reg_variants  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")


def obs_to_flat(o):
    if isinstance(o, torch.Tensor):
        return o
    try:
        return o["policy"]
    except Exception:
        return next(iter(o.values()))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[collect] policy loaded: {args_cli.checkpoint}")

    # pin velocity command
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    override = torch.zeros_like(cmd_term.command)
    override[:, 0] = args_cli.cmd_vx
    override[:, 1] = args_cli.cmd_vy
    override[:, 2] = args_cli.cmd_yaw
    def _patched_compute(dt):
        cmd_term.command.copy_(override)
    cmd_term.compute = _patched_compute

    obs, _ = env.reset()
    cmd_term.command[:] = override
    N = args_cli.num_envs
    DT = 0.02

    A_log = np.zeros((args_cli.num_steps, N, 12), dtype=np.float32)
    O_log = np.zeros((args_cli.num_steps, N, 48), dtype=np.float32)
    JP_log = np.zeros((args_cli.num_steps, N, 12), dtype=np.float32)
    JV_log = np.zeros((args_cli.num_steps, N, 12), dtype=np.float32)
    BV_log = np.zeros((args_cli.num_steps, N, 3), dtype=np.float32)
    fell_log = np.zeros(N, dtype=bool)

    t0 = time.time()
    for step in range(args_cli.num_steps):
        flat = obs_to_flat(obs)
        with torch.inference_mode():
            action = policy(obs)
        A_log[step] = action.detach().cpu().numpy()
        O_log[step] = flat.detach().cpu().numpy()
        JP_log[step] = flat[:, 12:24].detach().cpu().numpy()
        JV_log[step] = flat[:, 24:36].detach().cpu().numpy()
        BV_log[step] = flat[:, :3].detach().cpu().numpy()

        obs, _, dones, _ = env.step(action)
        cmd_term.command[:] = override
        try:
            tos = env.unwrapped.termination_manager.time_outs
            falls = dones & (~tos)
        except Exception:
            falls = dones
        fell_log |= falls.detach().cpu().numpy().astype(bool)

    wall = time.time() - t0
    # collapse env dim if num_envs == 1 to keep legacy shape
    def _sq(x):
        return x[:, 0] if x.shape[1] == 1 else x.reshape(-1, x.shape[-1])

    actions = _sq(A_log)
    obs_arr = _sq(O_log)
    jpos = _sq(JP_log)
    jvel = _sq(JV_log)
    bvel = _sq(BV_log)
    jacc = np.zeros_like(jvel)
    jacc[1:] = np.diff(jvel, axis=0) / DT

    print(f"[collect] {wall:.1f}s  steps={args_cli.num_steps}  fell={fell_log.sum()}/{N}")
    print(f"[collect] mean_vx={bvel[:,0].mean():.3f}  action_rate_rms="
          f"{np.sqrt(np.mean(np.diff(actions,axis=0)**2)):.4f}")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    out_path = os.path.join(args_cli.out_dir, f"rollout_{args_cli.variant}.npz")
    np.savez(out_path, obs=obs_arr, actions=actions, joint_pos=jpos, joint_vel=jvel,
             joint_acc=jacc, base_vel=bvel, fell=fell_log,
             cmd_vx=args_cli.cmd_vx, dt=DT, num_envs=N, num_steps=args_cli.num_steps,
             checkpoint=args_cli.checkpoint, variant=args_cli.variant)
    print(f"[collect] saved -> {out_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
