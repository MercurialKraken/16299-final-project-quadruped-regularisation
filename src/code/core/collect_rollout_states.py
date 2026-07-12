"""
collect_rollout_states.py — Collect rollout data WITH full simulation state.

Saves observations, actions, AND the complete robot state (root pose, root vel,
joint pos, joint vel) at every timestep so we can later restore the sim to any
recorded state and run candidate evaluations from there.

Usage:
    isaaclab.bat -p collect_rollout_states.py ^
        --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless ^
        --num_envs 1 --num_episodes 5 --checkpoint <path_to_model.pt>
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--output", type=str, default=None,
                    help="Output path. Default: rollout_states_{flat|rough}.npz")
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import torch, numpy as np, gymnasium as gym
installed_version = metadata.version("rsl-rl-lib")

import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner


def obs_to_flat(obs):
    if isinstance(obs, torch.Tensor):
        return obs
    try:
        return obs["policy"]
    except (KeyError, TypeError):
        pass
    try:
        return next(iter(obs.values()))
    except Exception:
        raise TypeError(f"Cannot extract tensor from obs: {type(obs)}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[collect-states] Loaded: {args_cli.checkpoint}")

    robot = env.unwrapped.scene["robot"]

    # Storage lists
    all_obs = []
    all_actions = []
    all_root_pos = []      # (T, 3) world position
    all_root_quat = []     # (T, 4) quaternion wxyz
    all_root_lin_vel = []  # (T, 3) world-frame linear vel
    all_root_ang_vel = []  # (T, 3) world-frame angular vel
    all_joint_pos = []     # (T, 12) joint positions
    all_joint_vel = []     # (T, 12) joint velocities
    all_applied_torque = []  # (T, 12) applied torques
    ep_returns = []
    ep_lengths = []

    obs = env.get_observations()

    for ep in range(args_cli.num_episodes):
        ep_ret, ep_len = 0.0, 0

        while ep_len < args_cli.max_steps:
            flat = obs_to_flat(obs)

            # ---- Snapshot full sim state BEFORE taking action ----
            root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().copy()     # (3,)
            root_quat = robot.data.root_quat_w[0].detach().cpu().numpy().copy()   # (4,) wxyz
            root_lv = robot.data.root_lin_vel_w[0].detach().cpu().numpy().copy()  # (3,)
            root_av = robot.data.root_ang_vel_w[0].detach().cpu().numpy().copy()  # (3,)
            jpos = robot.data.joint_pos[0].detach().cpu().numpy().copy()          # (12,)
            jvel = robot.data.joint_vel[0].detach().cpu().numpy().copy()          # (12,)

            all_obs.append(flat[0].cpu().numpy().copy())
            all_root_pos.append(root_pos)
            all_root_quat.append(root_quat)
            all_root_lin_vel.append(root_lv)
            all_root_ang_vel.append(root_av)
            all_joint_pos.append(jpos)
            all_joint_vel.append(jvel)

            # ---- Take action ----
            with torch.inference_mode():
                action = policy(obs)
            all_actions.append(action[0].cpu().numpy().copy())

            obs, reward, dones, extras = env.step(action)
            ep_ret += float(reward[0])
            ep_len += 1

            # Grab torque after step (applied during this step)
            torque = robot.data.applied_torque[0].detach().cpu().numpy().copy()
            all_applied_torque.append(torque)

            if dones.any():
                break

        print(f"  Ep {ep+1}/{args_cli.num_episodes} | steps={ep_len} | ret={ep_ret:.1f}")
        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)

    # ---- Save ----
    obs_dim = all_obs[0].shape[0]
    terrain = "rough" if obs_dim > 100 else "flat"
    if args_cli.output:
        out_path = args_cli.output
    else:
        out_path = os.path.join(os.path.dirname(__file__), f"rollout_states_{terrain}.npz")

    np.savez(
        out_path,
        obs=np.array(all_obs),                    # (T, obs_dim)
        actions=np.array(all_actions),             # (T, 12)
        root_pos=np.array(all_root_pos),           # (T, 3)
        root_quat=np.array(all_root_quat),         # (T, 4) wxyz
        root_lin_vel=np.array(all_root_lin_vel),    # (T, 3)
        root_ang_vel=np.array(all_root_ang_vel),    # (T, 3)
        joint_pos=np.array(all_joint_pos),         # (T, 12)
        joint_vel=np.array(all_joint_vel),         # (T, 12)
        applied_torque=np.array(all_applied_torque), # (T, 12)
        ep_returns=np.array(ep_returns),
        ep_lengths=np.array(ep_lengths),
        dt=0.02,
    )
    T = len(all_obs)
    print(f"\n[collect-states] Saved {T} timesteps ({terrain} terrain, obs_dim={obs_dim})")
    print(f"[collect-states] -> {out_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
