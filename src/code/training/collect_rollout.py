"""
collect_rollout.py  -- Collect joint/action data from trained Go1 policy.
Usage: isaaclab.bat -p collect_rollout.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 1 --num_episodes 5
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1000)
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import torch, numpy as np, gymnasium as gym
from packaging import version
installed_version = metadata.version("rsl-rl-lib")

import isaaclab_tasks
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

CHECKPOINT = r"A:\IsaacLab\logs\rsl_rl\unitree_go1_flat\2026-04-06_12-42-26\model_299.pt"

def obs_to_flat(obs):
    """Extract flat (N, 48) tensor from obs, which may be Tensor, dict, or TensorDict."""
    if isinstance(obs, torch.Tensor):
        return obs
    # TensorDict or dict -- get the "policy" key
    try:
        val = obs["policy"]
        if isinstance(val, torch.Tensor):
            return val
    except (KeyError, TypeError):
        pass
    # fallback: grab first value
    try:
        val = next(iter(obs.values()))
        if isinstance(val, torch.Tensor):
            return val
    except Exception:
        pass
    raise TypeError(f"Cannot extract tensor from obs type: {type(obs)}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CHECKPOINT)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"\n[collect] Loaded {CHECKPOINT}\n[collect] Running {args_cli.num_episodes} episodes\n")

    all_obs, all_act, all_jpos, all_jvel, all_bvel = [], [], [], [], []
    ep_returns, ep_lengths = [], []

    obs = env.get_observations()
    for ep in range(args_cli.num_episodes):
        ep_obs, ep_act, ep_jpos, ep_jvel, ep_bvel = [], [], [], [], []
        ep_ret, ep_len = 0.0, 0
        while ep_len < args_cli.max_steps:
            with torch.inference_mode():
                action = policy(obs)
            # Extract flat tensor for logging
            flat = obs_to_flat(obs)
            obs_np = flat.cpu().numpy()[0]
            ep_obs.append(obs_np.copy())
            ep_act.append(action.cpu().numpy()[0].copy())
            ep_jpos.append(obs_np[12:24].copy())
            ep_jvel.append(obs_np[24:36].copy())
            ep_bvel.append(obs_np[0:3].copy())

            obs, reward, dones, extras = env.step(action)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            ep_ret += float(reward[0])
            ep_len += 1
            if dones.any():
                break

        print(f"  Ep {ep+1}/{args_cli.num_episodes} | steps={ep_len} | ret={ep_ret:.1f}")
        all_obs.append(np.array(ep_obs)); all_act.append(np.array(ep_act))
        all_jpos.append(np.array(ep_jpos)); all_jvel.append(np.array(ep_jvel))
        all_bvel.append(np.array(ep_bvel))
        ep_returns.append(ep_ret); ep_lengths.append(ep_len)

    obs_c = np.concatenate(all_obs); act_c = np.concatenate(all_act)
    jpos_c = np.concatenate(all_jpos); jvel_c = np.concatenate(all_jvel)
    bvel_c = np.concatenate(all_bvel)
    DT = 0.02
    jacc_c = np.zeros_like(jvel_c); jacc_c[1:] = np.diff(jvel_c, axis=0) / DT
    ar = np.diff(act_c, axis=0)
    print(f"\n-- Baseline --")
    print(f"  timesteps       : {len(obs_c)}")
    print(f"  action_rate_rms : {np.sqrt(np.mean(ar**2)):.4f}")
    print(f"  joint_acc_rms   : {np.sqrt(np.mean(jacc_c[1:]**2)):.4f}")
    print(f"  mean_fwd_vel    : {np.mean(bvel_c[:,0]):.4f}")

    out = os.path.join(os.path.dirname(__file__), "rollout_data.npz")
    np.savez(out, obs=obs_c, actions=act_c, joint_pos=jpos_c, joint_vel=jvel_c,
             joint_acc=jacc_c, base_vel=bvel_c,
             ep_returns=np.array(ep_returns), ep_lengths=np.array(ep_lengths))
    print(f"[collect] Saved -> {out}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
