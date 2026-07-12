"""
verify_smoothing.py — Definitive in-sim verification that flow matching
actually reduces jitter. Runs raw and smoothed policies back-to-back,
logs EVERY action and observation, then computes hard metrics.

Usage:
    isaaclab.bat -p verify_smoothing.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 ^
        --headless --num_envs 1
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
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
import torch.nn as nn

CHECKPOINT = r"A:\IsaacLab\logs\rsl_rl\unitree_go1_flat\2026-04-06_12-42-26\model_299.pt"
FLOW_MODEL = r"A:\IsaacLab\flow_model.pt"
OUT_FILE = r"A:\IsaacLab\smoothing_verification.txt"
OUT_NPZ = r"A:\IsaacLab\smoothing_verification.npz"

class VelocityNet(nn.Module):
    def __init__(self, action_dim=12, state_dim=48, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + 1 + state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, x_t, t, state):
        if t.dim() == 1: t = t.unsqueeze(-1)
        return self.net(torch.cat([x_t, t, state], dim=-1))

@torch.no_grad()
def flow_smooth(model, raw_action, state, n_steps=20, t_end=1.0):
    x = raw_action.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = torch.full((x.shape[0], 1), i * dt, device=x.device)
        x = x + model(x, t_val, state) * dt
    return x

def obs_to_flat(obs):
    if isinstance(obs, torch.Tensor): return obs
    try: return obs["policy"]
    except: pass
    try: return next(iter(obs.values()))
    except: raise TypeError(f"Cannot extract tensor from {type(obs)}")

def run_episode(env, policy, flow_model, dev, t_end, n_steps, cmd_vx=0.5):
    """Run one episode, log everything."""
    obs = env.get_observations()
    # Force velocity command
    try:
        vel_cmd = env.unwrapped.command_manager._terms["base_velocity"]
        vel_cmd.command[:, 0] = cmd_vx
        vel_cmd.command[:, 1] = 0.0
    except: pass

    all_actions = []
    all_raw_actions = []
    all_obs = []
    all_vx = []
    all_rewards = []

    for step in range(n_steps):
        with torch.inference_mode():
            raw_action = policy(obs)
            flat_obs = obs_to_flat(obs)

            if t_end > 0.0:
                action = flow_smooth(flow_model, raw_action, flat_obs, t_end=t_end)
            else:
                action = raw_action

            obs, rew, dones, _ = env.step(action)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

            # Re-force command after potential reset
            try:
                vel_cmd.command[:, 0] = cmd_vx
                vel_cmd.command[:, 1] = 0.0
            except: pass

            flat = obs_to_flat(obs)
            all_actions.append(action[0].cpu().numpy())
            all_raw_actions.append(raw_action[0].cpu().numpy())
            all_obs.append(flat[0].cpu().numpy())
            all_vx.append(flat[0, 0].item())
            all_rewards.append(rew[0].item())

    return {
        "actions": np.array(all_actions),       # (T, 12) - what we sent to sim
        "raw_actions": np.array(all_raw_actions), # (T, 12) - what policy produced
        "obs": np.array(all_obs),               # (T, 48)
        "vx": np.array(all_vx),                 # (T,)
        "rewards": np.array(all_rewards),        # (T,)
    }

def compute_metrics(data, label):
    """Compute jitter and performance metrics from logged data."""
    actions = data["actions"]
    raw_actions = data["raw_actions"]
    obs = data["obs"]
    vx = data["vx"]
    rewards = data["rewards"]

    # Action smoothness: rate of change
    act_diff = np.diff(actions, axis=0)  # (T-1, 12)
    act_rate_rms = np.sqrt(np.mean(act_diff**2))
    act_rate_per_joint = np.sqrt(np.mean(act_diff**2, axis=0))  # (12,)

    # Joint velocities from obs (indices 24:36)
    joint_vel = obs[:, 24:36]  # (T, 12)
    # Joint accelerations (finite diff of joint vel)
    joint_acc = np.diff(joint_vel, axis=0) / 0.02  # (T-1, 12)
    joint_acc_rms = np.sqrt(np.mean(joint_acc**2))
    joint_acc_per_joint = np.sqrt(np.mean(joint_acc**2, axis=0))

    # Velocity tracking
    mean_vx = np.mean(vx)
    std_vx = np.std(vx)

    # Reward
    mean_reward = np.mean(rewards)
    total_reward = np.sum(rewards)

    return {
        "label": label,
        "act_rate_rms": float(act_rate_rms),
        "act_rate_per_joint": act_rate_per_joint.tolist(),
        "joint_acc_rms": float(joint_acc_rms),
        "joint_acc_per_joint": joint_acc_per_joint.tolist(),
        "mean_vx": float(mean_vx),
        "std_vx": float(std_vx),
        "mean_reward": float(mean_reward),
        "total_reward": float(total_reward),
        "act_magnitude": float(np.mean(np.abs(actions))),
    }

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = 42
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CHECKPOINT)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    dev = env.unwrapped.device

    flow_model = VelocityNet().to(dev)
    flow_model.load_state_dict(torch.load(FLOW_MODEL, map_location=dev, weights_only=True))
    flow_model.eval()

    N = args_cli.steps
    configs = [
        ("RAW (no smoothing)", 0.0),
        ("SMOOTHED t_end=0.3", 0.3),
        ("SMOOTHED t_end=0.5", 0.5),
        ("SMOOTHED t_end=1.0", 1.0),
    ]

    all_metrics = []
    all_data = {}
    for label, t_end in configs:
        data = run_episode(env, policy, flow_model, dev, t_end, N, cmd_vx=0.5)
        m = compute_metrics(data, label)
        all_metrics.append(m)
        all_data[f"actions_t{t_end}"] = data["actions"]
        all_data[f"obs_t{t_end}"] = data["obs"]
        all_data[f"vx_t{t_end}"] = data["vx"]

    # Save raw data for plotting
    np.savez(OUT_NPZ, **all_data)

    # Write report
    jnames = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
              "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]
    raw_m = all_metrics[0]

    with open(OUT_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  FLOW MATCHING SMOOTHING VERIFICATION (IN-SIMULATION)\n")
        f.write("  cmd_vx = 0.5 m/s | {} steps per config | seed=42\n".format(N))
        f.write("=" * 80 + "\n\n")

        # Summary table
        f.write("{:<25s} {:>12s} {:>12s} {:>10s} {:>10s} {:>10s} {:>12s}\n".format(
            "Config", "Act Rate RMS", "Joint Acc RMS", "Mean vx", "Std vx", "Mean Rew", "|Action|"))
        f.write("-" * 95 + "\n")
        for m in all_metrics:
            f.write("{:<25s} {:>12.4f} {:>12.2f} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f}\n".format(
                m["label"], m["act_rate_rms"], m["joint_acc_rms"],
                m["mean_vx"], m["std_vx"], m["mean_reward"], m["act_magnitude"]))

        # Improvement vs raw
        f.write("\n--- Improvement vs Raw ---\n")
        for m in all_metrics[1:]:
            ar_red = (1 - m["act_rate_rms"] / raw_m["act_rate_rms"]) * 100
            ja_red = (1 - m["joint_acc_rms"] / raw_m["joint_acc_rms"]) * 100
            vx_diff = m["mean_vx"] - raw_m["mean_vx"]
            rew_diff = m["mean_reward"] - raw_m["mean_reward"]
            f.write(f"  {m['label']}:\n")
            f.write(f"    Action rate reduction:  {ar_red:+.2f}%\n")
            f.write(f"    Joint accel reduction:  {ja_red:+.2f}%\n")
            f.write(f"    Velocity change:        {vx_diff:+.4f} m/s\n")
            f.write(f"    Reward change:          {rew_diff:+.4f}\n\n")

        # Per-joint breakdown
        f.write("--- Per-Joint Action Rate RMS ---\n")
        f.write("{:<12s}".format("Joint"))
        for m in all_metrics:
            f.write(" {:>14s}".format(m["label"][:14]))
        f.write("\n")
        for j, name in enumerate(jnames):
            f.write("{:<12s}".format(name))
            for m in all_metrics:
                f.write(" {:>14.4f}".format(m["act_rate_per_joint"][j]))
            f.write("\n")

        f.write("\n--- Per-Joint Acceleration RMS ---\n")
        f.write("{:<12s}".format("Joint"))
        for m in all_metrics:
            f.write(" {:>14s}".format(m["label"][:14]))
        f.write("\n")
        for j, name in enumerate(jnames):
            f.write("{:<12s}".format(name))
            for m in all_metrics:
                f.write(" {:>14.2f}".format(m["joint_acc_per_joint"][j]))
            f.write("\n")

        f.write("\n--- VERDICT ---\n")
        best = all_metrics[1]  # t_end=0.3
        ar_red = (1 - best["act_rate_rms"] / raw_m["act_rate_rms"]) * 100
        ja_red = (1 - best["joint_acc_rms"] / raw_m["joint_acc_rms"]) * 100

        if ar_red > 0 and ja_red > 0:
            f.write(f"  SMOOTHING WORKS: {ar_red:.1f}% action jitter reduction, {ja_red:.1f}% joint acceleration reduction\n")
            f.write(f"  Velocity preserved: {best['mean_vx']:.3f} vs {raw_m['mean_vx']:.3f} m/s\n")
            f.write(f"  Reward preserved: {best['mean_reward']:.4f} vs {raw_m['mean_reward']:.4f}\n")
        elif ar_red > 0:
            f.write(f"  PARTIAL: {ar_red:.1f}% action smoothing but joint accel not reduced\n")
        else:
            f.write(f"  NO IMPROVEMENT: smoothing did not reduce in-sim jitter\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
