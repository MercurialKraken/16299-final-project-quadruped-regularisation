"""
compare_vel.py — Run raw and smoothed policies back-to-back with forced
forward velocity, writing results directly to files (no stdout dependency).

Usage:
    isaaclab.bat -p compare_vel.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 \
        --headless --num_envs 1
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=500)
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import torch, numpy as np, gymnasium as gym, time
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
RESULTS_FILE = r"A:\IsaacLab\velocity_comparison.json"

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

def run_episode(env, policy, flow_model, device, t_end, max_steps, cmd_vx=0.5):
    """Run one episode and return velocity log."""
    obs = env.get_observations()
    
    # Override velocity command to force forward walking
    try:
        base_env = env.unwrapped
        vel_cmd = base_env.command_manager._terms["base_velocity"]
        vel_cmd.command[:, 0] = cmd_vx
        vel_cmd.command[:, 1] = 0.0  # no lateral
    except Exception as e:
        pass  # If override fails, continue with random command
    
    vel_data = []
    for step in range(max_steps):
        with torch.inference_mode():
            raw_action = policy(obs)
            flat_obs = obs_to_flat(obs)

            if t_end > 0.0:
                action = flow_smooth(flow_model, raw_action, flat_obs, t_end=t_end)
            else:
                action = raw_action
            
            obs, rew, dones, _ = env.step(action)
            
            from packaging import version as ver
            if ver.parse(installed_version) >= ver.parse("4.0.0"):
                policy.reset(dones)
            
            # Re-override command after potential reset
            try:
                vel_cmd.command[:, 0] = cmd_vx
                vel_cmd.command[:, 1] = 0.0
            except:
                pass
            
            flat = obs_to_flat(obs)
            vx = flat[0, 0].item()
            vy = flat[0, 1].item()
            act_mag = torch.mean(torch.abs(action)).item()
            raw_mag = torch.mean(torch.abs(raw_action)).item()
            act_rate = torch.mean(torch.abs(action - raw_action)).item() if t_end > 0 else 0.0
            vel_data.append({"step": step, "vx": vx, "vy": vy, 
                           "act_mag": act_mag, "raw_mag": raw_mag, "act_delta": act_rate})
    
    return vel_data

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

    results = {}
    t_end_values = [0.0, 0.3, 0.5, 1.0]
    
    for t_end in t_end_values:
        label = f"t_end={t_end}"
        vel_data = run_episode(env, policy, flow_model, dev, t_end, args_cli.max_steps, cmd_vx=0.5)
        
        vx_values = [d["vx"] for d in vel_data]
        act_mags = [d["act_mag"] for d in vel_data]
        
        results[label] = {
            "t_end": t_end,
            "mean_vx": float(np.mean(vx_values)),
            "std_vx": float(np.std(vx_values)),
            "mean_act_mag": float(np.mean(act_mags)),
            "steps_vx_positive": int(np.sum(np.array(vx_values) > 0.1)),
            "total_steps": len(vel_data),
            "vx_trace": vx_values,  # full trace for plotting
        }
    
    # Write results to JSON file (reliable, no stdout needed)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also write a simple summary to a text file
    with open(r"A:\IsaacLab\velocity_summary.txt", 'w') as f:
        f.write("VELOCITY COMPARISON: Raw vs Smoothed (cmd_vx=0.5 m/s)\n")
        f.write("="*70 + "\n\n")
        for label, r in results.items():
            f.write(f"{label}:\n")
            f.write(f"  Mean vx:        {r['mean_vx']:+.4f} m/s\n")
            f.write(f"  Std vx:         {r['std_vx']:.4f} m/s\n")
            f.write(f"  Mean |action|:  {r['mean_act_mag']:.4f}\n")
            f.write(f"  Steps vx > 0.1: {r['steps_vx_positive']}/{r['total_steps']}\n\n")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
