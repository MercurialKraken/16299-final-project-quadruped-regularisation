"""
replay_smoothed.py — Replay flow-matched (smoothed) actions in Isaac Sim and record video.
Supports --t_end for partial integration, --log_vel for velocity logging,
and --cmd_vx/--cmd_vy to override velocity commands (force forward walking).

Usage:
    isaaclab.bat -p replay_smoothed.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 \
        --headless --num_envs 1 --t_end 0.3 --log_vel --cmd_vx 0.5
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--t_end", type=float, default=1.0, help="Flow integration endpoint (0-1)")
parser.add_argument("--log_vel", action="store_true", help="Log base velocity each step")
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps to run")
parser.add_argument("--cmd_vx", type=float, default=None, help="Override forward vel command")
parser.add_argument("--cmd_vy", type=float, default=None, help="Override lateral vel command")
parser.add_argument("--model", type=str, default=None, help="Flow model path (default: flow_model.pt)")
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import torch, numpy as np, gymnasium as gym, time
from packaging import version
installed_version = metadata.version("rsl-rl-lib")

import isaaclab_tasks
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner
import torch.nn as nn

CHECKPOINT = r"A:\IsaacLab\logs\rsl_rl\unitree_go1_flat\2026-04-06_12-42-26\model_299.pt"
FLOW_MODEL = r"A:\IsaacLab\flow_model.pt"

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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_dir = os.path.dirname(CHECKPOINT)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", f"smoothed_t{args_cli.t_end}"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CHECKPOINT)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Load flow model
    model_path = args_cli.model if args_cli.model else FLOW_MODEL
    if not os.path.isabs(model_path):
        model_path = os.path.join(r"A:\IsaacLab", model_path)
    flow_model = VelocityNet().to(env.unwrapped.device)
    flow_model.load_state_dict(torch.load(model_path, map_location=env.unwrapped.device, weights_only=True))
    flow_model.eval()
    print(f"[INFO] Flow model loaded from {model_path}. t_end={args_cli.t_end}")

    # Override velocity command if specified
    if args_cli.cmd_vx is not None or args_cli.cmd_vy is not None:
        print(f"[INFO] Overriding velocity command: vx={args_cli.cmd_vx}, vy={args_cli.cmd_vy}")
        # Access the underlying Isaac Lab env to override command manager
        base_env = env.unwrapped
        cmd_mgr = base_env.command_manager
        # The velocity command term stores commands in its .command buffer
        vel_cmd_term = cmd_mgr._terms["base_velocity"]
        original_compute = vel_cmd_term._update_command
        
        def override_command(*a, **kw):
            original_compute(*a, **kw)
            if args_cli.cmd_vx is not None:
                vel_cmd_term.command[:, 0] = args_cli.cmd_vx
            if args_cli.cmd_vy is not None:
                vel_cmd_term.command[:, 1] = args_cli.cmd_vy
        vel_cmd_term._update_command = override_command
        # Also set the initial command
        if args_cli.cmd_vx is not None:
            vel_cmd_term.command[:, 0] = args_cli.cmd_vx
        if args_cli.cmd_vy is not None:
            vel_cmd_term.command[:, 1] = args_cli.cmd_vy

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    vel_log = []

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            raw_action = policy(obs)
            flat_obs = obs_to_flat(obs)
            
            if args_cli.t_end > 0.0:
                action = flow_smooth(flow_model, raw_action, flat_obs, t_end=args_cli.t_end)
            else:
                action = raw_action

            obs, rew, dones, _ = env.step(action)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

            if args_cli.log_vel:
                flat = obs_to_flat(obs)
                vx = flat[0, 0].item()
                vy = flat[0, 1].item()
                vz = flat[0, 2].item()
                act_mag = torch.mean(torch.abs(action)).item()
                raw_mag = torch.mean(torch.abs(raw_action)).item()
                vel_log.append([timestep, vx, vy, vz, act_mag, raw_mag])
                if timestep % 100 == 0:
                    print(f"  step={timestep:4d} vx={vx:+.3f} vy={vy:+.3f} |act|={act_mag:.3f} |raw|={raw_mag:.3f}", flush=True)

        timestep += 1
        if timestep >= args_cli.max_steps:
            break
        if args_cli.video and timestep >= args_cli.video_length:
            break
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.log_vel and vel_log:
        vel_arr = np.array(vel_log)
        out_path = os.path.join(r"A:\IsaacLab", f"vel_log_t{args_cli.t_end}.npz")
        np.savez(out_path, vel=vel_arr,
                 columns=["step","vx","vy","vz","act_mag","raw_mag"],
                 t_end=args_cli.t_end)
        print(f"\n[INFO] Velocity log saved: {out_path}", flush=True)
        print(f"  Mean vx={np.mean(vel_arr[:,1]):.4f}, Mean |act|={np.mean(vel_arr[:,4]):.4f}", flush=True)
        print(f"  Steps with vx > 0.1: {np.sum(vel_arr[:,1] > 0.1)} / {len(vel_arr)}", flush=True)
        print(f"  Steps with vx > 0.3: {np.sum(vel_arr[:,1] > 0.3)} / {len(vel_arr)}", flush=True)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
