"""Record rough terrain video with camera tracking the robot.
Based on push_video_eval.py but with explicit viewer camera configuration."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--cmd_vx", type=float, default=1.0)
parser.add_argument("--flow_model", type=str, default=r"A:\AllIsaac\IsaacLab\flow_model_rough.pt")
parser.add_argument("--state_dim", type=int, default=235)
parser.add_argument("--t_end", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video_folder", type=str, required=True)
parser.add_argument("--video_length", type=int, default=300)

import cli_args
cli_args.add_rsl_rl_args(parser)
# cli_args adds --checkpoint; set default after parsing if not provided
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import torch, numpy as np, gymnasium as gym, time
installed_version = metadata.version("rsl-rl-lib")

import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner
import torch.nn as nn


class VelocityNet(nn.Module):
    def __init__(self, action_dim=12, state_dim=235, hidden=256):
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
    except: raise TypeError(f"Cannot extract from {type(obs)}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # -- Configure the viewer camera to follow the robot from behind/above --
    env_cfg.viewer.eye = (2.0, 2.0, 2.0)        # camera position offset
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)      # look-at target (robot origin)
    env_cfg.viewer.origin_type = "asset_root"     # track the robot's root
    env_cfg.viewer.asset_name = "robot"           # name of the asset to track
    env_cfg.viewer.env_index = 0                  # which env to follow

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    os.makedirs(args_cli.video_folder, exist_ok=True)
    video_kwargs = {
        "video_folder": args_cli.video_folder,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print(f"[INFO] Recording video to {args_cli.video_folder}", flush=True)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] PPO loaded: {args_cli.checkpoint}", flush=True)

    flow_ckpt = torch.load(args_cli.flow_model, map_location=env.unwrapped.device, weights_only=True)
    if isinstance(flow_ckpt, dict) and "state_dict" in flow_ckpt:
        state_dim = flow_ckpt.get("state_dim", args_cli.state_dim)
        flow_sd = flow_ckpt["state_dict"]
    else:
        state_dim = args_cli.state_dim
        flow_sd = flow_ckpt
    flow_model = VelocityNet(state_dim=state_dim).to(env.unwrapped.device)
    flow_model.load_state_dict(flow_sd)
    flow_model.eval()
    print(f"[INFO] Flow model loaded (state_dim={state_dim}, t_end={args_cli.t_end})", flush=True)

    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    override_cmd = torch.zeros_like(cmd_term.command)
    override_cmd[:, 0] = args_cli.cmd_vx
    def patched_compute(dt):
        cmd_term.command[:] = override_cmd
        return None
    cmd_term.compute = patched_compute

    obs, _ = env.reset()
    cmd_term.command[:] = override_cmd

    fell = False
    t0 = time.time()
    for step in range(args_cli.max_steps):
        flat = obs_to_flat(obs)
        raw_action = policy(obs)
        if args_cli.t_end > 0.0:
            exec_action = flow_smooth(flow_model, raw_action, flat[:, :state_dim], t_end=args_cli.t_end)
        else:
            exec_action = raw_action

        step_ret = env.step(exec_action)
        obs = step_ret[0]
        cmd_term.command[:] = override_cmd

        try:
            dones = env.unwrapped.termination_manager.dones
            time_outs = env.unwrapped.termination_manager.time_outs
            if (dones & (~time_outs)).any().item():
                if not fell:
                    print(f"[INFO] FELL at step {step}", flush=True)
                fell = True
        except Exception:
            pass

    wall = time.time() - t0
    controller = "smoothed" if args_cli.t_end > 0 else "raw"
    print(f"[RESULT] wall={wall:.1f}s  fell={fell}  controller={controller}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
