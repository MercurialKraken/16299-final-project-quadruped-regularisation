"""Scan seeds to find discordant push outcomes (raw falls, smoothed survives).
Runs headless, no video, 1 env. Tests both raw and smoothed per seed."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Isaac-Velocity-Flat-Unitree-Go1-v0")
parser.add_argument("--max_steps", type=int, default=350)
parser.add_argument("--push_step", type=int, default=100)
parser.add_argument("--push_dur",  type=int, default=10)
parser.add_argument("--mag", type=float, default=250.0)
parser.add_argument("--cmd_vx", type=float, default=1.0)
parser.add_argument("--flow_model", type=str,
                    default=r"A:\AllIsaac\IsaacLab\flow_model_adaptive.pt")
parser.add_argument("--state_dim", type=int, default=48)
parser.add_argument("--seed_start", type=int, default=0)
parser.add_argument("--seed_end", type=int, default=100)

import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
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
def flow_smooth(model, raw_action, state, n_steps=20, t_end=0.3):
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


def run_one_episode(env, policy, flow_model, state_dim, args, use_smooth):
    """Run one push episode. Returns (fell: bool, fall_step: int or -1)."""
    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    override_cmd = torch.zeros_like(cmd_term.command)
    override_cmd[:, 0] = args.cmd_vx
    cmd_term.command[:] = override_cmd

    N = 1
    device = env.unwrapped.device
    body_names = robot.body_names
    base_id = 0
    for cand in ("base", "trunk"):
        if cand in body_names:
            base_id = body_names.index(cand); break
    num_bodies = len(body_names)
    forces_zero  = torch.zeros((N, num_bodies, 3), device=device)
    torques_zero = torch.zeros((N, num_bodies, 3), device=device)
    forces_push  = forces_zero.clone()
    forces_push[:, base_id, 1] = args.mag
    push_window = (args.push_step, args.push_step + args.push_dur)

    obs, _ = env.reset()
    cmd_term.command[:] = override_cmd

    fell = False
    fall_step = -1
    for step in range(args.max_steps):
        flat = obs_to_flat(obs)
        raw_action = policy(obs)
        if use_smooth:
            exec_action = flow_smooth(flow_model, raw_action,
                                      flat[:, :state_dim])
        else:
            exec_action = raw_action

        if push_window[0] <= step < push_window[1]:
            robot.set_external_force_and_torque(forces_push, torques_zero)
        else:
            robot.set_external_force_and_torque(forces_zero, torques_zero)

        obs = env.step(exec_action)[0]
        cmd_term.command[:] = override_cmd

        try:
            dones = env.unwrapped.termination_manager.dones
            time_outs = env.unwrapped.termination_manager.time_outs
            if (dones & (~time_outs)).any().item():
                if not fell:
                    fell = True
                    fall_step = step
        except Exception:
            pass

    return fell, fall_step


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.sim.device = (args_cli.device if args_cli.device is not None
                          else env_cfg.sim.device)

    # Load flow model once
    device_str = env_cfg.sim.device
    flow_ckpt = torch.load(args_cli.flow_model, map_location=device_str,
                           weights_only=True)
    if isinstance(flow_ckpt, dict) and "state_dict" in flow_ckpt:
        state_dim = flow_ckpt.get("state_dim", args_cli.state_dim)
        flow_sd = flow_ckpt["state_dict"]
    else:
        state_dim = args_cli.state_dim
        flow_sd = flow_ckpt
    flow_model = VelocityNet(state_dim=state_dim).to(device_str)
    flow_model.load_state_dict(flow_sd)
    flow_model.eval()
    print(f"[INFO] Flow model loaded (state_dim={state_dim})", flush=True)

    found = []
    for seed in range(args_cli.seed_start, args_cli.seed_end):
        env_cfg.seed = seed
        agent_cfg.seed = seed
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                                device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        raw_fell, raw_step = run_one_episode(env, policy, flow_model,
                                              state_dim, args_cli, False)
        smo_fell, smo_step = run_one_episode(env, policy, flow_model,
                                              state_dim, args_cli, True)

        tag = ""
        if raw_fell and not smo_fell:
            tag = " *** DISCORDANT (raw fell, smo OK) ***"
            found.append(seed)
        elif not raw_fell and smo_fell:
            tag = " (reverse discordant)"

        print(f"seed={seed:3d}  raw={'FELL@'+str(raw_step):>10s}"
              f"  smo={'FELL@'+str(smo_step) if smo_fell else 'OK':>10s}"
              f"{tag}", flush=True)

        env.close()

        if len(found) >= 3:
            print(f"\n=== Found {len(found)} discordant seeds: {found} ===",
                  flush=True)
            break

    if found:
        print(f"\n=== BEST SEED: {found[0]} at mag={args_cli.mag} N ===",
              flush=True)
    else:
        print(f"\n=== No discordant seeds found in "
              f"[{args_cli.seed_start}, {args_cli.seed_end}) ===", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
