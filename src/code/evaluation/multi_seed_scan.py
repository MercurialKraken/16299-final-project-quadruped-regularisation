"""Test multiple seeds with 1 env each by resetting.
Faster than creating new envs each time."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Isaac-Velocity-Flat-Unitree-Go1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--push_step", type=int, default=80)
parser.add_argument("--push_dur",  type=int, default=10)
parser.add_argument("--mag", type=float, default=300.0)
parser.add_argument("--cmd_vx", type=float, default=1.0)
parser.add_argument("--flow_model", type=str,
                    default=r"A:\AllIsaac\IsaacLab\flow_model_adaptive.pt")
parser.add_argument("--state_dim", type=int, default=48)
parser.add_argument("--seed_start", type=int, default=0)
parser.add_argument("--seed_end", type=int, default=50)

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

import isaaclab_tasks
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


def run_ep(env, policy, flow_model, state_dim, args, use_smooth):
    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    oc = torch.zeros_like(cmd_term.command)
    oc[:, 0] = args.cmd_vx

    N = 1
    device = env.unwrapped.device
    body_names = robot.body_names
    base_id = 0
    for c in ("base", "trunk"):
        if c in body_names: base_id = body_names.index(c); break
    nb = len(body_names)
    fz = torch.zeros((N, nb, 3), device=device)
    tz = torch.zeros((N, nb, 3), device=device)
    fp = fz.clone(); fp[:, base_id, 1] = args.mag
    pw = (args.push_step, args.push_step + args.push_dur)

    obs, _ = env.reset()
    cmd_term.command[:] = oc

    fell = False; fs = -1
    for step in range(args.max_steps):
        flat = obs_to_flat(obs)
        raw = policy(obs)
        act = flow_smooth(flow_model, raw, flat[:, :state_dim]) if use_smooth else raw
        if pw[0] <= step < pw[1]:
            robot.set_external_force_and_torque(fp, tz)
        else:
            robot.set_external_force_and_torque(fz, tz)
        obs = env.step(act)[0]
        cmd_term.command[:] = oc
        try:
            d = env.unwrapped.termination_manager.dones
            to = env.unwrapped.termination_manager.time_outs
            if (d & ~to).any().item() and not fell:
                fell = True; fs = step
        except: pass
    return fell, fs


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = args_cli.seed_start
    agent_cfg.seed = args_cli.seed_start
    dev = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.sim.device = dev

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    flow_ckpt = torch.load(args_cli.flow_model, map_location=dev,
                           weights_only=True)
    if isinstance(flow_ckpt, dict) and "state_dict" in flow_ckpt:
        state_dim = flow_ckpt.get("state_dim", args_cli.state_dim)
        flow_sd = flow_ckpt["state_dict"]
    else:
        state_dim = args_cli.state_dim
        flow_sd = flow_ckpt
    flow_model = VelocityNet(state_dim=state_dim).to(dev)
    flow_model.load_state_dict(flow_sd)
    flow_model.eval()

    print(f"[INFO] Scanning seeds {args_cli.seed_start}-{args_cli.seed_end} "
          f"at mag={args_cli.mag}N", flush=True)

    found = []
    for seed in range(args_cli.seed_start, args_cli.seed_end):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rf, rfs = run_ep(env, policy, flow_model, state_dim, args_cli, False)
        sf, sfs = run_ep(env, policy, flow_model, state_dim, args_cli, True)
        tag = ""
        if rf and not sf:
            tag = " *** DISCORDANT ***"
            found.append((seed, rfs))
        elif not rf and sf:
            tag = " (reverse)"
        r_str = f"FELL@{rfs}" if rf else "OK"
        s_str = f"FELL@{sfs}" if sf else "OK"
        print(f"seed={seed:3d}  raw={r_str:>10s}  smo={s_str:>10s}{tag}",
              flush=True)
        if len(found) >= 5:
            break

    print(f"\n=== Found {len(found)} discordant seeds: "
          f"{[(s,f) for s,f in found]} ===", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
