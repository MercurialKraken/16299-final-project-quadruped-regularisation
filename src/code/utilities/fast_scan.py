"""Fast discordant seed scanner.
Uses a SINGLE environment with multiple envs to test many seeds at once."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Isaac-Velocity-Flat-Unitree-Go1-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--push_step", type=int, default=80)
parser.add_argument("--push_dur",  type=int, default=10)
parser.add_argument("--mag", type=float, default=300.0)
parser.add_argument("--cmd_vx", type=float, default=1.0)
parser.add_argument("--flow_model", type=str,
                    default=r"A:\AllIsaac\IsaacLab\flow_model_adaptive.pt")
parser.add_argument("--state_dim", type=int, default=48)
parser.add_argument("--seed", type=int, default=42)

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


def run_episode(env, policy, flow_model, state_dim, args, use_smooth):
    """Run one episode with all N envs. Return (fell[N], fall_step[N])."""
    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    override_cmd = torch.zeros_like(cmd_term.command)
    override_cmd[:, 0] = args.cmd_vx
    cmd_term.command[:] = override_cmd

    N = args.num_envs
    device = env.unwrapped.device
    body_names = robot.body_names
    base_id = 0
    for cand in ("base", "trunk"):
        if cand in body_names:
            base_id = body_names.index(cand); break
    num_bodies = len(body_names)
    fz = torch.zeros((N, num_bodies, 3), device=device)
    tz = torch.zeros((N, num_bodies, 3), device=device)
    fp = fz.clone()
    fp[:, base_id, 1] = args.mag
    pw = (args.push_step, args.push_step + args.push_dur)

    obs, _ = env.reset()
    cmd_term.command[:] = override_cmd

    fell = torch.zeros(N, dtype=torch.bool, device=device)
    fall_step = torch.full((N,), -1, dtype=torch.long, device=device)

    for step in range(args.max_steps):
        flat = obs_to_flat(obs)
        raw_action = policy(obs)
        if use_smooth:
            exec_action = flow_smooth(flow_model, raw_action,
                                      flat[:, :state_dim])
        else:
            exec_action = raw_action

        if pw[0] <= step < pw[1]:
            robot.set_external_force_and_torque(fp, tz)
        else:
            robot.set_external_force_and_torque(fz, tz)

        obs = env.step(exec_action)[0]
        cmd_term.command[:] = override_cmd

        try:
            dones = env.unwrapped.termination_manager.dones
            time_outs = env.unwrapped.termination_manager.time_outs
            new_fall = dones & (~time_outs) & (~fell)
            if new_fall.any():
                fell |= new_fall
                fall_step[new_fall] = step
        except Exception:
            pass

    return fell.cpu().numpy(), fall_step.cpu().numpy()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    dev = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.sim.device = dev

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] PPO loaded", flush=True)

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
    print(f"[INFO] Flow model loaded", flush=True)

    N = args_cli.num_envs
    print(f"[INFO] Running {N} envs at mag={args_cli.mag}N ...", flush=True)

    # Raw episode
    t0 = time.time()
    raw_fell, raw_fs = run_episode(env, policy, flow_model, state_dim,
                                    args_cli, False)
    t1 = time.time()
    print(f"[INFO] Raw done in {t1-t0:.1f}s: {raw_fell.sum()}/{N} fell",
          flush=True)

    # Smoothed episode
    smo_fell, smo_fs = run_episode(env, policy, flow_model, state_dim,
                                    args_cli, True)
    t2 = time.time()
    print(f"[INFO] Smo done in {t2-t1:.1f}s: {smo_fell.sum()}/{N} fell",
          flush=True)

    # Find discordant
    disc_b = raw_fell & ~smo_fell  # raw fell, smo OK
    disc_c = ~raw_fell & smo_fell  # raw OK, smo fell
    b_idx = np.where(disc_b)[0]
    c_idx = np.where(disc_c)[0]

    print(f"\n=== DISCORDANT (raw fell, smo OK): {len(b_idx)} envs ===",
          flush=True)
    for i in b_idx:
        print(f"  env {i}: raw fell at step {raw_fs[i]}", flush=True)

    print(f"\n=== REVERSE (raw OK, smo fell): {len(c_idx)} envs ===",
          flush=True)
    for i in c_idx:
        print(f"  env {i}: smo fell at step {smo_fs[i]}", flush=True)

    print(f"\n[SUMMARY] seed={args_cli.seed} mag={args_cli.mag}N "
          f"raw_fell={raw_fell.sum()}/{N} smo_fell={smo_fell.sum()}/{N} "
          f"disc_b={len(b_idx)} disc_c={len(c_idx)}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
