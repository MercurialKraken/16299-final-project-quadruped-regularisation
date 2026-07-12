"""Effort eval: log per-step joint torques, joint velocities, and executed
actions across N parallel envs under a constant forward velocity command."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go1-v0")
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--cmd_vx", type=float, default=1.0)
parser.add_argument("--flow_model", type=str, required=True)
parser.add_argument("--state_dim", type=int, default=48)
parser.add_argument("--t_end", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--results_path", type=str, default="A:/IsaacLab/effort_results.npz")

import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import torch, numpy as np, gymnasium as gym, time
installed_version = metadata.version("rsl-rl-lib")

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
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
    start_pos = robot.data.root_pos_w[:, :2].clone().cpu().numpy()
    print(f"[INFO] {args_cli.num_envs} envs, start_pos[0]={start_pos[0]}", flush=True)

    dt_env = env.unwrapped.step_dt
    N, T, A = args_cli.num_envs, args_cli.max_steps, 12
    actions_log = np.zeros((T, N, A), dtype=np.float32)
    torque_log  = np.zeros((T, N, A), dtype=np.float32)
    jvel_log    = np.zeros((T, N, A), dtype=np.float32)
    pos_log     = np.zeros((T, N, 2), dtype=np.float32)
    fell        = torch.zeros(N, dtype=torch.bool, device=env.unwrapped.device)
    fall_step   = torch.full((N,), -1, dtype=torch.long, device=env.unwrapped.device)
    final_pos   = robot.data.root_pos_w[:, :2].clone()

    t0 = time.time()
    for step in range(T):
        flat = obs_to_flat(obs)
        raw_action = policy(obs)
        if args_cli.t_end > 0.0:
            exec_action = flow_smooth(flow_model, raw_action, flat[:, :state_dim], t_end=args_cli.t_end)
        else:
            exec_action = raw_action

        actions_log[step] = exec_action.detach().cpu().numpy()
        # Take step
        step_ret = env.step(exec_action)
        obs = step_ret[0]
        # Log post-step state
        torque_log[step] = robot.data.applied_torque.detach().cpu().numpy()
        jvel_log[step]   = robot.data.joint_vel.detach().cpu().numpy()
        pos_log[step]    = robot.data.root_pos_w[:, :2].detach().cpu().numpy()
        try:
            dones = env.unwrapped.termination_manager.dones
        except Exception:
            dones = torch.zeros(N, dtype=torch.bool, device=env.unwrapped.device)
        new_fell = dones & (~fell)
        if (~fell).any():
            cur = robot.data.root_pos_w[:, :2]
            alive = ~fell
            final_pos[alive] = cur[alive]
        if new_fell.any():
            fall_step[new_fell] = step
        fell = fell | dones
        cmd_term.command[:] = override_cmd

    wall = time.time() - t0
    print(f"[RESULT] {N} envs x {T} steps = {N*T} total samples, wall={wall:.1f}s", flush=True)

    np.savez(
        args_cli.results_path,
        actions=actions_log,
        torque=torque_log,
        joint_vel=jvel_log,
        pos=pos_log,
        start_pos=start_pos,
        final_pos=final_pos.cpu().numpy(),
        fell=fell.cpu().numpy(),
        fall_step=fall_step.cpu().numpy(),
        cmd_vx=args_cli.cmd_vx,
        t_end=args_cli.t_end,
        dt=dt_env,
        max_steps=T,
        seed=args_cli.seed,
        num_envs=N,
    )
    print(f"[INFO] Results saved: {args_cli.results_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
