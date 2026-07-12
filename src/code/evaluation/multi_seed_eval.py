"""Multi-seed eval: run N parallel Isaac envs under a constant forward
velocity command for max_steps, then save per-env final_x progress.

Run twice (raw vs smoothed) with same --seed and the parallel envs will
start from the same per-env spawns so we get *paired* samples for stats.
"""
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
parser.add_argument("--results_path", type=str, default="A:/IsaacLab/multiseed_results.npz")

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
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
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
    start_pos = robot.data.root_pos_w[:, :2].clone().cpu().numpy()  # (N, 2)
    print(f"[INFO] {args_cli.num_envs} envs, start_pos[0]={start_pos[0]}, start_pos[-1]={start_pos[-1]}", flush=True)


    # Roll out
    dt_env = env.unwrapped.step_dt
    t0 = time.time()
    # Track per-env termination so we don't reward post-fall drift
    fell = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    fall_step = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=env.unwrapped.device)
    final_pos = robot.data.root_pos_w[:, :2].clone()
    final_step_used = torch.full((args_cli.num_envs,), args_cli.max_steps, dtype=torch.long,
                                 device=env.unwrapped.device)

    for step in range(args_cli.max_steps):
        flat = obs_to_flat(obs)
        raw_action = policy(obs)
        if args_cli.t_end > 0.0:
            exec_action = flow_smooth(flow_model, raw_action, flat[:, :state_dim], t_end=args_cli.t_end)
        else:
            exec_action = raw_action
        step_ret = env.step(exec_action)
        obs = step_ret[0]
        # auto-reset detection: in Isaac vec env, dones cause auto-reset within step
        # Use the wrapped env's last "dones" if exposed; otherwise rely on env-side reset_buf
        try:
            dones = env.unwrapped.termination_manager.dones  # bool tensor (N,)
        except Exception:
            dones = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        cmd_term.command[:] = override_cmd
        # Capture *current* position (after step). For envs that just terminated we
        # freeze final_pos at the value from one step ago — but Isaac auto-resets in
        # step(), so root_pos_w is already the post-reset value. Best we can do:
        # mark the env as fallen and stop updating final_pos.
        new_fell = dones & (~fell)
        # update final_pos for still-alive envs (use pre-update value cached above)
        alive_now = ~fell
        cur_pos = robot.data.root_pos_w[:, :2]
        if alive_now.any():
            final_pos[alive_now] = cur_pos[alive_now]
            final_step_used[alive_now] = step + 1
        if new_fell.any():
            fall_step[new_fell] = step
        fell = fell | dones

    wall = time.time() - t0
    final_pos_np = final_pos.cpu().numpy()
    fell_np = fell.cpu().numpy()
    fall_step_np = fall_step.cpu().numpy()
    final_step_np = final_step_used.cpu().numpy()
    x_progress = final_pos_np[:, 0] - start_pos[:, 0]
    print(f"[RESULT] num_envs={args_cli.num_envs} max_steps={args_cli.max_steps} "
          f"dt={dt_env:.4f}s sim_time={args_cli.max_steps*dt_env:.2f}s wall={wall:.1f}s", flush=True)
    print(f"[RESULT] fell={int(fell_np.sum())}/{args_cli.num_envs}  "
          f"x_progress: mean={x_progress.mean():.3f}  std={x_progress.std():.3f}  "
          f"min={x_progress.min():.3f}  max={x_progress.max():.3f}", flush=True)

    np.savez(
        args_cli.results_path,
        x_progress=x_progress,
        final_pos=final_pos_np,
        start_pos=start_pos,
        fell=fell_np,
        fall_step=fall_step_np,
        final_step_used=final_step_np,
        cmd_vx=args_cli.cmd_vx,
        t_end=args_cli.t_end,
        dt=dt_env,
        max_steps=args_cli.max_steps,
        seed=args_cli.seed,
        num_envs=args_cli.num_envs,
    )
    print(f"[INFO] Results saved: {args_cli.results_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
