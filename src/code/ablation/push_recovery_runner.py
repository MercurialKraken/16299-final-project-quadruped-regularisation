"""Unified push-recovery runner supporting four inference modes:

  --mode raw           : raw PPO action
  --mode flow          : Bal-LP flow model with t_end
  --mode lp            : online causal Butterworth IIR (15 Hz, 1st-order)
  --mode flow_lp       : flow first, then causal IIR LP

For each magnitude bucket, applies a lateral force impulse to the trunk and
records fall rate. 80 envs by default.

Output: push_<variant>.npz   (compatible with analyze_ablation.py)
"""
import argparse
import os
import sys

sys.path.insert(0, r"A:\AllIsaac\IsaacLab\scripts\reinforcement_learning\rsl_rl")

from isaaclab.app import AppLauncher
import cli_args  # type: ignore

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go1-v0")
ap.add_argument("--variant", type=str, required=True)
ap.add_argument("--mode", type=str, required=True,
                choices=["raw", "flow", "lp", "flow_lp"])
ap.add_argument("--num_envs", type=int, default=80)
ap.add_argument("--max_steps", type=int, default=500)
ap.add_argument("--push_step", type=int, default=100)
ap.add_argument("--push_dur", type=int, default=10)
ap.add_argument("--mag_min", type=float, default=50.0)
ap.add_argument("--mag_max", type=float, default=500.0)
ap.add_argument("--mag_steps", type=int, default=10)
ap.add_argument("--cmd_vx", type=float, default=1.0)
ap.add_argument("--flow_model", type=str, default=None,
                help="Required for mode=flow or flow_lp")
ap.add_argument("--state_dim", type=int, default=48)
ap.add_argument("--t_end", type=float, default=1.0)
ap.add_argument("--n_flow_steps", type=int, default=20)
ap.add_argument("--lp_cutoff_hz", type=float, default=15.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out_dir", type=str,
                default=r"A:\AllIsaac\flow_matching_project\data\ablation")

cli_args.add_rsl_rl_args(ap)
AppLauncher.add_app_launcher_args(ap)
args_cli, hydra_args = ap.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import json
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import register_reg_variants  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")


# ---------- model definitions ----------------------------------------------

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
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x_t, t, state], dim=-1))


@torch.no_grad()
def flow_smooth(model, x0, state, n_steps, t_end):
    x = x0.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = torch.full((x.shape[0], 1), i * dt, device=x.device)
        x = x + model(x, t_val, state) * dt
    return x


# ---------- causal IIR low-pass (1st-order Butterworth-ish) -----------------
# y[n] = (1-alpha) * y[n-1] + alpha * x[n]
# alpha = dt / (RC + dt) where RC = 1/(2*pi*fc)

class CausalIIRLP:
    def __init__(self, num_envs, action_dim, fs, fc):
        rc = 1.0 / (2.0 * np.pi * fc)
        dt = 1.0 / fs
        self.alpha = float(dt / (rc + dt))
        self.y = None  # initialized lazily on first call
        self._shape = (num_envs, action_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.y is None:
            self.y = x.clone()
        else:
            self.y = (1.0 - self.alpha) * self.y + self.alpha * x
        return self.y


def obs_to_flat(o):
    if isinstance(o, torch.Tensor):
        return o
    try:
        return o["policy"]
    except Exception:
        return next(iter(o.values()))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[push] policy loaded: {args_cli.checkpoint}", flush=True)

    flow_model = None
    if args_cli.mode in ("flow", "flow_lp"):
        if not args_cli.flow_model:
            raise SystemExit("flow_model required for mode=flow / flow_lp")
        ckpt = torch.load(args_cli.flow_model, map_location=env.unwrapped.device,
                          weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dim = ckpt.get("state_dim", args_cli.state_dim)
            sd = ckpt["state_dict"]
        else:
            state_dim = args_cli.state_dim
            sd = ckpt
        flow_model = VelocityNet(state_dim=state_dim).to(env.unwrapped.device)
        flow_model.load_state_dict(sd)
        flow_model.eval()
        print(f"[push] flow model loaded: {args_cli.flow_model} state_dim={state_dim}",
              flush=True)
    else:
        state_dim = args_cli.state_dim

    iir = None
    if args_cli.mode in ("lp", "flow_lp"):
        iir = CausalIIRLP(args_cli.num_envs, 12, fs=50.0, fc=args_cli.lp_cutoff_hz)
        print(f"[push] causal IIR LP fc={args_cli.lp_cutoff_hz}Hz alpha={iir.alpha:.4f}",
              flush=True)

    # pin command
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    override = torch.zeros_like(cmd_term.command)
    override[:, 0] = args_cli.cmd_vx

    def _patched_compute(dt):
        cmd_term.command.copy_(override)
    cmd_term.compute = _patched_compute

    obs, _ = env.reset()
    cmd_term.command[:] = override

    N = args_cli.num_envs
    device = env.unwrapped.device
    n_per_mag = N // args_cli.mag_steps
    mags = torch.linspace(args_cli.mag_min, args_cli.mag_max, args_cli.mag_steps,
                          dtype=torch.float32, device=device)
    per_env_mag = mags.repeat_interleave(n_per_mag)[:N]
    print(f"[push] magnitudes: {mags.cpu().tolist()}  ({n_per_mag} envs each)", flush=True)

    robot = env.unwrapped.scene["robot"]
    body_names = robot.body_names
    base_id = 0
    for cand in ("base", "trunk"):
        if cand in body_names:
            base_id = body_names.index(cand)
            break
    num_bodies = len(body_names)
    forces_zero = torch.zeros((N, num_bodies, 3), device=device)
    torques_zero = torch.zeros((N, num_bodies, 3), device=device)
    forces_push = forces_zero.clone()
    forces_push[:, base_id, 1] = per_env_mag

    fell = torch.zeros(N, dtype=torch.bool, device=device)
    fall_step = torch.full((N,), -1, dtype=torch.long, device=device)
    push_window = (args_cli.push_step, args_cli.push_step + args_cli.push_dur)

    # latency profiling for the smoothing block
    lat_samples = []

    t0 = time.time()
    for step in range(args_cli.max_steps):
        flat = obs_to_flat(obs)
        with torch.inference_mode():
            raw_action = policy(obs)

        # ---- smoothing block (timed) ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bt = time.perf_counter()
        a = raw_action
        if flow_model is not None:
            a = flow_smooth(flow_model, a, flat[:, :state_dim],
                            n_steps=args_cli.n_flow_steps, t_end=args_cli.t_end)
        if iir is not None:
            a = iir(a)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_samples.append((time.perf_counter() - bt) * 1000.0)
        # ---------------------------------

        if push_window[0] <= step < push_window[1]:
            robot.set_external_force_and_torque(forces_push, torques_zero)
        else:
            robot.set_external_force_and_torque(forces_zero, torques_zero)

        step_ret = env.step(a)
        obs = step_ret[0]
        cmd_term.command[:] = override

        try:
            dones = env.unwrapped.termination_manager.dones
            time_outs = env.unwrapped.termination_manager.time_outs
            true_falls = dones & (~time_outs)
        except Exception:
            true_falls = step_ret[2]
        new_fell = true_falls & (~fell)
        if new_fell.any():
            fall_step[new_fell] = step
        fell = fell | true_falls

    wall = time.time() - t0
    fell_np = fell.cpu().numpy()
    fall_step_np = fall_step.cpu().numpy()
    mags_np = per_env_mag.cpu().numpy()

    print(f"[push] mode={args_cli.mode}  wall={wall:.1f}s  fell={int(fell_np.sum())}/{N}",
          flush=True)
    for m_val in mags.cpu().numpy():
        idx = np.isclose(mags_np, m_val)
        n_at = int(idx.sum())
        n_fell = int(fell_np[idx].sum())
        print(f"  F={m_val:.0f} N: {n_fell}/{n_at}", flush=True)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    out_npz = os.path.join(args_cli.out_dir, f"push_{args_cli.variant}.npz")
    np.savez(out_npz, fell=fell_np, fall_step=fall_step_np, magnitudes=mags_np,
             num_envs=N, mode=args_cli.mode, t_end=args_cli.t_end,
             lp_cutoff_hz=args_cli.lp_cutoff_hz)
    print(f"[push] saved -> {out_npz}", flush=True)

    # latency report
    if len(lat_samples) > 50:
        arr = np.array(lat_samples[50:])  # drop warmup
        infer_meta = {
            "variant": args_cli.variant,
            "mode": args_cli.mode,
            "n_samples": int(arr.size),
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }
        with open(os.path.join(args_cli.out_dir, f"inference_{args_cli.variant}.json"),
                  "w") as f:
            json.dump(infer_meta, f, indent=2)
        print(f"[push] latency mean={infer_meta['mean_ms']:.3f}ms  p95={infer_meta['p95_ms']:.3f}ms",
              flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
