"""
rough_terrain_eval.py -- Head-to-head evaluation of raw vs flow-smoothed PPO on rough terrain.
Collects velocity tracking, action smoothness, energy, and fall metrics.
Usage: _isaac_sim\python.bat rough_terrain_eval.py --headless --num_envs 1 --max_steps 1000 --num_episodes 5
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--t_end", type=float, default=0.3)
parser.add_argument("--cmd_vx", type=float, default=1.0)
parser.add_argument("--flow_model", type=str, default=r"A:\AllIsaac\IsaacLab\flow_model_rough.pt")
parser.add_argument("--mode", type=str, choices=["raw", "smoothed", "both"], default="both")
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--video_folder", type=str, default=r"A:\AllIsaac\IsaacLab\rough_videos")
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.record_video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import torch, numpy as np, gymnasium as gym, time, json
installed_version = metadata.version("rsl-rl-lib")

import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner
import torch.nn as nn

CHECKPOINT = r"A:\AllIsaac\IsaacLab\logs\rsl_rl\unitree_go1_rough\2026-04-13_14-04-33\model_100.pt"
OUT_DIR = r"A:\AllIsaac\IsaacLab"

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

def run_episodes(env, policy, flow_model, state_dim, use_flow, t_end, num_eps, max_steps, cmd_vx):
    """Run episodes and collect per-step metrics."""
    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    override_cmd = torch.zeros_like(cmd_term.command)
    override_cmd[:, 0] = cmd_vx
    def patched_compute(dt):
        cmd_term.command[:] = override_cmd
        return None
    cmd_term.compute = patched_compute

    all_vx, all_vy, all_vz = [], [], []
    all_actions, all_torques = [], []
    ep_returns, ep_lengths, ep_falls = [], [], []

    for ep in range(num_eps):
        obs, _ = env.reset()
        cmd_term.command[:] = override_cmd
        ep_ret, ep_len = 0.0, 0
        fell = False
        prev_action = None

        for step in range(max_steps):
            with torch.inference_mode():
                flat = obs_to_flat(obs)
                raw_action = policy(obs)
            with torch.inference_mode():
                if use_flow and flow_model is not None:
                    exec_action = flow_smooth(flow_model, raw_action, flat[:, :state_dim], t_end=t_end)
                else:
                    exec_action = raw_action

            act_np = exec_action.detach().cpu().numpy()[0]
            all_actions.append(act_np.copy())

            # Log velocities from observation
            flat_np = flat.cpu().numpy()[0]
            all_vx.append(flat_np[0])
            all_vy.append(flat_np[1])
            all_vz.append(flat_np[2])

            # Get torques
            try:
                torques = robot.data.applied_torque.cpu().numpy()[0]
                all_torques.append(torques.copy())
            except:
                all_torques.append(np.zeros(12))

            obs, reward, dones, extras = env.step(exec_action)
            cmd_term.command[:] = override_cmd
            ep_ret += float(reward[0])
            ep_len += 1

            try:
                d = env.unwrapped.termination_manager.dones
                t = env.unwrapped.termination_manager.time_outs
                if (d & (~t)).any().item():
                    fell = True
                    break
            except: pass
            if dones.any():
                break

        print(f"  [{('SMO' if use_flow else 'RAW')}] Ep {ep+1}/{num_eps} steps={ep_len} ret={ep_ret:.1f} fell={fell}")
        ep_returns.append(ep_ret); ep_lengths.append(ep_len); ep_falls.append(fell)

    actions = np.array(all_actions)
    torques = np.array(all_torques)
    vx = np.array(all_vx); vy = np.array(all_vy); vz = np.array(all_vz)

    # Compute metrics
    if len(actions) > 1:
        ar = np.diff(actions, axis=0)
        action_rate_rms = float(np.sqrt(np.mean(ar**2)))
    else:
        action_rate_rms = 0.0
    mean_vx = float(np.mean(vx))
    mean_abs_vy = float(np.mean(np.abs(vy)))
    mean_abs_vz = float(np.mean(np.abs(vz)))
    torque_rms = float(np.sqrt(np.mean(torques**2))) if len(torques) > 0 else 0.0
    # Power = |torque * joint_vel| approximation
    mean_ret = float(np.mean(ep_returns))
    fall_rate = sum(ep_falls) / len(ep_falls)
    mean_ep_len = float(np.mean(ep_lengths))

    results = {
        "mode": "smoothed" if use_flow else "raw",
        "action_rate_rms": action_rate_rms,
        "mean_vx": mean_vx,
        "mean_abs_vy": mean_abs_vy,
        "mean_abs_vz": mean_abs_vz,
        "torque_rms": torque_rms,
        "mean_return": mean_ret,
        "fall_rate": fall_rate,
        "mean_ep_length": mean_ep_len,
        "ep_returns": ep_returns,
        "ep_lengths": ep_lengths,
        "ep_falls": ep_falls,
    }
    # Save per-step data
    tag = "smo" if use_flow else "raw"
    npz_path = os.path.join(OUT_DIR, f"rough_eval_{tag}.npz")
    np.savez(npz_path, actions=actions, torques=torques, vx=vx, vy=vy, vz=vz,
             ep_returns=np.array(ep_returns), ep_lengths=np.array(ep_lengths))
    print(f"  Saved step data -> {npz_path}")
    return results

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    from packaging import version
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = 42
    agent_cfg.seed = 42
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.record_video:
        os.makedirs(args_cli.video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(env,
            video_folder=args_cli.video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.max_steps,
            disable_logger=True,
            name_prefix="rough")
    
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(CHECKPOINT)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] PPO loaded: {CHECKPOINT}")

    # Load flow model
    flow_model = None
    state_dim = 235
    if args_cli.mode in ("smoothed", "both") and os.path.exists(args_cli.flow_model):
        flow_ckpt = torch.load(args_cli.flow_model, map_location=env.unwrapped.device, weights_only=True)
        if isinstance(flow_ckpt, dict) and "state_dict" in flow_ckpt:
            state_dim = flow_ckpt.get("state_dim", 235)
            flow_sd = flow_ckpt["state_dict"]
        else:
            flow_sd = flow_ckpt
        flow_model = VelocityNet(state_dim=state_dim).to(env.unwrapped.device)
        flow_model.load_state_dict(flow_sd)
        flow_model.eval()
        print(f"[INFO] Flow model loaded (state_dim={state_dim})")
    elif args_cli.mode in ("smoothed", "both"):
        print(f"[WARN] Flow model not found at {args_cli.flow_model}, running raw only")
        args_cli.mode = "raw"

    all_results = {}
    modes = []
    if args_cli.mode in ("raw", "both"): modes.append(("raw", False))
    if args_cli.mode in ("smoothed", "both"): modes.append(("smoothed", True))

    for label, use_flow in modes:
        print(f"\n=== Running {label.upper()} ===")
        r = run_episodes(env, policy, flow_model, state_dim, use_flow,
                         args_cli.t_end, args_cli.num_episodes, args_cli.max_steps, args_cli.cmd_vx)
        all_results[label] = r

    # Print comparison table
    print("\n" + "="*70)
    print("ROUGH TERRAIN HEAD-TO-HEAD RESULTS")
    print("="*70)
    hdr = f"{'Metric':<25} "
    for label in all_results:
        hdr += f"| {label:>12} "
    print(hdr)
    print("-"*70)
    metrics = ["action_rate_rms", "mean_vx", "mean_abs_vy", "mean_abs_vz",
               "torque_rms", "mean_return", "fall_rate", "mean_ep_length"]
    for m in metrics:
        row = f"{m:<25} "
        for label in all_results:
            row += f"| {all_results[label][m]:>12.4f} "
        print(row)
    print("="*70)

    # Save results JSON
    json_path = os.path.join(OUT_DIR, "rough_eval_results.json")
    # Convert non-serializable types
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {mk: mv if not isinstance(mv, list) else mv for mk, mv in v.items()}
    with open(json_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved -> {json_path}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
