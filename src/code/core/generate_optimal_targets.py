"""
generate_optimal_targets.py — Generate locally-optimal action targets via
short-horizon candidate evaluation (random shooting).

For each recorded timestep, we:
  1. Restore the sim state to all K parallel envs
  2. Apply K candidate actions (PPO action + Gaussian perturbations)
  3. Roll out H steps with the PPO policy
  4. Score each candidate with a multi-objective cost
  5. Select x_star = argmin(cost)

This replaces LP-filtered targets with control-performance-aware targets.

Usage:
    isaaclab.bat -p generate_optimal_targets.py ^
        --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless ^
        --num_candidates 32 --horizon 10 ^
        --rollout_data rollout_states_flat.npz ^
        --checkpoint <path_to_model.pt> ^
        --output optimal_targets_flat.npz
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go1-v0")
parser.add_argument("--num_candidates", "-K", type=int, default=32,
                    help="Number of candidate actions to evaluate per timestep")
parser.add_argument("--horizon", "-H", type=int, default=10,
                    help="Number of simulation steps to roll out each candidate")
parser.add_argument("--sigma", type=float, default=0.1,
                    help="Std dev of Gaussian perturbation around PPO action")
parser.add_argument("--cmd_vx", type=float, default=1.0,
                    help="Commanded forward velocity for cost computation")
parser.add_argument("--rollout_data", type=str, required=True,
                    help="Path to rollout_states_{flat|rough}.npz from collect_rollout_states.py")
parser.add_argument("--output", type=str, required=True,
                    help="Output path for (s, x0, x_star) training dataset")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_timesteps", type=int, default=0,
                    help="Process only first N timesteps (0=all, useful for debugging)")
# Cost function weights
parser.add_argument("--w_tracking", type=float, default=2.0,
                    help="Weight for velocity tracking error")
parser.add_argument("--w_jerk", type=float, default=1.0,
                    help="Weight for action jerk (smoothness)")
parser.add_argument("--w_energy", type=float, default=0.5,
                    help="Weight for energy consumption")
parser.add_argument("--w_stability", type=float, default=5.0,
                    help="Weight for stability (orientation penalty)")

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

import isaaclab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner


def obs_to_flat(obs):
    if isinstance(obs, torch.Tensor):
        return obs
    try:
        return obs["policy"]
    except (KeyError, TypeError):
        pass
    try:
        return next(iter(obs.values()))
    except Exception:
        raise TypeError(f"Cannot extract tensor from obs: {type(obs)}")


def restore_state(robot, root_pos, root_quat, root_lin_vel, root_ang_vel,
                  joint_pos, joint_vel, env_ids=None):
    """Restore full robot state to simulation.

    Args:
        robot: Isaac Lab Articulation asset
        root_pos: (K, 3) world position
        root_quat: (K, 4) quaternion in wxyz format
        root_lin_vel: (K, 3) world-frame linear velocity
        root_ang_vel: (K, 3) world-frame angular velocity
        joint_pos: (K, 12) joint positions
        joint_vel: (K, 12) joint velocities
        env_ids: optional env indices
    """
    device = robot.device
    # Root pose: (K, 7) = [pos(3), quat(4)]
    root_pose = torch.cat([root_pos, root_quat], dim=-1).to(device)
    # Root velocity: (K, 6) = [lin_vel(3), ang_vel(3)]
    root_velocity = torch.cat([root_lin_vel, root_ang_vel], dim=-1).to(device)

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        joint_pos.to(device),
        joint_vel.to(device),
        env_ids=env_ids,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    K = args_cli.num_candidates
    H = args_cli.horizon
    sigma = args_cli.sigma

    # We need K parallel envs to evaluate candidates simultaneously
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = K
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    device = env.unwrapped.device
    print(f"[gen-targets] PPO loaded: {args_cli.checkpoint}")

    robot = env.unwrapped.scene["robot"]

    # Override velocity commands to constant forward
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    override_cmd = torch.zeros_like(cmd_term.command)   # (K, 3)
    override_cmd[:, 0] = args_cli.cmd_vx
    def patched_compute(dt):
        cmd_term.command[:] = override_cmd
        return None
    cmd_term.compute = patched_compute

    # ---- Load rollout data ----
    print(f"[gen-targets] Loading rollout data: {args_cli.rollout_data}")
    data = np.load(args_cli.rollout_data)
    obs_all = data["obs"]              # (T, obs_dim)
    actions_all = data["actions"]      # (T, 12)
    root_pos_all = data["root_pos"]    # (T, 3)
    root_quat_all = data["root_quat"]  # (T, 4)
    root_lv_all = data["root_lin_vel"] # (T, 3)
    root_av_all = data["root_ang_vel"] # (T, 3)
    jpos_all = data["joint_pos"]       # (T, 12)
    jvel_all = data["joint_vel"]       # (T, 12)
    ep_lengths = data["ep_lengths"]

    T_total = len(obs_all)
    if args_cli.max_timesteps > 0:
        T_total = min(T_total, args_cli.max_timesteps)
    obs_dim = obs_all.shape[1]
    action_dim = actions_all.shape[1]

    print(f"[gen-targets] {T_total} timesteps, obs_dim={obs_dim}, K={K}, H={H}, sigma={sigma}")
    print(f"[gen-targets] Cost weights: tracking={args_cli.w_tracking}, jerk={args_cli.w_jerk}, "
          f"energy={args_cli.w_energy}, stability={args_cli.w_stability}")

    # ---- Build episode boundary set (skip first step of each episode — no prev action) ----
    ep_starts = set()
    idx = 0
    for L in ep_lengths:
        ep_starts.add(idx)
        idx += L

    # ---- Main loop: generate x_star for each timestep ----
    x_star_all = np.zeros((T_total, action_dim), dtype=np.float32)
    cost_all = np.zeros(T_total, dtype=np.float32)
    improvement_all = np.zeros(T_total, dtype=np.float32)

    # We also need the action at t-1 for jerk cost
    prev_actions = np.zeros_like(actions_all)
    prev_actions[1:] = actions_all[:-1]
    # At episode boundaries, prev_action = current action (no jerk penalty)
    for s in ep_starts:
        prev_actions[s] = actions_all[s]

    torch.manual_seed(args_cli.seed)
    obs, _ = env.reset()
    cmd_term.command[:] = override_cmd

    t0 = time.time()
    for t_idx in range(T_total):
        if t_idx % 100 == 0 and t_idx > 0:
            elapsed = time.time() - t0
            rate = t_idx / elapsed
            eta = (T_total - t_idx) / rate
            print(f"  [{t_idx}/{T_total}] {rate:.1f} steps/s, ETA {eta:.0f}s", flush=True)

        # Current state and PPO action
        x0 = torch.tensor(actions_all[t_idx], device=device, dtype=torch.float32)      # (12,)
        prev_a = torch.tensor(prev_actions[t_idx], device=device, dtype=torch.float32)  # (12,)

        # State tensors for restoration — broadcast to K envs
        rp = torch.tensor(root_pos_all[t_idx], dtype=torch.float32).unsqueeze(0).expand(K, -1)
        rq = torch.tensor(root_quat_all[t_idx], dtype=torch.float32).unsqueeze(0).expand(K, -1)
        rlv = torch.tensor(root_lv_all[t_idx], dtype=torch.float32).unsqueeze(0).expand(K, -1)
        rav = torch.tensor(root_av_all[t_idx], dtype=torch.float32).unsqueeze(0).expand(K, -1)
        jp = torch.tensor(jpos_all[t_idx], dtype=torch.float32).unsqueeze(0).expand(K, -1)
        jv = torch.tensor(jvel_all[t_idx], dtype=torch.float32).unsqueeze(0).expand(K, -1)

        # ---- 1. Restore state to all K envs ----
        restore_state(robot, rp, rq, rlv, rav, jp, jv)

        # ---- 2. Generate K candidates ----
        # First candidate = the PPO action itself (so x_star is never worse than x0)
        noise = torch.randn(K, action_dim, device=device) * sigma
        noise[0] = 0.0  # candidate 0 = exact PPO action
        candidates = x0.unsqueeze(0).expand(K, -1) + noise  # (K, 12)

        # ---- 3. Roll out H steps and accumulate cost ----
        total_cost = torch.zeros(K, device=device)
        current_action = candidates.clone()
        prev_action_k = prev_a.unsqueeze(0).expand(K, -1)

        # Step 0: apply candidate action
        obs_k, reward_k, dones_k, _ = env.step(candidates)
        cmd_term.command[:] = override_cmd

        # Cost for first step
        vx = robot.data.root_lin_vel_w[:, 0]                          # (K,)
        tracking_err = (vx - args_cli.cmd_vx) ** 2                    # (K,)
        jerk = ((candidates - prev_action_k) ** 2).sum(dim=-1)        # (K,)
        torque = robot.data.applied_torque                             # (K, 12)
        j_vel = robot.data.joint_vel                                   # (K, 12)
        energy = (torque.abs() * j_vel.abs()).sum(dim=-1)              # (K,)
        # Stability: projected gravity z should be close to -1 (upright)
        # In the obs, projected_gravity is at indices 6:9 for flat terrain
        # But we can get it directly from the robot data
        grav_z = robot.data.projected_gravity_b[:, 2]                  # (K,)
        stability = (grav_z + 1.0) ** 2                                # (K,) 0 when upright

        step_cost = (args_cli.w_tracking * tracking_err
                     + args_cli.w_jerk * jerk
                     + args_cli.w_energy * energy
                     + args_cli.w_stability * stability)
        total_cost += step_cost

        prev_action_k = candidates.clone()

        # Steps 1..H-1: use PPO policy
        for h in range(1, H):
            with torch.inference_mode():
                ppo_action = policy(obs_k)
            obs_k, reward_k, dones_k, _ = env.step(ppo_action)
            cmd_term.command[:] = override_cmd

            vx = robot.data.root_lin_vel_w[:, 0]
            tracking_err = (vx - args_cli.cmd_vx) ** 2
            jerk_h = ((ppo_action - prev_action_k) ** 2).sum(dim=-1)
            torque = robot.data.applied_torque
            j_vel = robot.data.joint_vel
            energy = (torque.abs() * j_vel.abs()).sum(dim=-1)
            grav_z = robot.data.projected_gravity_b[:, 2]
            stability = (grav_z + 1.0) ** 2

            # Fall penalty: if an env fell, add large cost for remaining steps
            try:
                fell = env.unwrapped.termination_manager.dones & (~env.unwrapped.termination_manager.time_outs)
                total_cost += fell.float() * 100.0
            except Exception:
                pass

            step_cost = (args_cli.w_tracking * tracking_err
                         + args_cli.w_jerk * jerk_h
                         + args_cli.w_energy * energy
                         + args_cli.w_stability * stability)
            total_cost += step_cost
            prev_action_k = ppo_action.clone()

        # ---- 4. Select best candidate ----
        best_idx = total_cost.argmin().item()
        x_star = candidates[best_idx].detach().cpu().numpy()
        best_cost = total_cost[best_idx].item()
        ppo_cost = total_cost[0].item()  # candidate 0 = PPO action

        x_star_all[t_idx] = x_star
        cost_all[t_idx] = best_cost
        improvement_all[t_idx] = ppo_cost - best_cost  # positive = x_star is better

    elapsed = time.time() - t0
    print(f"\n[gen-targets] Done in {elapsed:.1f}s ({T_total/elapsed:.1f} steps/s)")

    # ---- Stats ----
    improved = (improvement_all > 0).sum()
    displacement = np.sqrt(np.mean((x_star_all[:T_total] - actions_all[:T_total]) ** 2))
    print(f"[gen-targets] {improved}/{T_total} timesteps improved ({improved/T_total*100:.1f}%)")
    print(f"[gen-targets] Mean improvement: {improvement_all.mean():.4f}")
    print(f"[gen-targets] Mean |x_star - x0|: {displacement:.4f}")
    print(f"[gen-targets] Mean cost (PPO): {cost_all.mean() + improvement_all.mean():.4f}")
    print(f"[gen-targets] Mean cost (x_star): {cost_all.mean():.4f}")

    # ---- Save ----
    np.savez(
        args_cli.output,
        obs=obs_all[:T_total],
        x0=actions_all[:T_total],
        x_star=x_star_all[:T_total],
        cost_x_star=cost_all[:T_total],
        cost_improvement=improvement_all[:T_total],
        ep_lengths=ep_lengths,
        # Metadata
        K=K, H=H, sigma=sigma, cmd_vx=args_cli.cmd_vx,
        w_tracking=args_cli.w_tracking, w_jerk=args_cli.w_jerk,
        w_energy=args_cli.w_energy, w_stability=args_cli.w_stability,
    )
    print(f"[gen-targets] Saved -> {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
