"""Train PPO on a Go1 flat-terrain reward-ablation variant.

Mirrors the IsaacLab train.py flow but:
  1. Imports our `register_reg_variants` to expose NoReg / ExtremeReg task IDs.
  2. Adds wall-clock timing instrumentation per RSL-RL iteration into a JSON
     sidecar at <log_dir>/timing.json.
  3. Saves a small ablation_meta.json next to the checkpoint with the variant
     name and the action_rate_l2 weight that was actually used.

Usage (from A:\\AllIsaac\\IsaacLab):
    .\\isaaclab.bat -p A:\\AllIsaac\\flow_matching_project\\scripts\\ablation\\train_ablation.py ^
        --task Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0 ^
        --headless --num_envs 4096 --max_iterations 300 --variant_name noreg
"""
import argparse
import json
import os
import sys
import time

# --- argparse + AppLauncher (mirrors IsaacLab/scripts/.../rsl_rl/train.py) ---
from isaaclab.app import AppLauncher

# locate cli_args.py shipped with IsaacLab so we get rsl_rl-specific flags
sys.path.insert(0, r"A:\AllIsaac\IsaacLab\scripts\reinforcement_learning\rsl_rl")
import cli_args  # type: ignore  # noqa: E402

parser = argparse.ArgumentParser(description="PPO ablation training (IsaacLab+RSL-RL).")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--variant_name", type=str, required=True,
                    help="Short label, e.g. noreg / somereg / extremereg.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--distributed", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Hydra reads from sys.argv[1:]
sys.argv = [sys.argv[0]] + hydra_args

# launch app BEFORE importing anything Omniverse-related
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- now safe to import the rest ---
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401, E402

# *** the only ablation-specific bit: register variant tasks ***
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import register_reg_variants  # noqa: F401, E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    import importlib.metadata as metadata
    installed_version = metadata.version("rsl-rl-lib")

    # CLI overrides
    from cli_args import update_rsl_rl_cfg  # type: ignore
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed

    # log dir = <ISAACLAB>/logs/rsl_rl/<exp>/<timestamp>_<variant>
    exp_name = agent_cfg.experiment_name + f"_{args_cli.variant_name}"
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", exp_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir
    print(f"[ablation] log_dir = {log_dir}")

    # env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # save ablation metadata
    meta = {
        "variant_name": args_cli.variant_name,
        "task": args_cli.task,
        "action_rate_l2_weight": float(env_cfg.rewards.action_rate_l2.weight),
        "max_iterations": int(agent_cfg.max_iterations),
        "num_envs": int(env_cfg.scene.num_envs),
        "seed": int(agent_cfg.seed),
    }
    with open(os.path.join(log_dir, "ablation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- training with timing -------------------------------------------------
    # Call learn() ONCE so RSL-RL's internal save_interval logic works correctly.
    # Per-iter timing is approximated by total / max_iterations.
    total_iters = int(agent_cfg.max_iterations)
    t0 = time.time()
    runner.learn(num_learning_iterations=total_iters, init_at_random_ep_len=True)
    wall_total = time.time() - t0

    # save final checkpoint explicitly in case save_interval doesn't divide N
    final_ckpt = os.path.join(log_dir, f"model_{total_iters - 1}.pt")
    if not os.path.exists(final_ckpt):
        try:
            runner.save(final_ckpt)
            print(f"[ablation] saved final checkpoint: {final_ckpt}")
        except Exception as e:
            print(f"[ablation] WARN: could not save final checkpoint: {e}")

    timings = {
        "variant": args_cli.variant_name,
        "iters_total": total_iters,
        "wall_total_s": wall_total,
        "wall_per_iter_s_avg": wall_total / max(total_iters, 1),
    }
    with open(os.path.join(log_dir, "timing.json"), "w") as f:
        json.dump(timings, f, indent=2)
    print(f"[ablation] total wall time: {wall_total:.1f}s "
          f"({wall_total / max(total_iters,1):.3f}s/iter avg)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
