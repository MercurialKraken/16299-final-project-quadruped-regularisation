"""Stage existing data into the ablation directory layout.

Variant 2 (somereg) data already exists: rollout_data.npz + push_raw.npz.
Variant 6 (somereg_lp) is computed by Butterworth-filtering the somereg rollout.

Output filenames follow the convention expected by analyze_ablation.py.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from scipy.signal import butter, filtfilt


def copy(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        print(f"  skip (missing): {src}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
    print(f"  staged: {os.path.basename(src)} -> {os.path.basename(dst)}")
    return True


def lp_filter(actions: np.ndarray, fc: float = 15.0, fs: float = 50.0,
              order: int = 2) -> np.ndarray:
    """2nd-order Butterworth, zero-phase (filtfilt). Matches Iter1 LP recipe."""
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, actions, axis=0).astype(np.float32)


def synth_lp_rollout(in_npz: str, out_npz: str, fc: float = 15.0):
    """Apply Butterworth LP to actions from an existing rollout, save a sibling."""
    d = dict(np.load(in_npz))
    d["actions_orig"] = d["actions"]
    d["actions"] = lp_filter(d["actions"], fc=fc)
    np.savez(out_npz, **d)
    print(f"  synthesized LP rollout: {out_npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaaclab_dir", default=r"A:\AllIsaac\IsaacLab")
    ap.add_argument("--out_dir", default=r"A:\AllIsaac\flow_matching_project\data\ablation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- variant 2: somereg = existing trained PPO -------------------------
    print("[variant somereg]")
    src_rollout = os.path.join(args.isaaclab_dir, "rollout_data.npz")
    dst_rollout = os.path.join(args.out_dir, "rollout_somereg.npz")
    copy(src_rollout, dst_rollout)

    src_push = os.path.join(args.isaaclab_dir, "push_raw.npz")
    dst_push = os.path.join(args.out_dir, "push_somereg.npz")
    copy(src_push, dst_push)

    # write meta from the existing config
    meta = {
        "variant_name": "somereg",
        "task": "Isaac-Velocity-Flat-Unitree-Go1-v0",
        "action_rate_l2_weight": -0.01,
        "max_iterations": 300,
        "num_envs": 4096,
        "seed": 42,
        "source": "existing_checkpoint_2026-04-06_12-42-26",
    }
    with open(os.path.join(args.out_dir, "meta_somereg.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote meta_somereg.json")

    # --- variant 6: somereg + offline Butterworth LP -----------------------
    print("[variant somereg_lp]")
    if os.path.exists(dst_rollout):
        synth_lp_rollout(dst_rollout, os.path.join(args.out_dir, "rollout_somereg_lp.npz"))
        # for fall rate we need a sim push-recovery run with LP applied online.
        # offline LP on a recorded trajectory cannot produce a fall outcome, so
        # we leave push_somereg_lp.npz unset; analyze_ablation.py will note it.
        meta_lp = dict(meta)
        meta_lp["variant_name"] = "somereg_lp"
        meta_lp["lp_cutoff_hz"] = 15.0
        meta_lp["lp_order"] = 2
        with open(os.path.join(args.out_dir, "meta_somereg_lp.json"), "w") as f:
            json.dump(meta_lp, f, indent=2)
        print("  wrote meta_somereg_lp.json")


if __name__ == "__main__":
    main()
