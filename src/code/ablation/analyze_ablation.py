"""Unified analysis pipeline for the 7-variant ablation study.

Inputs (all optional; missing files are skipped gracefully):
  rollout_<variant>.npz       - obs/actions/joint_pos/joint_vel/joint_acc/base_vel
  push_<variant>.npz          - fell, fall_step, magnitudes (from push_recovery_eval.py)
  timing_<variant>.json       - {variant, iters:[{iter,wall_s}], wall_total_s}
  inference_<variant>.json    - {flow_model, mean_ms, std_ms, n_steps}

Outputs:
  results.json   - dict[variant -> {hf_energy_pct, jerk_rms, action_rate_rms,
                                    mean_vx, mean_power_w, fall_rate,
                                    train_total_s, train_iters, infer_ms_mean,
                                    action_rate_l2_weight}]
  results.csv    - same data, flat
  comparison.png - dark-themed multi-panel: HF energy, fall rate, jerk, power
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# Sampling rate of the policy (Hz). decimation=4 with sim dt=0.005 -> 50 Hz.
FS = 50.0
# Frequency cutoff for "high-frequency" spectral energy.
HF_CUTOFF = 10.0


@dataclass
class VariantResult:
    variant: str
    action_rate_l2_weight: Optional[float] = None
    # smoothness / spectral
    hf_energy_pct: Optional[float] = None     # power above 10 Hz / total power, %
    action_rate_rms: Optional[float] = None   # RMS of |a_t - a_{t-1}|
    jerk_rms: Optional[float] = None          # RMS of |d^2 a / dt^2|
    # task / energy
    mean_vx: Optional[float] = None
    mean_power_w: Optional[float] = None
    # robustness
    fall_rate: Optional[float] = None
    push_n_envs: Optional[int] = None
    # cost
    train_total_s: Optional[float] = None
    train_iters: Optional[int] = None
    infer_ms_mean: Optional[float] = None
    infer_ms_std: Optional[float] = None
    notes: list = field(default_factory=list)


# -------------------------- metric computations -----------------------------

def hf_spectral_energy_pct(actions: np.ndarray, fs: float = FS, cutoff: float = HF_CUTOFF) -> float:
    """Average over joints of (HF power / total power), as a percentage."""
    actions = np.asarray(actions, dtype=np.float64)
    T, J = actions.shape
    # remove DC offset per joint so the bin at 0 doesn't dominate
    a = actions - actions.mean(axis=0, keepdims=True)
    F = np.fft.rfft(a, axis=0)
    psd = (F.real ** 2 + F.imag ** 2)  # one-sided power
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    hf = (freqs > cutoff)
    total_p = psd.sum(axis=0)
    hf_p = psd[hf].sum(axis=0)
    # protect against zero
    pct = np.where(total_p > 0, 100.0 * hf_p / total_p, 0.0)
    return float(pct.mean())


def action_rate_rms(actions: np.ndarray) -> float:
    da = np.diff(actions, axis=0)
    return float(np.sqrt((da ** 2).mean()))


def jerk_rms(actions: np.ndarray) -> float:
    """RMS of the discrete second difference, |d^2 a|."""
    if actions.shape[0] < 3:
        return float("nan")
    d2 = actions[2:] - 2 * actions[1:-1] + actions[:-2]
    return float(np.sqrt((d2 ** 2).mean()))


def mean_forward_velocity(base_vel: np.ndarray) -> float:
    return float(np.asarray(base_vel)[:, 0].mean())


def mean_power_watts(joint_vel: np.ndarray, joint_acc: np.ndarray, mass_proxy: float = 1.0) -> float:
    """Approximate mechanical power as |joint_vel * joint_acc| summed over joints.

    This is a proxy (no torque available in rollout npz). Useful for *relative*
    comparison between variants since the actuator network is shared.
    """
    p = np.abs(np.asarray(joint_vel) * np.asarray(joint_acc) * mass_proxy)
    return float(p.sum(axis=1).mean())


def fall_rate_from_push(npz_path: str) -> tuple[float, int]:
    d = np.load(npz_path)
    fell = np.asarray(d["fell"]).astype(bool)
    return float(fell.mean()), int(fell.size)


# -------------------------- driver -----------------------------------------

def find_input(d: str, pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(d, pattern)))
    return matches[0] if matches else None


def analyze_variant(variant: str, in_dir: str) -> VariantResult:
    r = VariantResult(variant=variant)

    rollout = find_input(in_dir, f"rollout_{variant}.npz")
    if rollout is None and variant in ("somereg", "some_reg"):
        # fall back to the historical filename
        rollout = find_input(in_dir, "rollout_data.npz")
    if rollout:
        d = np.load(rollout)
        actions = d["actions"]
        r.hf_energy_pct = hf_spectral_energy_pct(actions)
        r.action_rate_rms = action_rate_rms(actions)
        r.jerk_rms = jerk_rms(actions)
        if "base_vel" in d:
            r.mean_vx = mean_forward_velocity(d["base_vel"])
        if "joint_vel" in d and "joint_acc" in d:
            r.mean_power_w = mean_power_watts(d["joint_vel"], d["joint_acc"])
    else:
        r.notes.append("no rollout npz")

    push = find_input(in_dir, f"push_{variant}.npz")
    if push is None and variant in ("somereg", "some_reg"):
        push = find_input(in_dir, "push_raw.npz")
    if push:
        rate, n = fall_rate_from_push(push)
        r.fall_rate = rate
        r.push_n_envs = n
    else:
        r.notes.append("no push npz")

    timing = find_input(in_dir, f"timing_{variant}.json")
    if timing:
        with open(timing) as f:
            t = json.load(f)
        r.train_total_s = t.get("wall_total_s")
        r.train_iters = len(t.get("iters", []))

    infer = find_input(in_dir, f"inference_{variant}.json")
    if infer:
        with open(infer) as f:
            i = json.load(f)
        r.infer_ms_mean = i.get("mean_ms")
        r.infer_ms_std = i.get("std_ms")

    meta = find_input(in_dir, f"meta_{variant}.json")
    if meta:
        with open(meta) as f:
            m = json.load(f)
        r.action_rate_l2_weight = m.get("action_rate_l2_weight")

    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=r"A:\AllIsaac\flow_matching_project\data\ablation",
                    help="Directory containing the rollout/push/timing files.")
    ap.add_argument("--out_dir", default=r"A:\AllIsaac\flow_matching_project\data\ablation",
                    help="Directory for results.json/results.csv/plots.")
    ap.add_argument("--variants", nargs="+",
                    default=["noreg", "somereg", "extremereg",
                             "noreg_flow", "noreg_lp", "somereg_lp",
                             "noreg_flow_lp", "somereg_flow",
                             "extremereg_flow"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for v in args.variants:
        r = analyze_variant(v, args.in_dir)
        results[v] = asdict(r)
        present = [k for k, val in results[v].items()
                   if val not in (None, [], "") and k not in ("variant", "notes")]
        print(f"[{v}] {len(present)} fields populated, notes={r.notes}")

    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_json}")

    # CSV in a stable column order
    cols = ["variant", "action_rate_l2_weight", "hf_energy_pct", "action_rate_rms",
            "jerk_rms", "mean_vx", "mean_power_w", "fall_rate", "push_n_envs",
            "train_total_s", "train_iters", "infer_ms_mean"]
    out_csv = os.path.join(args.out_dir, "results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for v in args.variants:
            row = results[v]
            w.writerow([row.get(c, "") for c in cols])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
