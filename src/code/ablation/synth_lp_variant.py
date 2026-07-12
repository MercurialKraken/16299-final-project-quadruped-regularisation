"""Apply offline Butterworth LP to a rollout npz, save as a new variant."""
import argparse
import os

import numpy as np
from scipy.signal import butter, filtfilt


def lp(actions, fc=15.0, fs=50.0, order=2):
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, actions, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--fc", type=float, default=15.0)
    args = ap.parse_args()

    d = dict(np.load(args.in_npz))
    raw = d["actions"]
    smoothed = lp(raw, fc=args.fc)
    d["actions_orig"] = raw
    d["actions"] = smoothed
    d["lp_cutoff_hz"] = args.fc
    d["lp_order"] = 2
    np.savez(args.out_npz, **d)
    print(f"[lp] {args.in_npz} -> {args.out_npz}  fc={args.fc}Hz  shape={smoothed.shape}")


if __name__ == "__main__":
    main()
