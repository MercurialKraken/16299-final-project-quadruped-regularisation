"""LP-filter the x_star (optimal) targets per-episode for the Bal-LP recipe."""
import argparse

import numpy as np
from scipy.signal import butter, filtfilt


def lp(actions, fc=15.0, fs=50.0, order=2):
    b, a = butter(order, fc / (fs / 2), btype="low")
    out = np.zeros_like(actions)
    for j in range(actions.shape[1]):
        out[:, j] = filtfilt(b, a, actions[:, j])
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--fc", type=float, default=15.0)
    args = ap.parse_args()

    d = dict(np.load(args.in_npz))
    print(f"keys: {list(d.keys())}")
    x_star = d["x_star"].astype(np.float32)
    x0 = d["x0"].astype(np.float32)
    ep_lengths = d["ep_lengths"]

    smoothed = np.zeros_like(x_star)
    idx = 0
    for L in ep_lengths:
        ep = x_star[idx:idx + L]
        if L > 12:
            smoothed[idx:idx + L] = lp(ep, fc=args.fc)
        else:
            smoothed[idx:idx + L] = ep
        idx += L

    before = np.sqrt(np.mean((x_star - x0) ** 2))
    after = np.sqrt(np.mean((smoothed - x0) ** 2))
    delta = np.sqrt(np.mean((smoothed - x_star) ** 2))
    print(f"|x_star - x0| (before LP): {before:.4f}")
    print(f"|x_smooth - x0| (after LP): {after:.4f}")
    print(f"|smoothing delta|: {delta:.4f}")

    d["x_star_orig"] = x_star
    d["x_star"] = smoothed
    d["lp_cutoff_hz"] = args.fc
    np.savez(args.out_npz, **d)
    print(f"saved -> {args.out_npz}")


if __name__ == "__main__":
    main()
