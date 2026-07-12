"""Build a 4-panel dark-themed comparison plot for the 7-way ablation."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ABL = r"A:\AllIsaac\flow_matching_project\data\ablation"
OUT = os.path.join(ABL, "ablation_comparison.png")


VARIANTS = ["noreg", "somereg", "extremereg",
            "noreg_flow", "somereg_flow", "extremereg_flow",
            "noreg_lp", "somereg_lp", "noreg_flow_lp"]
LABELS = {
    "noreg": "no-reg\nraw",
    "somereg": "some-reg\nraw",
    "extremereg": "extreme-reg\nraw",
    "noreg_flow": "no-reg\n+Flow",
    "somereg_flow": "some-reg\n+Flow",
    "extremereg_flow": "extreme-reg\n+Flow",
    "noreg_lp": "no-reg\n+LP",
    "somereg_lp": "some-reg\n+LP",
    "noreg_flow_lp": "no-reg\n+Flow+LP",
}
COLOR = {
    "noreg":           "#ff7e6b",
    "somereg":         "#fbb13c",
    "extremereg":      "#9b6dff",
    "noreg_flow":      "#3ec7e7",
    "somereg_flow":    "#1e9bd6",
    "extremereg_flow": "#7d4dd6",
    "noreg_lp":        "#5be58c",
    "somereg_lp":      "#36c69b",
    "noreg_flow_lp":   "#56e1c2",
}


def main():
    with open(os.path.join(ABL, "results.json")) as f:
        R = json.load(f)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor="#101424")
    fig.suptitle("PPO Reward Reg × Smoother Ablation — 9 variants  (Go1, vx=1.0, 80 envs, push 50–500N) — fall rate: lower is better",
                 fontsize=14, color="white")

    metrics = [
        ("hf_energy_pct", "HF Spectral Energy >10 Hz (% of total)", axes[0, 0]),
        ("fall_rate", "Fall Rate (lower = more push-robust)", axes[0, 1]),
        ("jerk_rms", "Action Jerk RMS", axes[1, 0]),
        ("infer_ms_mean", "Per-step Inference Latency (ms)", axes[1, 1]),
    ]

    for key, title, ax in metrics:
        x = np.arange(len(VARIANTS))
        vals = [R[v].get(key) for v in VARIANTS]
        valid = [(i, v, val) for i, (v, val) in enumerate(zip(VARIANTS, vals))
                 if val is not None]
        if not valid:
            continue
        idx = [v[0] for v in valid]
        ys = [v[2] for v in valid]
        cols = [COLOR[VARIANTS[i]] for i in idx]
        bars = ax.bar([x[i] for i in idx], ys, color=cols, edgecolor="#2a2f44", linewidth=1)

        # value labels
        ymax = max(ys) if ys else 1
        for i, b in zip(idx, bars):
            v = ys[idx.index(i)]
            label = (f"{v*100:.1f}%" if key == "fall_rate"
                     else f"{v:.2f}%" if key == "hf_energy_pct"
                     else f"{v:.3f}")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.02,
                    label, ha="center", color="white", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=9, color="#cfd2dc")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_facecolor("#161a2c")
        ax.grid(axis="y", alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color("#2a2f44")
        if key == "infer_ms_mean":
            ax.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT, dpi=140, facecolor="#101424")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
