"""Generate dark-themed comparison plots for rough terrain head-to-head evaluation."""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Dark theme colors (matching presentation)
BG      = "#0F172A"
CARD    = "#1E293B"
TEXT    = "#F8FAFC"
MUTED   = "#94A3B8"
RED     = "#EF4444"
GREEN   = "#22C55E"
ACCENT  = "#0EA5E9"

BASE = r"A:\AllIsaac\IsaacLab"
PLOT_DIR = os.path.join(BASE, "rough_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#334155", "grid.alpha": 0.5,
    "font.family": "sans-serif", "font.size": 13,
})

def load_data():
    raw = np.load(os.path.join(BASE, "rough_eval_raw.npz"))
    smo = np.load(os.path.join(BASE, "rough_eval_smo.npz"))
    with open(os.path.join(BASE, "rough_eval_results.json")) as f:
        results = json.load(f)
    return raw, smo, results

def plot_action_rate_comparison(raw, smo, results):
    """Bar chart comparing action smoothness metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["action_rate_rms", "action_magnitude"]
    labels = ["Action Rate\n(RMS)", "Action Magnitude\n(Mean |a|)"]
    raw_vals = [results["raw"].get(m, 0) for m in metrics]
    smo_vals = [results["smoothed"].get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, raw_vals, w, label="Raw PPO", color=RED, alpha=0.85, edgecolor="none")
    bars2 = ax.bar(x + w/2, smo_vals, w, label="Flow-Smoothed", color=GREEN, alpha=0.85, edgecolor="none")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Rough Terrain: Action Smoothness Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, color=TEXT)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, color=TEXT)
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "rough_smoothness_bars.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

def plot_velocity_tracking(raw, smo):
    """Time series of forward velocity for raw vs smoothed."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    N = min(500, len(raw["vx"]), len(smo["vx"]))
    t = np.arange(N) * 0.02  # dt=0.02s
    
    ax.plot(t, raw["vx"][:N], color=RED, alpha=0.7, linewidth=1.0, label="Raw PPO")
    ax.plot(t, smo["vx"][:N], color=GREEN, alpha=0.7, linewidth=1.0, label="Flow-Smoothed")
    ax.axhline(y=1.0, color=ACCENT, linestyle="--", alpha=0.5, label="Command (1.0 m/s)")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Forward Velocity (m/s)")
    ax.set_title("Rough Terrain: Velocity Tracking", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linewidth=0.5)
    ax.set_xlim(0, t[-1])
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "rough_velocity_tracking.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

def plot_action_traces(raw, smo):
    """Overlay action traces for a single joint (FL_hip) to show smoothing."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    N = min(500, len(raw["actions"]), len(smo["actions"]))
    t = np.arange(N) * 0.02
    joint_idx = 0  # FL_hip
    
    axes[0].plot(t, raw["actions"][:N, joint_idx], color=RED, linewidth=0.8)
    axes[0].set_ylabel("Action (FL_hip)")
    axes[0].set_title("Raw PPO Actions on Rough Terrain", fontsize=13, fontweight="bold")
    axes[0].grid(True, linewidth=0.5)
    
    axes[1].plot(t, smo["actions"][:N, joint_idx], color=GREEN, linewidth=0.8)
    axes[1].set_ylabel("Action (FL_hip)")
    axes[1].set_title("Flow-Smoothed Actions on Rough Terrain", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, linewidth=0.5)
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "rough_action_traces.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

def plot_summary_table(results):
    """Create a table image summarizing all metrics."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    
    metrics = [
        ("Action Rate RMS", "action_rate_rms", "lower"),
        ("Action Magnitude", "action_magnitude", "~same"),
        ("Mean Fwd Velocity", "mean_vx", "~1.0"),
        ("Lateral Vel |vy|", "mean_abs_vy", "lower"),
        ("Vertical Vel |vz|", "mean_abs_vz", "lower"),
    ]
    
    cell_text = []
    for label, key, better in metrics:
        rv = results["raw"].get(key, 0)
        sv = results["smoothed"].get(key, 0)
        cell_text.append([label, f"{rv:.4f}", f"{sv:.4f}", better])
    
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Raw PPO", "Flow-Smoothed", "Better"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    # Style header
    for j in range(4):
        table[0, j].set_facecolor(ACCENT)
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Style data rows
    for i in range(1, len(cell_text) + 1):
        for j in range(4):
            table[i, j].set_facecolor(CARD)
            table[i, j].set_text_props(color=TEXT)
    
    ax.set_title("Rough Terrain: Head-to-Head Comparison", fontsize=15,
                 fontweight="bold", pad=20, color=TEXT)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "rough_summary_table.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

if __name__ == "__main__":
    print("Loading rough terrain evaluation data...")
    raw, smo, results = load_data()
    
    print("\nRaw results:", json.dumps({k: v for k, v in results["raw"].items() 
          if not isinstance(v, list)}, indent=2))
    print("\nSmoothed results:", json.dumps({k: v for k, v in results["smoothed"].items()
          if not isinstance(v, list)}, indent=2))
    
    print("\nGenerating plots...")
    plot_action_rate_comparison(raw, smo, results)
    plot_velocity_tracking(raw, smo)
    plot_action_traces(raw, smo)
    plot_summary_table(results)
    print("\nAll plots saved to", PLOT_DIR)
