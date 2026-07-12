"""Generate comparison plots for the presentation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load data
d = np.load(r"A:\IsaacLab\adaptive_comparison.npz", allow_pickle=True)
raw = d["raw"]              # (5000, 12)
ada_03 = d["adaptive_t0.3"]
ada_10 = d["adaptive_t1.0"]
uni_03 = d["uniform_t0.3"]
uni_10 = d["uniform_t1.0"]
ep_lengths = d["ep_lengths"]

JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Style
plt.rcParams.update({
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#475569",
    "text.color": "#F8FAFC",
    "axes.labelcolor": "#F8FAFC",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
})

SKY = "#0EA5E9"
SKY2 = "#38BDF8"
GREEN = "#22C55E"
RED = "#EF4444"
AMBER = "#F59E0B"
WHITE = "#F8FAFC"
MUTED = "#94A3B8"

# ── PLOT 1: Per-joint improvement bar chart (adaptive vs uniform) ─────────
fig1, ax1 = plt.subplots(figsize=(12, 5))

raw_rates = np.array([np.sqrt(np.mean(np.diff(raw[:, j])**2)) for j in range(12)])
ada_rates = np.array([np.sqrt(np.mean(np.diff(ada_03[:, j])**2)) for j in range(12)])
uni_rates = np.array([np.sqrt(np.mean(np.diff(uni_03[:, j])**2)) for j in range(12)])

ada_pct = (1 - ada_rates / raw_rates) * 100
uni_pct = (1 - uni_rates / raw_rates) * 100

x = np.arange(12)
w = 0.35
bars1 = ax1.bar(x - w/2, ada_pct, w, label="Adaptive Cutoff", color=SKY, edgecolor="none")
bars2 = ax1.bar(x + w/2, uni_pct, w, label="Uniform 15 Hz", color=MUTED, edgecolor="none", alpha=0.7)

ax1.set_xticks(x)
ax1.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Action Rate Reduction (%)", fontsize=12)
ax1.set_title("Per-Joint Smoothing: Adaptive vs Uniform Cutoff (t=0.3)", fontsize=14, fontweight="bold", pad=12)
ax1.legend(loc="upper right", framealpha=0.8, facecolor="#1E293B", edgecolor="#475569")
ax1.axhline(y=0, color="#475569", linewidth=0.8)
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(-2, 18)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    if h > 0.5:
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7, color=SKY)

fig1.tight_layout()
fig1.savefig(r"A:\IsaacLab\plot_perjoint_comparison.png", dpi=200, bbox_inches="tight")
print("Saved plot_perjoint_comparison.png")

# ── PLOT 2: Action traces for 3 representative joints ────────────────────
# Show raw vs adaptive-smoothed for FR_calf (best), RR_thigh (worst), FL_hip (moderate)
showcase = [("FR_calf", 5), ("FL_hip", 0), ("RR_thigh", 10)]
ep_start = 0
ep_len = int(ep_lengths[0])
t_axis = np.arange(200) * 0.02  # 200 steps = 4 seconds

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for i, (name, j) in enumerate(showcase):
    ax = axes2[i]
    sl = slice(ep_start, ep_start + 200)
    ax.plot(t_axis, raw[sl, j], color=RED, alpha=0.7, linewidth=1.0, label="Raw RL")
    ax.plot(t_axis, ada_10[sl, j], color=SKY, alpha=0.9, linewidth=1.5, label="Adaptive t=1.0")
    ax.plot(t_axis, uni_10[sl, j], color=MUTED, alpha=0.6, linewidth=1.0, linestyle="--", label="Uniform t=1.0")
    ax.set_ylabel(f"{name}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", framealpha=0.8, facecolor="#1E293B", edgecolor="#475569", fontsize=9)

axes2[0].set_title("Action Traces: Raw vs Flow-Smoothed (4 seconds)", fontsize=14, fontweight="bold", pad=12)
axes2[-1].set_xlabel("Time (s)", fontsize=12)
fig2.tight_layout()
fig2.savefig(r"A:\IsaacLab\plot_action_traces.png", dpi=200, bbox_inches="tight")
print("Saved plot_action_traces.png")

# ── PLOT 3: Cutoff frequency map (visual of per-joint tuning) ────────────
CUTOFFS = [10, 12, 22, 8, 15, 8, 22, 10, 15, 12, 25, 10]

fig3, ax3 = plt.subplots(figsize=(12, 4))
colors = [GREEN if c <= 10 else (SKY if c <= 15 else AMBER) for c in CUTOFFS]
bars3 = ax3.bar(range(12), CUTOFFS, color=colors, edgecolor="none", width=0.7)
ax3.axhline(y=15, color=WHITE, linewidth=1, linestyle="--", alpha=0.4, label="Uniform baseline (15 Hz)")
ax3.axhline(y=5.6, color=RED, linewidth=1, linestyle=":", alpha=0.6, label="Gait frequency (5.6 Hz)")
ax3.set_xticks(range(12))
ax3.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=9)
ax3.set_ylabel("Cutoff Frequency (Hz)", fontsize=12)
ax3.set_title("Per-Joint Adaptive Cutoff Frequencies", fontsize=14, fontweight="bold", pad=12)
ax3.legend(loc="upper right", framealpha=0.8, facecolor="#1E293B", edgecolor="#475569", fontsize=9)
ax3.set_ylim(0, 30)
ax3.grid(axis="y", alpha=0.3)

# Add value labels
for bar, val in zip(bars3, CUTOFFS):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val}", 
             ha="center", va="bottom", fontsize=9, color=WHITE, fontweight="bold")

fig3.tight_layout()
fig3.savefig(r"A:\IsaacLab\plot_cutoff_map.png", dpi=200, bbox_inches="tight")
print("Saved plot_cutoff_map.png")

# ── PLOT 4: Overall smoothing progression (5Hz -> 15Hz -> adaptive) ──────
fig4, ax4 = plt.subplots(figsize=(10, 5))

# The story: 5Hz destroyed gait, 15Hz gave 3-6%, adaptive gives 6-16%
methods = ["5 Hz Cutoff\n(FAILED)", "15 Hz Uniform\nt=0.3", "15 Hz Uniform\nt=1.0", "Adaptive\nt=0.3", "Adaptive\nt=1.0"]
reductions = [0, 3.8, 10.1, 5.8, 15.6]  # action rate reductions
bar_colors = [RED, MUTED, MUTED, SKY, SKY]
bar_edge = ["none"] * 5

bars4 = ax4.bar(range(5), reductions, color=bar_colors, edgecolor=bar_edge, width=0.6)
# X on the failed bar
ax4.text(0, 1, "X", ha="center", va="bottom", fontsize=28, color=RED, fontweight="bold")
ax4.annotate("Destroyed\ngait signal", xy=(0, 0), xytext=(0.7, 8), fontsize=9, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), ha="center")

for i, (bar, val) in enumerate(zip(bars4, reductions)):
    if i > 0:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=11, color=WHITE, fontweight="bold")

ax4.set_xticks(range(5))
ax4.set_xticklabels(methods, fontsize=10)
ax4.set_ylabel("Action Rate Reduction (%)", fontsize=12)
ax4.set_title("Smoothing Journey: From Failure to Adaptive Success", fontsize=14, fontweight="bold", pad=12)
ax4.set_ylim(0, 20)
ax4.grid(axis="y", alpha=0.3)

fig4.tight_layout()
fig4.savefig(r"A:\IsaacLab\plot_journey.png", dpi=200, bbox_inches="tight")
print("Saved plot_journey.png")

# ── PLOT 5: Training loss curve ──────────────────────────────────────────
losses = np.load(r"A:\IsaacLab\adaptive_training_loss.npy")
fig5, ax5 = plt.subplots(figsize=(8, 4))
ax5.plot(range(1, len(losses)+1), losses, color=SKY, linewidth=1.5)
ax5.set_xlabel("Epoch", fontsize=12)
ax5.set_ylabel("MSE Loss", fontsize=12)
ax5.set_title("Adaptive Flow Model Training Convergence", fontsize=14, fontweight="bold", pad=12)
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(r"A:\IsaacLab\plot_training_loss.png", dpi=200, bbox_inches="tight")
print("Saved plot_training_loss.png")

print("\nAll plots generated successfully!")
