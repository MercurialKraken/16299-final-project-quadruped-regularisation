"""Regenerate A-to-B race comparison plots with a dark theme to match the deck."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dark deck palette
BG = "#0F172A"          # slate-900 (slide background)
CARD = "#1E293B"        # slate-800 (axes background)
TEXT = "#F8FAFC"        # slate-50
MUTED = "#94A3B8"       # slate-400
RAW = "#EF4444"         # red-500
SMOOTH = "#22C55E"      # green-500
ACCENT = "#0EA5E9"      # sky-500
ACCENT2 = "#38BDF8"     # sky-400

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.grid": True,
    "grid.color": "#334155",
    "grid.alpha": 0.4,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "legend.facecolor": CARD,
    "legend.edgecolor": MUTED,
    "legend.labelcolor": TEXT,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

RAW_NPZ = r"A:\IsaacLab\ab_race_raw.npz"
SMOOTH_NPZ = r"A:\IsaacLab\ab_race_smoothed.npz"
OUT = r"A:\IsaacLab\ab_race_plots_dark"
os.makedirs(OUT, exist_ok=True)

raw = np.load(RAW_NPZ)
sm = np.load(SMOOTH_NPZ)

# Guard: take env 0
def e0(a):
    return a[:, 0] if a.ndim == 2 else a[:, 0, :] if a.ndim == 3 else a

dt = 0.02
raw_pos = raw["positions"]       # (T, 2): x, y
sm_pos  = sm["positions"]
raw_vel = raw["velocities"]      # (T, 2): vx, vy
sm_vel  = sm["velocities"]
raw_aexec = raw["actions_exec"]  # (T, 12)
sm_aexec  = sm["actions_exec"]

T = min(len(raw_pos), len(sm_pos))
t = np.arange(T) * dt

# --- Plot 1: Forward progress vs time ---
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
ax.plot(t, raw_pos[:T, 0] - raw_pos[0, 0], color=RAW, lw=2.2, label="Raw PPO")
ax.plot(t, sm_pos[:T, 0]  - sm_pos[0, 0],  color=SMOOTH, lw=2.2, label="Flow-smoothed")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Forward progress  x  (m)")
ax.set_title("A-to-B Race: Forward Progress", color=TEXT, fontweight="bold")
ax.legend(loc="lower right")
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "progress_vs_time.png"), facecolor=BG, dpi=130)
plt.close(fig)

# --- Plot 2: XY path ---
fig, ax = plt.subplots(figsize=(6, 5.2), dpi=130)
ax.plot(raw_pos[:T, 0], raw_pos[:T, 1], color=RAW, lw=2.2, label="Raw")
ax.plot(sm_pos[:T, 0],  sm_pos[:T, 1],  color=SMOOTH, lw=2.2, label="Smoothed")
ax.scatter([raw_pos[0, 0]], [raw_pos[0, 1]], color=ACCENT, s=60, zorder=5, label="Start")
ax.scatter([raw_pos[T-1, 0]], [raw_pos[T-1, 1]], color=RAW, s=90, marker="X", zorder=5, label="Raw end")
ax.scatter([sm_pos[T-1, 0]],  [sm_pos[T-1, 1]],  color=SMOOTH, s=90, marker="X", zorder=5, label="Smoothed end")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Top-down XY Path", color=TEXT, fontweight="bold")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(loc="upper left", fontsize=9)
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "path_xy.png"), facecolor=BG, dpi=130)
plt.close(fig)

# --- Plot 3: Forward velocity over time ---
fig, ax = plt.subplots(figsize=(6, 5.2), dpi=130)
ax.plot(t, raw_vel[:T, 0], color=RAW, lw=1.2, alpha=0.85, label="Raw v_x")
ax.plot(t, sm_vel[:T, 0],  color=SMOOTH, lw=1.2, alpha=0.85, label="Smoothed v_x")
ax.axhline(1.0, color=ACCENT2, ls="--", lw=1, alpha=0.7, label="cmd 1.0 m/s")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Forward velocity (m/s)")
ax.set_title("Forward Velocity Tracking", color=TEXT, fontweight="bold")
ax.legend(loc="lower right")
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "vx_vs_time.png"), facecolor=BG, dpi=130)
plt.close(fig)

# --- Plot 4: Action rate of change (|a_t - a_{t-1}|.mean over joints) ---
ra = np.linalg.norm(np.diff(raw_aexec[:T], axis=0), axis=1)
sa = np.linalg.norm(np.diff(sm_aexec[:T], axis=0),  axis=1)
fig, ax = plt.subplots(figsize=(6, 5.2), dpi=130)
ax.plot(t[1:], ra, color=RAW, lw=1.0, alpha=0.8, label=f"Raw  mean={ra.mean():.3f}")
ax.plot(t[1:], sa, color=SMOOTH, lw=1.0, alpha=0.8, label=f"Smoothed  mean={sa.mean():.3f}")
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\|a_t - a_{t-1}\|_2$")
ax.set_title("Per-step Action Change (Jitter Proxy)", color=TEXT, fontweight="bold")
ax.legend(loc="upper right")
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "action_rate.png"), facecolor=BG, dpi=130)
plt.close(fig)

# --- Plot 5: summary bars ---
metrics = ["final_x (m)", "mean v_x", "peak v_x", "path length (m)"]
raw_vals = [raw_pos[T-1, 0] - raw_pos[0, 0], raw_vel[:T, 0].mean(), raw_vel[:T, 0].max(),
            np.sum(np.linalg.norm(np.diff(raw_pos[:T, :2], axis=0), axis=1))]
sm_vals  = [sm_pos[T-1, 0] - sm_pos[0, 0],  sm_vel[:T, 0].mean(),  sm_vel[:T, 0].max(),
            np.sum(np.linalg.norm(np.diff(sm_pos[:T, :2], axis=0), axis=1))]

x = np.arange(len(metrics))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
ax.bar(x - w/2, raw_vals, w, color=RAW, label="Raw")
ax.bar(x + w/2, sm_vals,  w, color=SMOOTH, label="Smoothed")
for i, (r, s) in enumerate(zip(raw_vals, sm_vals)):
    ax.text(i - w/2, r, f"{r:.2f}", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.text(i + w/2, s, f"{s:.2f}", ha="center", va="bottom", color=TEXT, fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("Summary Metrics", color=TEXT, fontweight="bold")
ax.legend()
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "summary_bars.png"), facecolor=BG, dpi=130)
plt.close(fig)

print("RAW  final_x:", raw_vals[0], "mean_vx:", raw_vals[1], "peak:", raw_vals[2], "path:", raw_vals[3])
print("SMTH final_x:", sm_vals[0],  "mean_vx:", sm_vals[1],  "peak:", sm_vals[2],  "path:", sm_vals[3])
print("Saved to", OUT)
