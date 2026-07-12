"""Generate dark-themed plots for effort/energy + push-recovery results."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette (matches deck)
BG     = "#0F172A"
CARD   = "#1E293B"
TEXT   = "#F8FAFC"
MUTED  = "#94A3B8"
RAW    = "#EF4444"
SMOOTH = "#22C55E"
ACCENT = "#0EA5E9"
ACCENT2 = "#38BDF8"
GRID   = "#334155"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
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

OUT = r"A:\IsaacLab\eff_push_plots"
os.makedirs(OUT, exist_ok=True)


# =========================================================================
# Effort / energy plots
# =========================================================================
raw = np.load(r"A:\IsaacLab\effort_raw.npz")
smo = np.load(r"A:\IsaacLab\effort_smoothed.npz")

dt = float(raw["dt"])
N = int(raw["num_envs"])
actions_r  = raw["actions"]            # (T, N, 12)
actions_s  = smo["actions"]
torque_r   = raw["torque"]             # (T, N, 12)
torque_s   = smo["torque"]
qdot_r     = raw["joint_vel"]          # (T, N, 12)
qdot_s     = smo["joint_vel"]

T = min(actions_r.shape[0], actions_s.shape[0])
t = np.arange(T) * dt

def compute_per_env(actions, torque, qdot):
    a = actions[:T]
    tau = torque[:T]
    qv  = qdot[:T]
    da  = np.diff(a, axis=0) / dt                 # (T-1,N,12)
    d2a = np.diff(a, n=2, axis=0) / (dt**2)       # (T-2,N,12)
    action_rate = np.sqrt(np.mean(da**2, axis=(0, 2)))   # per-env rms
    jerk        = np.sqrt(np.mean(d2a**2, axis=(0, 2)))
    torque_rms  = np.sqrt(np.mean(tau**2, axis=(0, 2)))
    power       = np.abs(tau * qv).sum(axis=2)            # (T,N)
    power_mean  = power.mean(axis=0)
    energy      = power_mean * dt * T
    return dict(action_rate=action_rate, jerk=jerk, torque_rms=torque_rms,
                power_mean=power_mean, energy=energy, power_t=power.mean(axis=1))

mr = compute_per_env(actions_r, torque_r, qdot_r)
ms = compute_per_env(actions_s, torque_s, qdot_s)

# ---- Plot E1: bar chart of normalized deltas ----
metrics = ["action_rate", "jerk", "torque_rms", "power_mean", "energy"]
labels  = ["Action rate", "Jerk", "Torque RMS", "Mean power", "Total energy"]
deltas_pct = [100.0 * (ms[m].mean() - mr[m].mean()) / mr[m].mean() for m in metrics]
wins = [int((ms[m] < mr[m]).sum()) for m in metrics]

fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=130)
colors = [SMOOTH if d < 0 else RAW for d in deltas_pct]
bars = ax.bar(labels, deltas_pct, color=colors, edgecolor=MUTED, linewidth=0.6)
for b, d, w in zip(bars, deltas_pct, wins):
    ypos = d - 0.4 if d < 0 else d + 0.4
    va = "top" if d < 0 else "bottom"
    ax.text(b.get_x() + b.get_width()/2, ypos,
            f"{d:+.2f}%\n({w}/{N} wins)", ha="center", va=va,
            color=TEXT, fontsize=10, fontweight="bold")
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylabel("Smoothed vs Raw  (%)")
ax.set_title("Control-effort metrics: smoothed - raw  (lower = smoother)",
             color=TEXT, fontweight="bold")
ymax = max(abs(min(deltas_pct)), abs(max(deltas_pct))) * 1.6 + 1
ax.set_ylim(-ymax, ymax)
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "effort_bars.png"), facecolor=BG, dpi=130)
plt.close(fig)

# ---- Plot E2: per-env paired scatter for energy + jerk ----
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), dpi=130)
for ax, key, name in zip(axes, ["jerk", "energy"], ["Jerk RMS  (rad/s^2)", "Total energy  (J*s)"]):
    r_v = mr[key]; s_v = ms[key]
    lims = [min(r_v.min(), s_v.min()) * 0.95, max(r_v.max(), s_v.max()) * 1.05]
    ax.plot(lims, lims, color=MUTED, lw=1, ls="--", alpha=0.7)
    ax.scatter(r_v, s_v, c=SMOOTH, edgecolors=TEXT, s=55, lw=0.5, zorder=3)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"Raw {name}"); ax.set_ylabel(f"Smoothed {name}")
    ax.set_title(name.split('(')[0].strip(), color=TEXT, fontweight="bold")
    ax.text(0.04, 0.95,
            f"smoothed wins {(s_v < r_v).sum()}/{N}\nDmean = {(s_v.mean() - r_v.mean()):+.3g}",
            transform=ax.transAxes, va="top", color=TEXT, fontsize=10,
            bbox=dict(facecolor=BG, edgecolor=MUTED, boxstyle="round,pad=0.3"))
    for sp in ax.spines.values(): sp.set_color(MUTED)
fig.suptitle("Paired comparison across 20 seeds: every dot below the diagonal = smoothed wins",
             color=TEXT, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "effort_paired.png"), facecolor=BG, dpi=130)
plt.close(fig)

# ---- Plot E3: power over time (env-mean) ----
fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=130)
ax.plot(t, mr["power_t"], color=RAW, lw=1.4, alpha=0.9, label=f"Raw  mean={mr['power_t'].mean():.2f} W")
ax.plot(t, ms["power_t"], color=SMOOTH, lw=1.4, alpha=0.9, label=f"Smoothed  mean={ms['power_t'].mean():.2f} W")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Mechanical power  sum|tau*qdot|  (W)")
ax.set_title("Instantaneous mechanical power, averaged over 20 envs",
             color=TEXT, fontweight="bold")
ax.legend(loc="upper right")
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "power_vs_time.png"), facecolor=BG, dpi=130)
plt.close(fig)


# =========================================================================
# Push recovery plots
# =========================================================================
pr = np.load(r"A:\IsaacLab\push2_raw.npz")
ps = np.load(r"A:\IsaacLab\push2_smoothed.npz")

dt_p = float(pr["dt"])
mags = np.unique(pr["magnitudes"])
fell_r  = pr["fell"];  fell_s  = ps["fell"]
mag_arr = pr["magnitudes"]

rate_r = np.array([fell_r[mag_arr == m].mean() for m in mags])
rate_s = np.array([fell_s[mag_arr == m].mean() for m in mags])

# ---- Plot P1: fall rate per magnitude ----
fig, ax = plt.subplots(figsize=(9.0, 4.4), dpi=130)
x = np.arange(len(mags))
w = 0.38
ax.bar(x - w/2, 100 * rate_r, w, color=RAW,    label="Raw PPO",      edgecolor=MUTED, lw=0.5)
ax.bar(x + w/2, 100 * rate_s, w, color=SMOOTH, label="Flow-smoothed", edgecolor=MUTED, lw=0.5)
for i, (rr, rs) in enumerate(zip(rate_r, rate_s)):
    if rr > 0:
        ax.text(i - w/2, 100*rr + 2, f"{int(round(rr*100))}%", ha="center", color=TEXT, fontsize=9)
    if rs > 0:
        ax.text(i + w/2, 100*rs + 2, f"{int(round(rs*100))}%", ha="center", color=TEXT, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"{int(m)}" for m in mags])
ax.set_xlabel("Lateral push magnitude  (N)")
ax.set_ylabel("Fall rate  (%)")
ax.set_title("Push recovery: fall rate vs lateral impulse magnitude",
             color=TEXT, fontweight="bold")
ax.set_ylim(0, 110)
ax.legend(loc="upper left")
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "push_fall_rate.png"), facecolor=BG, dpi=130)
plt.close(fig)

# ---- Plot P2: McNemar discordant + survival delta ----
b = int(((fell_r) & (~fell_s)).sum())     # raw fell, smo OK
c = int(((~fell_r) & (fell_s)).sum())     # raw OK,   smo fell
both = int((fell_r & fell_s).sum())
none = int((~fell_r & ~fell_s).sum())

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=130)
# Discordant pair bar
ax = axes[0]
bars = ax.bar(
    ["Raw fell\n+ Smoothed OK", "Smoothed fell\n+ Raw OK"],
    [b, c],
    color=[SMOOTH, RAW], edgecolor=MUTED, lw=0.5,
)
for bar, val in zip(bars, [b, c]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.4, str(val),
            ha="center", color=TEXT, fontsize=12, fontweight="bold")
ax.set_ylabel("# of paired envs (n=80)")
ax.set_title("Discordant pairs  (McNemar p = 0.0146)",
             color=TEXT, fontweight="bold")
ax.set_ylim(0, max(b, c) + 4)
for s in ax.spines.values(): s.set_color(MUTED)

# Time-to-fall delta among "both fell"
ax = axes[1]
both_mask = fell_r & fell_s
fall_step_r = pr["fall_step"][both_mask]
fall_step_s = ps["fall_step"][both_mask]
delta_steps = fall_step_s - fall_step_r
delta_ms = delta_steps * dt_p * 1000
ax.hist(delta_ms, bins=12, color=ACCENT, edgecolor=MUTED, lw=0.7, alpha=0.9)
ax.axvline(0, color=MUTED, ls="--", lw=1)
ax.axvline(delta_ms.mean(), color=SMOOTH, lw=2, label=f"mean = {delta_ms.mean():+.1f} ms")
ax.set_xlabel("Time-to-fall delta  (ms,  smoothed - raw)")
ax.set_ylabel("# of pairs")
ax.set_title(f"Both fell (n={both_mask.sum()}): smoothed lasts longer",
             color=TEXT, fontweight="bold")
ax.legend(loc="upper right")
for s in ax.spines.values(): s.set_color(MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "push_mcnemar.png"), facecolor=BG, dpi=130)
plt.close(fig)

print(f"Saved plots to {OUT}")
print("Effort metrics summary:")
for k, lbl, d, w in zip(metrics, labels, deltas_pct, wins):
    print(f"  {lbl:14s}  delta = {d:+6.3f}%   wins {w}/{N}")
print(f"Push: raw fell {fell_r.sum()}/80   smo fell {fell_s.sum()}/80   "
      f"discordant b={b} c={c}   delta_ms_mean={delta_ms.mean():+.2f}")
