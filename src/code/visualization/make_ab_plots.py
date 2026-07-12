"""Generate A-to-B race comparison plots from NPZ data."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

raw = np.load('A:/IsaacLab/ab_race_raw.npz')
smo = np.load('A:/IsaacLab/ab_race_smoothed.npz')

out_dir = 'A:/IsaacLab/ab_race_plots'
os.makedirs(out_dir, exist_ok=True)

dt_raw = float(raw['dt']); dt_smo = float(smo['dt'])
t_raw = np.arange(len(raw['positions'])) * dt_raw
t_smo = np.arange(len(smo['positions'])) * dt_smo

start_raw = raw['start_pos']
start_smo = smo['start_pos']

x_raw = raw['positions'][:, 0] - start_raw[0]
y_raw = raw['positions'][:, 1] - start_raw[1]
x_smo = smo['positions'][:, 0] - start_smo[0]
y_smo = smo['positions'][:, 1] - start_smo[1]

dist_raw = np.linalg.norm(raw['positions'] - start_raw[None, :], axis=1)
dist_smo = np.linalg.norm(smo['positions'] - start_smo[None, :], axis=1)

vx_raw = raw['velocities'][:, 0]
vx_smo = smo['velocities'][:, 0]

# Plot 1: forward progress (x) over time
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_raw, x_raw, label='Raw (t_end=0.0)', color='#d9534f', lw=2)
ax.plot(t_smo, x_smo, label='Smoothed (t_end=0.3)', color='#337ab7', lw=2)
ax.axhline(y=float(raw['cmd_vx'])*t_raw[-1], color='gray', ls='--',
           label=f'Ideal @ {float(raw["cmd_vx"])} m/s')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Forward progress x (m)')
ax.set_title('A→B Forward Progress Over Time')
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f'{out_dir}/progress_vs_time.png', dpi=150); plt.close()
print(f'[PLOT] progress_vs_time.png')

# Plot 2: XY path
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_raw, y_raw, label='Raw', color='#d9534f', lw=2)
ax.plot(x_smo, y_smo, label='Smoothed', color='#337ab7', lw=2)
ax.scatter([0], [0], c='green', s=100, marker='o', zorder=5, label='Start')
ax.scatter([x_raw[-1]], [y_raw[-1]], c='#d9534f', s=80, marker='x', zorder=5)
ax.scatter([x_smo[-1]], [y_smo[-1]], c='#337ab7', s=80, marker='x', zorder=5)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Top-down Path Comparison')
ax.grid(alpha=0.3); ax.legend(); ax.axis('equal')
plt.tight_layout(); plt.savefig(f'{out_dir}/path_xy.png', dpi=150); plt.close()
print(f'[PLOT] path_xy.png')

# Plot 3: forward velocity over time
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_raw, vx_raw, label='Raw', color='#d9534f', lw=1.5, alpha=0.9)
ax.plot(t_smo, vx_smo, label='Smoothed', color='#337ab7', lw=1.5, alpha=0.9)
ax.axhline(y=float(raw['cmd_vx']), color='gray', ls='--', label='Commanded 1.0 m/s')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Forward velocity vx (m/s)')
ax.set_title('Forward Velocity Tracking')
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f'{out_dir}/vx_vs_time.png', dpi=150); plt.close()
print(f'[PLOT] vx_vs_time.png')

# Plot 4: joint action jitter — action rate L2 per timestep
a_raw_exec = raw['actions_exec']
a_smo_exec = smo['actions_exec']
rate_raw = np.linalg.norm(np.diff(a_raw_exec, axis=0), axis=1)
rate_smo = np.linalg.norm(np.diff(a_smo_exec, axis=0), axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_raw[1:], rate_raw, label=f'Raw  (mean={rate_raw.mean():.3f})',
        color='#d9534f', lw=1.2, alpha=0.9)
ax.plot(t_smo[1:], rate_smo, label=f'Smoothed  (mean={rate_smo.mean():.3f})',
        color='#337ab7', lw=1.2, alpha=0.9)
ax.set_xlabel('Time (s)'); ax.set_ylabel('‖a[t] − a[t−1]‖₂  (rad)')
ax.set_title('Per-step Action Change (Jitter)')
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f'{out_dir}/action_rate.png', dpi=150); plt.close()
print(f'[PLOT] action_rate.png')

# Plot 5: summary bar chart
labels = ['Forward\nprogress (m)', 'Mean\nvx (m/s)', 'Path length\n(m)', 'Mean action\njitter']
raw_vals = [float(x_raw[-1]), float(vx_raw.mean()), float(dist_raw[-1]), float(rate_raw.mean())]
smo_vals = [float(x_smo[-1]), float(vx_smo.mean()), float(dist_smo[-1]), float(rate_smo.mean())]
x = np.arange(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, raw_vals, w, label='Raw', color='#d9534f')
b2 = ax.bar(x + w/2, smo_vals, w, label='Smoothed', color='#337ab7')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title(f'A→B Race Summary (500 steps = {t_raw[-1]:.1f}s, cmd_vx=1.0 m/s)')
ax.legend(); ax.grid(alpha=0.3, axis='y')
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)
plt.tight_layout(); plt.savefig(f'{out_dir}/summary_bars.png', dpi=150); plt.close()
print(f'[PLOT] summary_bars.png')

print('\n=== SUMMARY ===')
print(f'{"":<28}{"Raw":>12}{"Smoothed":>14}{"Δ":>12}')
print(f'{"Forward progress (m)":<28}{x_raw[-1]:>12.3f}{x_smo[-1]:>14.3f}{x_smo[-1]-x_raw[-1]:>+12.3f}')
print(f'{"Total path (m)":<28}{dist_raw[-1]:>12.3f}{dist_smo[-1]:>14.3f}{dist_smo[-1]-dist_raw[-1]:>+12.3f}')
print(f'{"Mean vx (m/s)":<28}{vx_raw.mean():>12.3f}{vx_smo.mean():>14.3f}{vx_smo.mean()-vx_raw.mean():>+12.3f}')
print(f'{"Mean action jitter":<28}{rate_raw.mean():>12.4f}{rate_smo.mean():>14.4f}{rate_smo.mean()-rate_raw.mean():>+12.4f}')
print(f'{"Straightness (x/path)":<28}{x_raw[-1]/dist_raw[-1]:>12.3f}{x_smo[-1]/dist_smo[-1]:>14.3f}{x_smo[-1]/dist_smo[-1]-x_raw[-1]/dist_raw[-1]:>+12.3f}')
