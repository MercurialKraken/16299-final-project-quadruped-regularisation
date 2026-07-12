"""LP-filter the balanced optimal targets, then retrain flow model."""
import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff_freq=15.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    norm_cutoff = cutoff_freq / nyq
    b, a = butter(order, norm_cutoff, btype="low", analog=False)
    filtered = np.zeros_like(data)
    for j in range(data.shape[1]):
        filtered[:, j] = filtfilt(b, a, data[:, j])
    return filtered

# Load balanced optimal targets
d = np.load('optimal_targets_balanced.npz')
obs = d['obs']
x0 = d['x0']           # raw PPO actions
x_star = d['x_star']   # balanced optimal targets
ep_lengths = d['ep_lengths']

print(f"Loaded: {len(x_star)} samples, {len(ep_lengths)} episodes")
print(f"x_star range: [{x_star.min():.3f}, {x_star.max():.3f}]")

# LP-filter x_star per episode
x_star_smooth = np.zeros_like(x_star)
idx = 0
for L in ep_lengths:
    ep = x_star[idx:idx+L]
    if L > 12:
        x_star_smooth[idx:idx+L] = butter_lowpass_filter(ep, cutoff_freq=15.0)
    else:
        x_star_smooth[idx:idx+L] = ep
    idx += L

displacement_before = np.sqrt(np.mean((x_star - x0) ** 2))
displacement_after = np.sqrt(np.mean((x_star_smooth - x0) ** 2))
smoothing_delta = np.sqrt(np.mean((x_star_smooth - x_star) ** 2))

print(f"|x_star - x0| (before LP): {displacement_before:.4f}")
print(f"|x_star_smooth - x0| (after LP): {displacement_after:.4f}")
print(f"|x_star_smooth - x_star| (LP moved targets by): {smoothing_delta:.4f}")

# Save new targets with x_star replaced by LP-filtered version
np.savez(
    'optimal_targets_balanced_lp.npz',
    obs=obs,
    x0=x0,
    x_star=x_star_smooth,
    cost_x_star=d['cost_x_star'],
    cost_improvement=d['cost_improvement'],
    ep_lengths=ep_lengths,
    K=d['K'], H=d['H'], sigma=d['sigma'], cmd_vx=d['cmd_vx'],
    w_tracking=d['w_tracking'], w_jerk=d['w_jerk'],
    w_energy=d['w_energy'], w_stability=d['w_stability'],
)
print("Saved: optimal_targets_balanced_lp.npz")
