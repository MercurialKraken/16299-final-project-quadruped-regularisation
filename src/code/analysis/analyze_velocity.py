"""
analyze_velocity.py — Offline analysis of raw vs smoothed actions.
Compares action profiles and checks if smoothing destroys gait structure.
Does NOT require Isaac Sim — runs on any Python with numpy/torch.
"""
import numpy as np
import torch
import torch.nn as nn
import sys

DATA_PATH = r"A:\IsaacLab\rollout_data.npz"
MODEL_PATH = r"A:\IsaacLab\flow_model.pt"

# Load rollout data
d = np.load(DATA_PATH)
obs = d["obs"]          # (5000, 48)
actions = d["actions"]  # (5000, 12)
base_vel = d["base_vel"]  # (5000, 3)
ep_lengths = d["ep_lengths"]

print("="*80)
print("  ROLLOUT DATA ANALYSIS")
print("="*80)

# 1) Base velocity from rollout (this is what the RAW policy achieved)
print("\n--- Base Velocity from Raw Policy Rollout ---")
print(f"  Overall: mean_vx={np.mean(base_vel[:,0]):.4f}, mean_vy={np.mean(base_vel[:,1]):.4f}")
idx = 0
for ep_i, L in enumerate(ep_lengths):
    ep_v = base_vel[idx:idx+L]
    cmd_vel = obs[idx:idx+L, 9:12]  # velocity_commands are obs[9:12]
    print(f"  Episode {ep_i} ({L} steps):")
    print(f"    Achieved:  mean_vx={np.mean(ep_v[:,0]):+.4f}, mean_vy={np.mean(ep_v[:,1]):+.4f}")
    print(f"    Commanded: mean_cmd_vx={np.mean(cmd_vel[:,0]):+.4f}, mean_cmd_vy={np.mean(cmd_vel[:,1]):+.4f}")
    # How well did it track?
    vx_err = np.mean(np.abs(ep_v[:,0] - cmd_vel[:,0]))
    print(f"    Tracking error |vx - cmd_vx|: {vx_err:.4f}")
    idx += L

# 2) Load flow model and compute smoothed actions at various t_end
class VelocityNet(nn.Module):
    def __init__(self, action_dim=12, state_dim=48, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + 1 + state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, x_t, t, state):
        if t.dim() == 1: t = t.unsqueeze(-1)
        return self.net(torch.cat([x_t, t, state], dim=-1))

@torch.no_grad()
def flow_smooth(model, raw, states, n_steps=20, t_end=1.0):
    x = raw.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = torch.full((x.shape[0], 1), i * dt, device=x.device)
        x = x + model(x, t_val, states) * dt
    return x

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = VelocityNet().to(dev)
model.load_state_dict(torch.load(MODEL_PATH, map_location=dev, weights_only=True))
model.eval()

obs_t = torch.tensor(obs.astype(np.float32)).to(dev)
act_t = torch.tensor(actions.astype(np.float32)).to(dev)

print("\n--- Action Analysis: Raw vs Smoothed at various t_end ---")
print(f"  {'t_end':>6s} | {'mean|act|':>10s} | {'act_rate':>10s} | {'jitter_red':>10s} | {'max_delta':>10s} | {'gait_freq_preserved':>20s}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*20}")

raw_np = actions
raw_rate = np.sqrt(np.mean(np.diff(raw_np, axis=0)**2))
raw_mag = np.mean(np.abs(raw_np))

# Compute dominant gait frequency from raw actions using FFT
from scipy.signal import welch
fs = 50.0
freqs, psd = welch(raw_np[:, 1], fs=fs, nperseg=256)  # FL_thigh as representative
peak_freq = freqs[np.argmax(psd)]
print(f"\n  Raw policy dominant gait frequency: {peak_freq:.1f} Hz (FL_thigh PSD)")
print(f"  Raw action magnitude: {raw_mag:.4f}")
print(f"  Raw action rate: {raw_rate:.4f}\n")

for t_end in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    if t_end == 0.0:
        sm_np = raw_np.copy()
        label = "raw"
    else:
        sm = flow_smooth(model, act_t, obs_t, t_end=t_end)
        sm_np = sm.cpu().numpy()
        label = f"{t_end:.1f}"
    
    sm_mag = np.mean(np.abs(sm_np))
    sm_rate = np.sqrt(np.mean(np.diff(sm_np, axis=0)**2))
    jitter_red = (1 - sm_rate / raw_rate) * 100
    max_delta = np.max(np.abs(sm_np - raw_np))
    
    # Check gait frequency preservation: compute PSD of smoothed FL_thigh
    _, psd_sm = welch(sm_np[:, 1], fs=fs, nperseg=256)
    sm_peak = freqs[np.argmax(psd_sm)]
    freq_match = "YES" if abs(sm_peak - peak_freq) < 1.0 else f"NO (shifted to {sm_peak:.1f} Hz)"
    
    print(f"  {label:>6s} | {sm_mag:>10.4f} | {sm_rate:>10.4f} | {jitter_red:>9.1f}% | {max_delta:>10.4f} | {freq_match:>20s}")

# 3) Critical check: temporal autocorrelation of actions
# A good walking gait should have periodic autocorrelation
# If smoothing destroys the periodicity, the robot will stop walking
print("\n--- Gait Periodicity Check (autocorrelation of FL_thigh action) ---")
from numpy.fft import fft, ifft

def autocorr(x):
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[n-1:] / result[n-1]  # normalize

raw_ac = autocorr(raw_np[:1000, 1])  # first episode, FL_thigh

# Find first peak after lag 0 (this is the gait period)
from scipy.signal import find_peaks
peaks, _ = find_peaks(raw_ac[1:], height=0.1, distance=3)
if len(peaks) > 0:
    gait_period_steps = peaks[0] + 1
    gait_period_sec = gait_period_steps * 0.02
    gait_freq = 1.0 / gait_period_sec
    print(f"  Raw gait period: {gait_period_steps} steps = {gait_period_sec:.3f}s = {gait_freq:.1f} Hz")
else:
    print(f"  WARNING: No periodic gait detected in raw actions!")
    gait_period_steps = 0

# Check if smoothed actions preserve the gait period
for t_end in [0.3, 0.5, 1.0]:
    sm = flow_smooth(model, act_t, obs_t, t_end=t_end)
    sm_np_ep = sm.cpu().numpy()[:1000, 1]
    sm_ac = autocorr(sm_np_ep)
    sm_peaks, _ = find_peaks(sm_ac[1:], height=0.1, distance=3)
    if len(sm_peaks) > 0:
        sm_period = sm_peaks[0] + 1
        print(f"  t_end={t_end:.1f}: gait period = {sm_period} steps ({sm_period*0.02:.3f}s)")
        if gait_period_steps > 0:
            period_err = abs(sm_period - gait_period_steps)
            print(f"    Period shift from raw: {period_err} steps ({'OK' if period_err <= 2 else 'CONCERNING'})")
    else:
        print(f"  t_end={t_end:.1f}: WARNING - no periodic gait detected!")

# 4) The real question: was the BASELINE policy even walking forward?
print("\n" + "="*80)
print("  KEY FINDING: BASELINE POLICY BEHAVIOR")
print("="*80)
mean_vx = np.mean(base_vel[:, 0])
if mean_vx < 0:
    print(f"  The RAW RL policy has mean vx = {mean_vx:.4f} m/s (WALKING BACKWARDS)")
    print(f"  This means the robot was already walking backwards BEFORE any smoothing.")
    print(f"  The smoothing is NOT the cause of backward walking.")
    print(f"  The issue is the RL policy itself or the velocity command.")
else:
    print(f"  The RAW RL policy has mean vx = {mean_vx:.4f} m/s (walking forward)")

# Check what velocity was commanded
cmd_vx = obs[:, 9]  # velocity command x
print(f"  Commanded vx: mean={np.mean(cmd_vx):.4f}, min={np.min(cmd_vx):.4f}, max={np.max(cmd_vx):.4f}")
print(f"  The env uses UniformVelocityCommand which randomizes the target.")
print(f"  If cmd_vx is near 0 or negative, the robot is SUPPOSED to go slow/backward.")

print("\nDone.")
