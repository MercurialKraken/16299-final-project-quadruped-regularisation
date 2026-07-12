"""
flow_matching.py — Conditional Flow Matching to smooth Go1 RL actions.

Source distribution: raw RL actions (jittery)
Target distribution: low-pass filtered actions (smooth)
Conditioning: robot observation state

Usage:
    python flow_matching.py --train        # train the model
    python flow_matching.py --eval         # evaluate smoothed vs baseline
    python flow_matching.py --train --eval # both
    python flow_matching.py --eval --t_end 0.3  # partial integration (less aggressive)
    python flow_matching.py --train --cutoff 15  # higher LP cutoff
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = r"A:\IsaacLab\rollout_data.npz"
MODEL_PATH = r"A:\IsaacLab\flow_model.pt"

# ─── Low-pass filter to create smooth target actions ──────────────────────────

def butter_lowpass_filter(data, cutoff_freq=15.0, fs=50.0, order=2):
    """Apply Butterworth low-pass filter along time axis (axis=0).
    cutoff_freq: cutoff in Hz (15 Hz preserves gait dynamics, removes high-freq jitter)
    fs: sampling frequency (50 Hz = 1/0.02s)
    """
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    norm_cutoff = cutoff_freq / nyq
    b, a = butter(order, norm_cutoff, btype="low", analog=False)
    filtered = np.zeros_like(data)
    for j in range(data.shape[1]):
        filtered[:, j] = filtfilt(b, a, data[:, j])
    return filtered


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ActionSmoothingDataset(Dataset):
    """Each sample: (state, raw_action, smooth_action)."""
    def __init__(self, data_path, cutoff=15.0):
        d = np.load(data_path)
        self.obs = d["obs"].astype(np.float32)
        raw_actions = d["actions"].astype(np.float32)
        ep_lengths = d["ep_lengths"]
        smooth_actions = np.zeros_like(raw_actions)
        idx = 0
        for L in ep_lengths:
            ep_raw = raw_actions[idx:idx+L]
            if L > 12:
                smooth_actions[idx:idx+L] = butter_lowpass_filter(ep_raw, cutoff_freq=cutoff)
            else:
                smooth_actions[idx:idx+L] = ep_raw
            idx += L
        self.x0 = raw_actions
        self.x1 = smooth_actions

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return (torch.tensor(self.obs[i]),
                torch.tensor(self.x0[i]),
                torch.tensor(self.x1[i]))


# ─── Velocity field network ──────────────────────────────────────────────────

class VelocityNet(nn.Module):
    """Predicts the velocity field v(x_t, t, s) for flow matching.
    Input: [x_t (12), t (1), state (48)] -> v (12)
    """
    def __init__(self, action_dim=12, state_dim=48, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + 1 + state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x_t, t, state):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x_t, t, state], dim=-1)
        return self.net(inp)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(epochs=200, batch_size=512, lr=1e-3, cutoff=15.0):
    print(f"[flow] Loading data from {DATA_PATH}")
    print(f"[flow] LP cutoff = {cutoff} Hz")
    ds = ActionSmoothingDataset(DATA_PATH, cutoff=cutoff)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[flow] Dataset: {len(ds)} samples, {len(loader)} batches/epoch")

    model = VelocityNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0.0
        for state, x0, x1 in loader:
            state, x0, x1 = state.to(DEVICE), x0.to(DEVICE), x1.to(DEVICE)
            t = torch.rand(x0.shape[0], 1, device=DEVICE)
            x_t = (1 - t) * x0 + t * x1
            target_v = x1 - x0
            pred_v = model(x_t, t, state)
            loss = nn.functional.mse_loss(pred_v, target_v)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[flow] Model saved to {MODEL_PATH}")
    return model


# ─── Inference (ODE integration) ─────────────────────────────────────────────

@torch.no_grad()
def flow_smooth(model, raw_actions, states, n_steps=10, t_end=1.0):
    """Integrate the learned velocity field from x0 (raw) toward x1 (smooth).
    
    t_end controls how far toward the smooth target we go:
      t_end=1.0 -> full smoothing (maps to LP-filtered target)
      t_end=0.3 -> partial smoothing (30% of the way, preserves more gait dynamics)
      t_end=0.0 -> no smoothing (returns raw actions)
    """
    model.eval()
    x = raw_actions.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = torch.full((x.shape[0], 1), i * dt, device=x.device)
        v = model(x, t_val, states)
        x = x + v * dt
    return x


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(t_end=1.0, cutoff=15.0):
    print(f"\n[eval] Loading data and model...")
    print(f"[eval] Integration endpoint t_end={t_end}, LP cutoff={cutoff} Hz")
    d = np.load(DATA_PATH)
    obs = torch.tensor(d["obs"].astype(np.float32)).to(DEVICE)
    raw = torch.tensor(d["actions"].astype(np.float32)).to(DEVICE)
    base_vel = d["base_vel"]  # (T, 3) — [vx, vy, vyaw]
    ep_lengths = d["ep_lengths"]

    model = VelocityNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    raw_np = raw.cpu().numpy()

    # LP-filtered reference at the specified cutoff
    lp_actions = np.zeros_like(raw_np)
    idx = 0
    for L in ep_lengths:
        if L > 12:
            lp_actions[idx:idx+L] = butter_lowpass_filter(raw_np[idx:idx+L], cutoff_freq=cutoff)
        else:
            lp_actions[idx:idx+L] = raw_np[idx:idx+L]
        idx += L

    # Sweep t_end values to find the sweet spot
    t_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] if t_end == 1.0 else [t_end]
    
    print(f"\n{'='*90}")
    print(f"  BASE VELOCITY FROM ROLLOUT DATA (ground truth from raw policy)")
    print(f"{'='*90}")
    print(f"  Mean forward vel (vx): {np.mean(base_vel[:, 0]):.4f} m/s")
    print(f"  Mean lateral vel (vy): {np.mean(base_vel[:, 1]):.4f} m/s")
    print(f"  Std  forward vel (vx): {np.std(base_vel[:, 0]):.4f} m/s")
    
    # Per-episode velocity
    idx = 0
    for ep_i, L in enumerate(ep_lengths):
        ep_vx = base_vel[idx:idx+L, 0]
        print(f"    Episode {ep_i}: mean_vx={np.mean(ep_vx):.4f}, max_vx={np.max(ep_vx):.4f}")
        idx += L

    print(f"\n{'='*90}")
    print(f"  ACTION SMOOTHNESS vs DEVIATION FROM RAW (sweep t_end)")
    print(f"  Lower action_rate = smoother | Lower deviation = closer to original policy")
    print(f"{'='*90}")
    print(f"  {'t_end':>6s} | {'act_rate_rms':>12s} | {'jitter_reduction':>16s} | {'max_dev_from_raw':>16s} | {'mean_dev_from_raw':>17s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*16}-+-{'-'*16}-+-{'-'*17}")
    
    raw_rate = np.sqrt(np.mean(np.diff(raw_np, axis=0)**2))
    
    best_results = {}
    for t_e in t_values:
        smoothed = flow_smooth(model, raw, obs, n_steps=20, t_end=t_e)
        sm_np = smoothed.cpu().numpy()
        
        # Action rate (smoothness)
        sm_rate = np.sqrt(np.mean(np.diff(sm_np, axis=0)**2))
        jitter_red = (1 - sm_rate / raw_rate) * 100
        
        # Deviation from raw actions (how much we changed the policy)
        dev = np.abs(sm_np - raw_np)
        max_dev = np.max(dev)
        mean_dev = np.mean(dev)
        
        print(f"  {t_e:>6.2f} | {sm_rate:>12.4f} | {jitter_red:>15.1f}% | {max_dev:>16.4f} | {mean_dev:>17.4f}")
        best_results[t_e] = {"smoothed": sm_np, "rate": sm_rate, "jitter_red": jitter_red, "mean_dev": mean_dev}

    # LP reference stats
    lp_rate = np.sqrt(np.mean(np.diff(lp_actions, axis=0)**2))
    print(f"  {'LP':>6s} | {lp_rate:>12.4f} | {(1-lp_rate/raw_rate)*100:>15.1f}% | {'(oracle)':>16s} | {'(oracle)':>17s}")
    print(f"  {'raw':>6s} | {raw_rate:>12.4f} | {'0.0%':>16s} | {'0.0000':>16s} | {'0.0000':>17s}")

    # KEY INSIGHT: show how much the action magnitudes change
    # If smoothing kills action magnitude, the robot will stop moving
    print(f"\n{'='*90}")
    print(f"  ACTION MAGNITUDE ANALYSIS (does smoothing kill the actions?)")
    print(f"{'='*90}")
    raw_mag = np.mean(np.abs(raw_np))
    print(f"  Raw action mean |a|: {raw_mag:.4f}")
    for t_e in t_values:
        sm_np = best_results[t_e]["smoothed"]
        sm_mag = np.mean(np.abs(sm_np))
        mag_ratio = sm_mag / raw_mag * 100
        print(f"  t_end={t_e:.2f} mean |a|: {sm_mag:.4f}  ({mag_ratio:.1f}% of raw)")
    lp_mag = np.mean(np.abs(lp_actions))
    print(f"  LP filter  mean |a|: {lp_mag:.4f}  ({lp_mag/raw_mag*100:.1f}% of raw)")

    # Per-joint analysis: which joints are most affected?
    print(f"\n{'='*90}")
    print(f"  PER-JOINT ACTION MAGNITUDE (raw vs smoothed at best t_end)")
    print(f"  Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,")
    print(f"               RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf")
    print(f"{'='*90}")
    joint_names = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
                   "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]
    
    # Use t_end=0.3 as the "conservative" choice for per-joint
    t_check = 0.3 if 0.3 in best_results else t_values[0]
    sm_check = best_results[t_check]["smoothed"]
    
    for j, name in enumerate(joint_names):
        raw_j = np.mean(np.abs(raw_np[:, j]))
        sm_j = np.mean(np.abs(sm_check[:, j]))
        ratio = sm_j / raw_j * 100 if raw_j > 1e-6 else 100.0
        print(f"  {name:>10s}: raw={raw_j:.4f}  smooth={sm_j:.4f}  ({ratio:.1f}%)")

    # Save best smoothed actions
    best_t = min(best_results.keys(), key=lambda t: best_results[t]["mean_dev"])
    sm_best = best_results[best_t]["smoothed"]
    np.savez(r"A:\IsaacLab\smoothed_actions.npz",
             smoothed=sm_best, raw=raw_np, lp_filtered=lp_actions,
             obs=d["obs"], base_vel=base_vel, t_end=best_t)
    print(f"\n[eval] Saved smoothed_actions.npz (t_end={best_t})")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--cutoff", type=float, default=15.0, help="LP filter cutoff Hz (default: 15)")
    parser.add_argument("--t_end", type=float, default=1.0, help="Integration endpoint (0.0-1.0, lower=less smoothing)")
    args = parser.parse_args()

    if not args.train and not args.eval:
        args.train = True
        args.eval = True

    if args.train:
        train(epochs=args.epochs, cutoff=args.cutoff)
    if args.eval:
        evaluate(t_end=args.t_end, cutoff=args.cutoff)
