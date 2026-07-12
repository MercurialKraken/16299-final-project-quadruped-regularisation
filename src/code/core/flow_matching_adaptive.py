"""
flow_matching_adaptive.py — Per-joint adaptive cutoff flow matching.

Improves on flow_matching.py by using different LP cutoff frequencies per joint,
based on in-sim verification showing some joints benefit from aggressive smoothing
while others get WORSE. Joints that carry the gait signal (thighs, some hips)
need a high cutoff to preserve dynamics; joints that are mostly jittery (calves,
some hips) can tolerate lower cutoffs for more smoothing.

Usage:
    python flow_matching_adaptive.py --train
    python flow_matching_adaptive.py --eval
    python flow_matching_adaptive.py --train --eval
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = r"A:\IsaacLab\rollout_data.npz"
MODEL_PATH = r"A:\IsaacLab\flow_model_adaptive.pt"

# ─── Per-joint adaptive cutoff frequencies ────────────────────────────────────
# Based on in-sim verification (smoothing_verification.txt):
#   - FR_calf: 17% improvement → aggressive smoothing OK (low cutoff)
#   - FR_hip:  30% improvement → aggressive smoothing OK
#   - FL_hip:  9% improvement  → moderate smoothing
#   - FL_thigh: 12% improvement → moderate smoothing
#   - RL_thigh: 18% improvement at t=0.3 → moderate smoothing
#   - RR_thigh: 10% WORSE → minimal smoothing (high cutoff)
#   - RL_hip: 5% WORSE at t=0.3 → minimal smoothing
#   - FL_calf: 4% WORSE → minimal smoothing
#   - FR_thigh: mixed results → moderate cutoff

JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Per-joint cutoff frequencies (Hz) — tuned from verification data
# Low cutoff = more smoothing, High cutoff = less smoothing (preserve dynamics)
JOINT_CUTOFFS = {
    "FL_hip":    10.0,   # benefits from smoothing
    "FL_thigh":  12.0,   # moderate benefit
    "FL_calf":   22.0,   # gets WORSE — barely filter
    "FR_hip":     8.0,   # strong benefit (30% reduction)
    "FR_thigh":  15.0,   # mixed — keep at baseline
    "FR_calf":    8.0,   # strong benefit (17% reduction)
    "RL_hip":    22.0,   # gets WORSE — barely filter
    "RL_thigh":  10.0,   # good benefit (18% reduction)
    "RL_calf":   15.0,   # moderate
    "RR_hip":    12.0,   # mixed, slight benefit at t=1.0
    "RR_thigh":  25.0,   # gets 10% WORSE — almost no filter
    "RR_calf":   10.0,   # good benefit (16% reduction)
}


def get_cutoff_array():
    """Return array of shape (12,) with per-joint cutoff frequencies."""
    return np.array([JOINT_CUTOFFS[name] for name in JOINT_NAMES])


# ─── Low-pass filter with per-joint cutoffs ───────────────────────────────────

def butter_lowpass_filter_adaptive(data, cutoffs=None, fs=50.0, order=2):
    """Apply Butterworth LP filter with DIFFERENT cutoff per joint.
    data: (T, 12) array of joint actions
    cutoffs: (12,) array of per-joint cutoff frequencies in Hz
    """
    from scipy.signal import butter, filtfilt
    if cutoffs is None:
        cutoffs = get_cutoff_array()
    nyq = 0.5 * fs
    filtered = np.zeros_like(data)
    for j in range(data.shape[1]):
        norm_cutoff = cutoffs[j] / nyq
        # Clamp to valid range (0, 1)
        norm_cutoff = min(max(norm_cutoff, 0.01), 0.99)
        b, a = butter(order, norm_cutoff, btype="low", analog=False)
        filtered[:, j] = filtfilt(b, a, data[:, j])
    return filtered


def butter_lowpass_filter_uniform(data, cutoff_freq=15.0, fs=50.0, order=2):
    """Original uniform cutoff for comparison."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    norm_cutoff = cutoff_freq / nyq
    b, a = butter(order, norm_cutoff, btype="low", analog=False)
    filtered = np.zeros_like(data)
    for j in range(data.shape[1]):
        filtered[:, j] = filtfilt(b, a, data[:, j])
    return filtered


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AdaptiveActionSmoothingDataset(Dataset):
    """Each sample: (state, raw_action, smooth_action).
    Uses per-joint adaptive cutoffs for the smooth targets."""
    def __init__(self, data_path, adaptive=True):
        d = np.load(data_path)
        self.obs = d["obs"].astype(np.float32)
        raw_actions = d["actions"].astype(np.float32)
        ep_lengths = d["ep_lengths"]
        smooth_actions = np.zeros_like(raw_actions)
        cutoffs = get_cutoff_array() if adaptive else None
        idx = 0
        for L in ep_lengths:
            ep_raw = raw_actions[idx:idx+L]
            if L > 12:
                if adaptive:
                    smooth_actions[idx:idx+L] = butter_lowpass_filter_adaptive(ep_raw, cutoffs)
                else:
                    smooth_actions[idx:idx+L] = butter_lowpass_filter_uniform(ep_raw, cutoff_freq=15.0)
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

def train(epochs=200, batch_size=512, lr=1e-3, adaptive=True):
    print(f"[flow-adaptive] Loading data from {DATA_PATH}")
    mode = "ADAPTIVE per-joint cutoffs" if adaptive else "UNIFORM 15 Hz cutoff"
    print(f"[flow-adaptive] Mode: {mode}")
    if adaptive:
        cutoffs = get_cutoff_array()
        for name, c in zip(JOINT_NAMES, cutoffs):
            print(f"    {name:>10s}: {c:.0f} Hz")

    ds = AdaptiveActionSmoothingDataset(DATA_PATH, adaptive=adaptive)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[flow-adaptive] Dataset: {len(ds)} samples, {len(loader)} batches/epoch")

    model = VelocityNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses = []
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
        losses.append(avg)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), MODEL_PATH)
    np.save(r"A:\IsaacLab\adaptive_training_loss.npy", np.array(losses))
    print(f"[flow-adaptive] Model saved to {MODEL_PATH}")
    print(f"[flow-adaptive] Training loss saved to adaptive_training_loss.npy")
    return model


# ─── Inference (ODE integration) ─────────────────────────────────────────────

@torch.no_grad()
def flow_smooth(model, raw_actions, states, n_steps=10, t_end=1.0):
    """Integrate learned velocity field from x0 (raw) toward x1 (smooth)."""
    model.eval()
    x = raw_actions.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = torch.full((x.shape[0], 1), i * dt, device=x.device)
        v = model(x, t_val, states)
        x = x + v * dt
    return x


# ─── Evaluation with comparison ──────────────────────────────────────────────

def evaluate(t_end=1.0):
    """Compare adaptive model vs uniform 15Hz baseline on offline data."""
    print(f"\n[eval] Loading data...")
    d = np.load(DATA_PATH)
    obs = torch.tensor(d["obs"].astype(np.float32)).to(DEVICE)
    raw = torch.tensor(d["actions"].astype(np.float32)).to(DEVICE)
    ep_lengths = d["ep_lengths"]
    raw_np = raw.cpu().numpy()

    # Load adaptive model
    model_adaptive = VelocityNet().to(DEVICE)
    model_adaptive.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    # Load uniform baseline model (if exists)
    uniform_path = r"A:\IsaacLab\flow_model.pt"
    has_uniform = os.path.exists(uniform_path)
    if has_uniform:
        model_uniform = VelocityNet().to(DEVICE)
        model_uniform.load_state_dict(torch.load(uniform_path, map_location=DEVICE, weights_only=True))

    # Compute smoothed actions
    t_values = [0.3, 0.5, 1.0]
    results = {}
    
    for t_e in t_values:
        sm_adaptive = flow_smooth(model_adaptive, raw, obs, n_steps=20, t_end=t_e).cpu().numpy()
        results[f"adaptive_t{t_e}"] = sm_adaptive
        if has_uniform:
            sm_uniform = flow_smooth(model_uniform, raw, obs, n_steps=20, t_end=t_e).cpu().numpy()
            results[f"uniform_t{t_e}"] = sm_uniform

    # Also compute the LP filter targets for reference
    cutoffs_adaptive = get_cutoff_array()
    lp_adaptive = np.zeros_like(raw_np)
    lp_uniform = np.zeros_like(raw_np)
    idx = 0
    for L in ep_lengths:
        if L > 12:
            lp_adaptive[idx:idx+L] = butter_lowpass_filter_adaptive(raw_np[idx:idx+L], cutoffs_adaptive)
            lp_uniform[idx:idx+L] = butter_lowpass_filter_uniform(raw_np[idx:idx+L], cutoff_freq=15.0)
        else:
            lp_adaptive[idx:idx+L] = raw_np[idx:idx+L]
            lp_uniform[idx:idx+L] = raw_np[idx:idx+L]
        idx += L

    # ─── Print comparison table ───────────────────────────────────────────────
    raw_rate = np.sqrt(np.mean(np.diff(raw_np, axis=0)**2))
    
    print(f"\n{'='*95}")
    print(f"  ADAPTIVE vs UNIFORM COMPARISON (offline action analysis)")
    print(f"{'='*95}")
    print(f"  {'Config':>25s} | {'act_rate_rms':>12s} | {'reduction':>10s} | {'mean|dev|':>10s} | {'mean|a|':>8s}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'RAW':>25s} | {raw_rate:>12.4f} | {'---':>10s} | {'---':>10s} | {np.mean(np.abs(raw_np)):>8.4f}")

    for t_e in t_values:
        for label, key in [("Adaptive", f"adaptive_t{t_e}"), ("Uniform 15Hz", f"uniform_t{t_e}")]:
            if key not in results:
                continue
            sm = results[key]
            rate = np.sqrt(np.mean(np.diff(sm, axis=0)**2))
            red = (1 - rate / raw_rate) * 100
            dev = np.mean(np.abs(sm - raw_np))
            mag = np.mean(np.abs(sm))
            print(f"  {f'{label} t={t_e}':>25s} | {rate:>12.4f} | {red:>9.1f}% | {dev:>10.4f} | {mag:>8.4f}")

    # LP oracle references
    lp_a_rate = np.sqrt(np.mean(np.diff(lp_adaptive, axis=0)**2))
    lp_u_rate = np.sqrt(np.mean(np.diff(lp_uniform, axis=0)**2))
    print(f"  {'LP Adaptive (oracle)':>25s} | {lp_a_rate:>12.4f} | {(1-lp_a_rate/raw_rate)*100:>9.1f}% | {'---':>10s} | {np.mean(np.abs(lp_adaptive)):>8.4f}")
    print(f"  {'LP Uniform 15Hz (oracle)':>25s} | {lp_u_rate:>12.4f} | {(1-lp_u_rate/raw_rate)*100:>9.1f}% | {'---':>10s} | {np.mean(np.abs(lp_uniform)):>8.4f}")

    # ─── Per-joint comparison ─────────────────────────────────────────────────
    t_compare = 0.3
    sm_a = results[f"adaptive_t{t_compare}"]
    
    print(f"\n{'='*95}")
    print(f"  PER-JOINT ACTION RATE RMS — Adaptive vs Uniform (t_end={t_compare})")
    print(f"  Cutoff shown for adaptive model. Lower rate = smoother.")
    print(f"{'='*95}")
    print(f"  {'Joint':>10s} | {'Cutoff':>7s} | {'RAW':>8s} | {'Adaptive':>8s} | {'d Adapt':>8s} |", end="")
    if has_uniform:
        sm_u = results[f"uniform_t{t_compare}"]
        print(f" {'Uniform':>8s} | {'d Unif':>8s} |")
    else:
        print()

    cutoffs = get_cutoff_array()
    joint_results_adaptive = []
    joint_results_uniform = []
    
    for j, name in enumerate(JOINT_NAMES):
        raw_j_rate = np.sqrt(np.mean(np.diff(raw_np[:, j])**2))
        ada_j_rate = np.sqrt(np.mean(np.diff(sm_a[:, j])**2))
        ada_pct = (1 - ada_j_rate / raw_j_rate) * 100
        joint_results_adaptive.append(ada_pct)
        
        line = f"  {name:>10s} | {cutoffs[j]:>5.0f}Hz | {raw_j_rate:>8.4f} | {ada_j_rate:>8.4f} | {ada_pct:>+7.1f}% |"
        if has_uniform:
            uni_j_rate = np.sqrt(np.mean(np.diff(sm_u[:, j])**2))
            uni_pct = (1 - uni_j_rate / raw_j_rate) * 100
            joint_results_uniform.append(uni_pct)
            line += f" {uni_j_rate:>8.4f} | {uni_pct:>+7.1f}% |"
        print(line)

    # ─── Summary statistics ───────────────────────────────────────────────────
    ada_improved = sum(1 for p in joint_results_adaptive if p > 0)
    ada_mean = np.mean(joint_results_adaptive)
    print(f"\n  Adaptive: {ada_improved}/12 joints improved, mean reduction = {ada_mean:+.1f}%")
    if has_uniform and joint_results_uniform:
        uni_improved = sum(1 for p in joint_results_uniform if p > 0)
        uni_mean = np.mean(joint_results_uniform)
        print(f"  Uniform:  {uni_improved}/12 joints improved, mean reduction = {uni_mean:+.1f}%")

    # ─── Save all data for plotting ───────────────────────────────────────────
    save_dict = {
        "raw": raw_np,
        "lp_adaptive": lp_adaptive,
        "lp_uniform": lp_uniform,
        "joint_names": JOINT_NAMES,
        "cutoffs_adaptive": cutoffs,
        "ep_lengths": ep_lengths,
    }
    for key, val in results.items():
        save_dict[key] = val
    np.savez(r"A:\IsaacLab\adaptive_comparison.npz", **save_dict)
    print(f"\n[eval] Saved adaptive_comparison.npz for plotting")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--t_end", type=float, default=1.0)
    args = parser.parse_args()

    if not args.train and not args.eval:
        args.train = True
        args.eval = True

    if args.train:
        train(epochs=args.epochs)
    if args.eval:
        evaluate(t_end=args.t_end)
