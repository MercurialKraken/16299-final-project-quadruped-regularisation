"""
flow_matching_optimal.py — Train flow matching with short-horizon optimal targets.

Replaces LP-filtered targets with locally-optimal actions found via candidate
evaluation in generate_optimal_targets.py. The flow model learns:

    v_theta(x, t, s) ≈ x_star - x0

where x_star is the best candidate action based on multi-objective cost
(tracking + smoothness + energy + stability), NOT a low-pass filter.

This turns flow matching from "learn to smooth" into "learn to improve."

Usage:
    python flow_matching_optimal.py --train --data optimal_targets_flat.npz
    python flow_matching_optimal.py --eval --data optimal_targets_flat.npz
    python flow_matching_optimal.py --train --eval --data optimal_targets_flat.npz
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Dataset ─────────────────────────────────────────────────────────────────

class OptimalTargetDataset(Dataset):
    """Training dataset using (state, x0, x_star) from candidate evaluation.

    Each sample provides:
        state: robot observation at time t
        x0:    PPO action (raw)
        x1:    locally-optimal action (x_star from random shooting)

    The flow model learns v(x_t, t, s) ≈ x_star - x0.
    """
    def __init__(self, data_path, min_improvement=0.0):
        """
        Args:
            data_path: path to optimal_targets_*.npz
            min_improvement: only use samples where x_star actually improved
                             on the PPO action by at least this amount.
                             Set to 0.0 to use all samples (recommended for
                             first pass; some "no improvement" samples teach
                             the model when to leave actions alone).
        """
        d = np.load(data_path)
        self.obs = d["obs"].astype(np.float32)
        self.x0 = d["x0"].astype(np.float32)
        self.x_star = d["x_star"].astype(np.float32)
        improvement = d["cost_improvement"]

        if min_improvement > 0:
            mask = improvement >= min_improvement
            self.obs = self.obs[mask]
            self.x0 = self.x0[mask]
            self.x_star = self.x_star[mask]
            print(f"[dataset] Filtered: {mask.sum()}/{len(mask)} samples with improvement >= {min_improvement}")

        # Stats
        displacement = np.sqrt(np.mean((self.x_star - self.x0) ** 2, axis=1))
        print(f"[dataset] {len(self.obs)} samples, obs_dim={self.obs.shape[1]}")
        print(f"[dataset] Mean |x_star - x0|: {displacement.mean():.4f}")
        print(f"[dataset] Median |x_star - x0|: {np.median(displacement):.4f}")
        print(f"[dataset] Samples where x_star == x0: {(displacement < 1e-6).sum()}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return (torch.tensor(self.obs[i]),
                torch.tensor(self.x0[i]),
                torch.tensor(self.x_star[i]))


# ─── Network (same architecture as before) ──────────────────────────────────

class VelocityNet(nn.Module):
    """Velocity field v(x_t, t, s) for flow matching.
    Input: [x_t(12), t(1), state(obs_dim)] → v(12)
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
        return self.net(torch.cat([x_t, t, state], dim=-1))


# ─── Training ────────────────────────────────────────────────────────────────

def train(data_path, model_path, state_dim, epochs=200, batch_size=512, lr=1e-3,
          min_improvement=0.0):
    print(f"\n{'='*70}")
    print(f"  FLOW MATCHING WITH OPTIMAL TARGETS")
    print(f"  Data: {data_path}")
    print(f"  Model: {model_path}")
    print(f"  State dim: {state_dim}")
    print(f"{'='*70}\n")

    ds = OptimalTargetDataset(data_path, min_improvement=min_improvement)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[train] {len(ds)} samples, {len(loader)} batches/epoch")

    model = VelocityNet(state_dim=state_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for state, x0, x_star in loader:
            state, x0, x_star = state.to(DEVICE), x0.to(DEVICE), x_star.to(DEVICE)

            # Flow matching: interpolate between x0 and x_star
            t = torch.rand(x0.shape[0], 1, device=DEVICE)
            x_t = (1 - t) * x0 + t * x_star

            # Target velocity = displacement
            target_v = x_star - x0

            # Predict
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

    # Save model with metadata
    torch.save({
        "state_dict": model.state_dict(),
        "state_dim": state_dim,
        "method": "optimal_targets",
    }, model_path)

    loss_path = model_path.replace(".pt", "_loss.npy")
    np.save(loss_path, np.array(losses))
    print(f"\n[train] Model saved: {model_path}")
    print(f"[train] Loss curve saved: {loss_path}")
    print(f"[train] Final loss: {losses[-1]:.6f}")
    return model


# ─── Offline Evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def flow_smooth(model, raw_actions, states, n_steps=20, t_end=0.3):
    """Integrate velocity field from x0 toward x_star."""
    model.eval()
    x = raw_actions.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = torch.full((x.shape[0], 1), i * dt, device=x.device)
        x = x + model(x, t_val, states) * dt
    return x


def evaluate(data_path, model_path, state_dim):
    print(f"\n{'='*70}")
    print(f"  OFFLINE EVALUATION — OPTIMAL TARGETS MODEL")
    print(f"{'='*70}\n")

    d = np.load(data_path)
    obs = torch.tensor(d["obs"].astype(np.float32)).to(DEVICE)
    x0 = torch.tensor(d["x0"].astype(np.float32)).to(DEVICE)
    x_star = torch.tensor(d["x_star"].astype(np.float32)).to(DEVICE)
    x0_np = x0.cpu().numpy()
    x_star_np = x_star.cpu().numpy()

    # Load model
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dim"] if "state_dim" in ckpt else state_dim
        model = VelocityNet(state_dim=sd).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = VelocityNet(state_dim=state_dim).to(DEVICE)
        model.load_state_dict(ckpt)

    raw_rate = np.sqrt(np.mean(np.diff(x0_np, axis=0) ** 2))
    star_rate = np.sqrt(np.mean(np.diff(x_star_np, axis=0) ** 2))

    print(f"  {'Method':>25s} | {'act_rate_rms':>12s} | {'reduction':>10s} | {'|dev from x0|':>14s}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*14}")
    print(f"  {'RAW (x0)':>25s} | {raw_rate:>12.4f} | {'—':>10s} | {'—':>14s}")
    print(f"  {'x_star (oracle)':>25s} | {star_rate:>12.4f} | {(1-star_rate/raw_rate)*100:>9.1f}% | {np.mean(np.abs(x_star_np - x0_np)):>14.4f}")

    for t_end in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        smoothed = flow_smooth(model, x0, obs, n_steps=20, t_end=t_end).cpu().numpy()
        sm_rate = np.sqrt(np.mean(np.diff(smoothed, axis=0) ** 2))
        dev = np.mean(np.abs(smoothed - x0_np))
        red = (1 - sm_rate / raw_rate) * 100
        print(f"  {f'Flow t_end={t_end}':>25s} | {sm_rate:>12.4f} | {red:>9.1f}% | {dev:>14.4f}")

    # Compare with LP baseline if available
    lp_model_path = os.path.join(os.path.dirname(model_path), "flow_model_adaptive.pt")
    if os.path.exists(lp_model_path):
        print(f"\n  --- LP Baseline (flow_model_adaptive.pt) ---")
        lp_ckpt = torch.load(lp_model_path, map_location=DEVICE, weights_only=True)
        if isinstance(lp_ckpt, dict) and "state_dict" in lp_ckpt:
            lp_model = VelocityNet(state_dim=lp_ckpt.get("state_dim", state_dim)).to(DEVICE)
            lp_model.load_state_dict(lp_ckpt["state_dict"])
        else:
            lp_model = VelocityNet(state_dim=state_dim).to(DEVICE)
            lp_model.load_state_dict(lp_ckpt)

        for t_end in [0.3, 1.0]:
            smoothed = flow_smooth(lp_model, x0, obs, n_steps=20, t_end=t_end).cpu().numpy()
            sm_rate = np.sqrt(np.mean(np.diff(smoothed, axis=0) ** 2))
            dev = np.mean(np.abs(smoothed - x0_np))
            red = (1 - sm_rate / raw_rate) * 100
            print(f"  {f'LP-flow t_end={t_end}':>25s} | {sm_rate:>12.4f} | {red:>9.1f}% | {dev:>14.4f}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to optimal_targets_*.npz")
    parser.add_argument("--model", type=str, default=None,
                        help="Model output path (default: flow_model_optimal.pt)")
    parser.add_argument("--state_dim", type=int, default=48,
                        help="Observation dimension (48=flat, 235=rough)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min_improvement", type=float, default=0.0,
                        help="Only train on samples with at least this much improvement")
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(os.path.dirname(args.data), "flow_model_optimal.pt")

    if not args.train and not args.eval:
        args.train = True
        args.eval = True

    if args.train:
        train(args.data, args.model, args.state_dim, epochs=args.epochs,
              min_improvement=args.min_improvement)
    if args.eval:
        evaluate(args.data, args.model, args.state_dim)
