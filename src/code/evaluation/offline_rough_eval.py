"""Offline evaluation: compute raw vs smoothed metrics from rollout_data_rough.npz + flow model.
No Isaac Sim needed. Just numpy + torch."""
import numpy as np, torch, torch.nn as nn, json, os

BASE = r"A:\AllIsaac\IsaacLab"
DATA = os.path.join(BASE, "rollout_data_rough.npz")
MODEL = os.path.join(BASE, "flow_model_rough.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VelocityNet(nn.Module):
    def __init__(self, action_dim=12, state_dim=235, hidden=256):
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
def flow_smooth_batch(model, raw, state, n_steps=20, t_end=0.3, batch_size=512):
    """Apply flow smoothing to all actions in batches."""
    N = raw.shape[0]
    smoothed = torch.zeros_like(raw)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = raw[start:end].clone()
        s = state[start:end]
        dt = t_end / n_steps
        for i in range(n_steps):
            t_val = torch.full((x.shape[0], 1), i * dt, device=x.device)
            x = x + model(x, t_val, s) * dt
        smoothed[start:end] = x
    return smoothed

def compute_metrics(actions, base_vel, ep_lengths, label):
    """Compute per-episode and aggregate metrics."""
    DT = 0.02
    # Action rate
    ar = np.diff(actions, axis=0)
    action_rate_rms = float(np.sqrt(np.mean(ar**2)))
    
    # Velocity
    mean_vx = float(np.mean(base_vel[:, 0]))
    mean_abs_vy = float(np.mean(np.abs(base_vel[:, 1])))
    mean_abs_vz = float(np.mean(np.abs(base_vel[:, 2])))
    
    # Action magnitude
    action_mag = float(np.mean(np.abs(actions)))
    
    # Per-episode stats
    ep_vx = []
    idx = 0
    for L in ep_lengths:
        ep_vx.append(float(np.mean(base_vel[idx:idx+L, 0])))
        idx += L
    
    results = {
        "mode": label,
        "action_rate_rms": action_rate_rms,
        "mean_vx": mean_vx,
        "mean_abs_vy": mean_abs_vy,
        "mean_abs_vz": mean_abs_vz,
        "action_magnitude": action_mag,
        "ep_mean_vx": ep_vx,
    }
    return results

def main():
    print(f"Loading data from {DATA}")
    d = np.load(DATA)
    obs = d["obs"]
    raw_actions = d["actions"]
    base_vel = d["base_vel"]
    joint_vel = d["joint_vel"]
    ep_lengths = d["ep_lengths"]
    print(f"  samples={len(obs)}, obs_dim={obs.shape[1]}, episodes={len(ep_lengths)}")
    print(f"  ep_lengths={list(ep_lengths)}")

    # Load flow model
    print(f"Loading flow model from {MODEL}")
    ckpt = torch.load(MODEL, map_location=DEVICE, weights_only=True)
    state_dim = ckpt.get("state_dim", 235)
    model = VelocityNet(state_dim=state_dim).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  state_dim={state_dim}")
    
    # Apply flow smoothing
    raw_t = torch.tensor(raw_actions, dtype=torch.float32, device=DEVICE)
    obs_t = torch.tensor(obs[:, :state_dim], dtype=torch.float32, device=DEVICE)
    print("Running flow smoothing...")
    for t_end_val in [0.3, 0.5, 1.0]:
        smo_test = flow_smooth_batch(model, raw_t, obs_t, t_end=t_end_val).cpu().numpy()
        ar_test = np.diff(smo_test, axis=0)
        rms_test = float(np.sqrt(np.mean(ar_test**2)))
        red = (1 - rms_test / float(np.sqrt(np.mean(np.diff(raw_actions, axis=0)**2)))) * 100
        print(f"  t_end={t_end_val:.1f}  action_rate_rms={rms_test:.4f}  reduction={red:.1f}%")
    
    T_END = 0.5  # Use best t_end for final results
    smo_actions = flow_smooth_batch(model, raw_t, obs_t, t_end=T_END).cpu().numpy()
    
    # Compute metrics for both
    print("\n--- RAW PPO ---")
    raw_results = compute_metrics(raw_actions, base_vel, ep_lengths, "raw")
    for k, v in raw_results.items():
        if not isinstance(v, list):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n--- FLOW-SMOOTHED ---")
    smo_results = compute_metrics(smo_actions, base_vel, ep_lengths, "smoothed")
    for k, v in smo_results.items():
        if not isinstance(v, list):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Reduction
    reduction = (1 - smo_results["action_rate_rms"] / raw_results["action_rate_rms"]) * 100
    print(f"\n  Action jitter reduction: {reduction:.1f}%")
    
    # Save results
    all_results = {"raw": raw_results, "smoothed": smo_results, "jitter_reduction_pct": reduction}
    json_path = os.path.join(BASE, "rough_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved -> {json_path}")
    
    # Save per-step data for plotting
    np.savez(os.path.join(BASE, "rough_eval_raw.npz"),
             actions=raw_actions, vx=base_vel[:, 0], vy=base_vel[:, 1], vz=base_vel[:, 2],
             joint_vel=joint_vel, ep_lengths=ep_lengths)
    np.savez(os.path.join(BASE, "rough_eval_smo.npz"),
             actions=smo_actions, vx=base_vel[:, 0], vy=base_vel[:, 1], vz=base_vel[:, 2],
             joint_vel=joint_vel, ep_lengths=ep_lengths)
    print("Step data saved for plotting")

if __name__ == "__main__":
    main()
