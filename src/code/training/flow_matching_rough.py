"""Adaptive flow matching for rough terrain (235-dim obs)."""
import argparse, os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = r"A:\AllIsaac\IsaacLab\rollout_data_rough.npz"
MODEL_PATH = r"A:\AllIsaac\IsaacLab\flow_model_rough.pt"

JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Per-joint adaptive cutoffs (Hz). Same strategy as flat but slightly more
# conservative for rough because the actions carry more legitimate
# high-frequency terrain-response signal.
JOINT_CUTOFFS = {
    "FL_hip":    12.0, "FL_thigh":  14.0, "FL_calf":   22.0,
    "FR_hip":    10.0, "FR_thigh":  15.0, "FR_calf":   10.0,
    "RL_hip":    22.0, "RL_thigh":  12.0, "RL_calf":   15.0,
    "RR_hip":    14.0, "RR_thigh":  25.0, "RR_calf":   12.0,
}

def get_cutoffs():
    return np.array([JOINT_CUTOFFS[n] for n in JOINT_NAMES])


def butter_lp_adaptive(data, cutoffs, fs=50.0, order=2):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    filtered = np.zeros_like(data)
    for j in range(data.shape[1]):
        nc = min(max(cutoffs[j]/nyq, 0.01), 0.99)
        b, a = butter(order, nc, btype="low", analog=False)
        filtered[:, j] = filtfilt(b, a, data[:, j])
    return filtered


class RoughDataset(Dataset):
    def __init__(self, data_path):
        d = np.load(data_path)
        self.obs = d["obs"].astype(np.float32)
        raw = d["actions"].astype(np.float32)
        ep_lens = d["ep_lengths"]
        smooth = np.zeros_like(raw)
        cutoffs = get_cutoffs()
        idx = 0
        for L in ep_lens:
            if L > 12:
                smooth[idx:idx+L] = butter_lp_adaptive(raw[idx:idx+L], cutoffs)
            else:
                smooth[idx:idx+L] = raw[idx:idx+L]
            idx += L
        self.x0 = raw
        self.x1 = smooth
        self.state_dim = self.obs.shape[1]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return (torch.tensor(self.obs[i]),
                torch.tensor(self.x0[i]),
                torch.tensor(self.x1[i]))


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


def train(epochs=200, batch_size=512, lr=1e-3):
    print(f"[flow-rough] Loading {DATA_PATH}", flush=True)
    ds = RoughDataset(DATA_PATH)
    state_dim = ds.state_dim
    print(f"[flow-rough] state_dim={state_dim}, samples={len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = VelocityNet(state_dim=state_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses = []
    for ep in range(epochs):
        tot = 0.0
        for state, x0, x1 in loader:
            state, x0, x1 = state.to(DEVICE), x0.to(DEVICE), x1.to(DEVICE)
            t = torch.rand(x0.shape[0], 1, device=DEVICE)
            x_t = (1-t)*x0 + t*x1
            target_v = x1 - x0
            pred_v = model(x_t, t, state)
            loss = nn.functional.mse_loss(pred_v, target_v)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        avg = tot / len(loader)
        losses.append(avg)
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}/{epochs} | loss={avg:.6f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "state_dim": state_dim}, MODEL_PATH)
    np.save(r"A:\AllIsaac\IsaacLab\rough_training_loss.npy", np.array(losses))
    print(f"[flow-rough] Saved {MODEL_PATH}", flush=True)


@torch.no_grad()
def flow_smooth(model, raw, state, n_steps=20, t_end=1.0):
    model.eval()
    x = raw.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = torch.full((x.shape[0], 1), i*dt, device=x.device)
        x = x + model(x, t_val, state) * dt
    return x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()
    if args.train:
        train(epochs=args.epochs)
