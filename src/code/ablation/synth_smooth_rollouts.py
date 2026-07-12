"""Generate rollout npz files for the smoothed variants by applying the
smoother offline to a base rollout. Honest for spectral comparison; not for
closed-loop fall-rate.

Outputs:
    rollout_noreg_lp.npz
    rollout_noreg_flow.npz
    rollout_noreg_flow_lp.npz
    rollout_somereg_lp.npz   (re-create from fresh rollout_somereg.npz)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt


ABL_DIR = r"A:\AllIsaac\flow_matching_project\data\ablation"
FLOW_PATH = r"A:\AllIsaac\IsaacLab\flow_model_balanced_lp_noreg.pt"


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
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x_t, t, state], dim=-1))


def offline_lp(actions, fc=15.0, fs=50.0, order=2):
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, actions, axis=0).astype(np.float32)


@torch.no_grad()
def offline_flow(actions, obs, model, n_steps=20, t_end=1.0,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
    a = torch.from_numpy(actions).to(device)
    s = torch.from_numpy(obs).to(device)
    out = a.clone()
    dt = t_end / n_steps
    for i in range(n_steps):
        t_val = torch.full((out.shape[0], 1), i * dt, device=device)
        out = out + model(out, t_val, s) * dt
    return out.cpu().numpy().astype(np.float32)


def load_flow_path(p):
    ckpt = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        sdim = ckpt.get("state_dim", 48)
    else:
        sd = ckpt
        sdim = 48
    m = VelocityNet(state_dim=sdim)
    m.load_state_dict(sd)
    m.eval()
    if torch.cuda.is_available():
        m = m.cuda()
    return m


def load_flow():
    return load_flow_path(FLOW_PATH)


def derive(in_npz, out_npz, mode, model=None):
    d = dict(np.load(in_npz))
    raw = d["actions"]
    obs = d["obs"]

    if mode == "lp":
        smoothed = offline_lp(raw)
    elif mode == "flow":
        smoothed = offline_flow(raw, obs, model)
    elif mode == "flow_lp":
        flow_out = offline_flow(raw, obs, model)
        smoothed = offline_lp(flow_out)
    else:
        raise ValueError(mode)

    d["actions_orig"] = raw
    d["actions"] = smoothed
    d["smoother_mode"] = mode
    np.savez(out_npz, **d)
    print(f"[derive] {mode}: {os.path.basename(in_npz)} -> {os.path.basename(out_npz)}")


def main():
    flow = load_flow()
    base_noreg = os.path.join(ABL_DIR, "rollout_noreg.npz")
    base_somereg = os.path.join(ABL_DIR, "rollout_somereg.npz")

    derive(base_noreg, os.path.join(ABL_DIR, "rollout_noreg_lp.npz"), "lp")
    derive(base_noreg, os.path.join(ABL_DIR, "rollout_noreg_flow.npz"), "flow", flow)
    derive(base_noreg, os.path.join(ABL_DIR, "rollout_noreg_flow_lp.npz"), "flow_lp", flow)
    derive(base_somereg, os.path.join(ABL_DIR, "rollout_somereg_lp.npz"), "lp")

    # somereg + Bal-LP flow (using the original somereg-trained flow)
    flow_somereg = load_flow_path(r"A:\AllIsaac\IsaacLab\flow_model_balanced_lp.pt")
    derive(base_somereg, os.path.join(ABL_DIR, "rollout_somereg_flow.npz"), "flow", flow_somereg)

    # extremereg + Bal-LP flow (using the new extremereg-trained flow)
    base_extremereg = os.path.join(ABL_DIR, "rollout_extremereg.npz")
    flow_xreg = load_flow_path(r"A:\AllIsaac\IsaacLab\flow_model_balanced_lp_extremereg.pt")
    derive(base_extremereg, os.path.join(ABL_DIR, "rollout_extremereg_flow.npz"), "flow", flow_xreg)


if __name__ == "__main__":
    main()
