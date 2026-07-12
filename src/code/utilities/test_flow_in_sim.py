"""
Minimal test: can we load and call the flow model inside Isaac Sim's Python?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go1-v0")
parser.add_argument("--num_envs", type=int, default=1)
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
print("[TEST] torch loaded OK")
print(f"[TEST] CUDA available: {torch.cuda.is_available()}")

FLOW_MODEL = r"A:\IsaacLab\flow_model.pt"

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

try:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VelocityNet().to(dev)
    model.load_state_dict(torch.load(FLOW_MODEL, map_location=dev, weights_only=True))
    model.eval()
    print(f"[TEST] Flow model loaded OK on {dev}")
    
    # Test forward pass
    x = torch.randn(1, 12, device=dev)
    t = torch.tensor([[0.0]], device=dev)
    s = torch.randn(1, 48, device=dev)
    out = model(x, t, s)
    print(f"[TEST] Forward pass OK: output shape={out.shape}")
except Exception as e:
    print(f"[TEST] ERROR: {e}")
    import traceback
    traceback.print_exc()

print("[TEST] All done, closing sim")
simulation_app.close()
