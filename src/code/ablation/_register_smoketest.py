"""Quick check: import register_reg_variants and confirm new task IDs exist."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import register_reg_variants  # noqa: F401

want = [
    "Isaac-Velocity-Flat-Unitree-Go1-v0",
    "Isaac-Velocity-Flat-Unitree-Go1-NoReg-v0",
    "Isaac-Velocity-Flat-Unitree-Go1-ExtremeReg-v0",
]
for tid in want:
    print(f"{tid}: {'OK' if tid in gym.envs.registry else 'MISSING'}")
