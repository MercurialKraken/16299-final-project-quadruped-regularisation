"""Find discordant envs from push2 results: raw fell + smoothed survived."""
import numpy as np

raw = np.load(r"A:\AllIsaac\IsaacLab\push2_raw.npz")
smo = np.load(r"A:\AllIsaac\IsaacLab\push2_smoothed.npz")

rf = raw["fell"]
sf = smo["fell"]
mags = raw["magnitudes"]
rfs = raw["fall_step"]
sfs = smo["fall_step"]

# Discordant: raw fell, smoothed OK
b_mask = rf & ~sf
b_idx = np.where(b_mask)[0]
print(f"=== Discordant b (raw fell, smo OK): {len(b_idx)} envs ===")
for i in b_idx:
    print(f"  env {i:3d}  mag={mags[i]:.0f} N  raw_fall_step={rfs[i]}")

# Discordant: smoothed fell, raw OK
c_mask = sf & ~rf
c_idx = np.where(c_mask)[0]
print(f"\n=== Discordant c (smo fell, raw OK): {len(c_idx)} envs ===")
for i in c_idx:
    print(f"  env {i:3d}  mag={mags[i]:.0f} N  smo_fall_step={sfs[i]}")

# Print magnitude distribution of b-type discordants
print(f"\n=== Magnitude distribution of b-type (raw fell, smo OK) ===")
for m in sorted(set(mags[b_idx])):
    n = np.sum(mags[b_idx] == m)
    print(f"  {m:.0f} N: {n} envs  ->  indices: {list(np.where(b_mask & (mags == m))[0])}")
