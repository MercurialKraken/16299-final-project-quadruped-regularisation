"""Paired analysis of multiseed eval: raw vs smoothed at same seed.
Since envs spawn with random yaw, world-x progress is misleading for the
aggregate. We use total distance traveled (||final - start||) which is paired
across the two runs (same env_idx -> same spawn pose & orientation).
"""
import numpy as np
from pathlib import Path
from scipy import stats

RAW = np.load(r"A:\IsaacLab\multiseed_raw.npz")
SMO = np.load(r"A:\IsaacLab\multiseed_smoothed.npz")

print("--- Sanity check: shared spawn? ---")
print("raw start_pos[:3]:\n", RAW["start_pos"][:3])
print("smo start_pos[:3]:\n", SMO["start_pos"][:3])
same_spawn = np.allclose(RAW["start_pos"], SMO["start_pos"])
print("same spawn across runs?", same_spawn)

raw_disp  = RAW["final_pos"] - RAW["start_pos"]
smo_disp  = SMO["final_pos"] - SMO["start_pos"]
raw_dist  = np.linalg.norm(raw_disp, axis=1)
smo_dist  = np.linalg.norm(smo_disp, axis=1)

print("\n--- Per-env (paired) distance traveled ---")
print(f"{'idx':>3}  {'raw':>8}  {'smo':>8}  {'delta':>8}  {'pct':>7}")
for i, (r, s) in enumerate(zip(raw_dist, smo_dist)):
    d = s - r
    pct = 100.0 * d / max(r, 1e-6)
    print(f"{i:>3}  {r:>8.3f}  {s:>8.3f}  {d:>+8.3f}  {pct:>+6.1f}%")

print("\n--- Aggregate ---")
def stats_block(name, x):
    n = len(x)
    m = x.mean(); s = x.std(ddof=1)
    sem = s / np.sqrt(n)
    ci = 1.96 * sem
    print(f"{name:>10}: n={n}  mean={m:.3f}  std={s:.3f}  95%CI=[{m-ci:.3f}, {m+ci:.3f}]")
    return m, s, sem

mr, sr, _ = stats_block("raw_dist", raw_dist)
ms, ss, _ = stats_block("smo_dist", smo_dist)

diff = smo_dist - raw_dist
md, sd, _ = stats_block("smo - raw", diff)

print(f"\nfell raw: {RAW['fell'].sum()}/{len(raw_dist)}    fell smo: {SMO['fell'].sum()}/{len(smo_dist)}")

t_stat, p_val = stats.ttest_rel(smo_dist, raw_dist)
w_stat, w_p   = stats.wilcoxon(smo_dist, raw_dist)
print(f"\nPaired t-test  : t={t_stat:.3f}  p={p_val:.4f}")
print(f"Wilcoxon signed: W={w_stat:.3f}  p={w_p:.4f}")

mean_pct = 100.0 * (ms - mr) / mr
print(f"\nRelative gap (mean dist): {mean_pct:+.2f}%")
print(f"Wins for smoothed: {(diff > 0).sum()}/{len(diff)}")

# Also report along-heading projection if the spawn pose can be reconstructed.
# We don't have headings saved, so we approximate "intended forward" as the
# direction of the displacement vector for the *raw* run (same env, same
# initial heading): heading_hat ≈ raw_disp / ||raw_disp|| (only valid if raw
# walked far enough in the right direction).
norms = np.linalg.norm(raw_disp, axis=1, keepdims=True)
heading_hat = raw_disp / np.clip(norms, 1e-3, None)
raw_along = (raw_disp * heading_hat).sum(axis=1)
smo_along = (smo_disp * heading_hat).sum(axis=1)
print("\n--- Along-raw-heading projection (smoothed measured against raw's chosen direction) ---")
stats_block("raw_along", raw_along)
stats_block("smo_along", smo_along)
diff_a = smo_along - raw_along
stats_block("smo-raw_a", diff_a)
t_a, p_a = stats.ttest_rel(smo_along, raw_along)
print(f"Paired t (along): t={t_a:.3f}  p={p_a:.4f}")
print(f"Wins for smoothed (along): {(diff_a > 0).sum()}/{len(diff_a)}")
