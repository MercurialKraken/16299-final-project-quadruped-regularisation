"""Paired push-recovery analysis."""
import numpy as np
from scipy import stats
from collections import OrderedDict

R = np.load(r"A:\IsaacLab\push2_raw.npz")
S = np.load(r"A:\IsaacLab\push2_smoothed.npz")

mags_r = R["magnitudes"]
mags_s = S["magnitudes"]
fell_r = R["fell"]
fell_s = S["fell"]
fs_r   = R["fall_step"]
fs_s   = S["fall_step"]

assert np.allclose(mags_r, mags_s), "magnitude assignments differ"

unique_mags = sorted(set(mags_r.tolist()))
print(f"=== Push recovery: lateral impulse on base, dur={int(R['push_dur'])}*dt={int(R['push_dur'])*float(R['dt']):.2f}s ===")
print(f"{'F (N)':>7}  {'n':>3}  {'raw_fell':>9}  {'smo_fell':>9}  {'rate_raw':>9}  {'rate_smo':>9}  {'delta':>7}")
total_raw_fell = 0
total_smo_fell = 0
n_total = 0
for m in unique_mags:
    idx = np.isclose(mags_r, m)
    n = int(idx.sum())
    rf = int(fell_r[idx].sum())
    sf = int(fell_s[idx].sum())
    print(f"{m:>7.0f}  {n:>3}  {rf:>9}  {sf:>9}  {rf/n:>8.0%}  {sf/n:>8.0%}  {(sf-rf):>+7d}")
    total_raw_fell += rf
    total_smo_fell += sf
    n_total += n

print(f"\nTOTAL: raw fell {total_raw_fell}/{n_total} ({total_raw_fell/n_total:.1%})  "
      f"smo fell {total_smo_fell}/{n_total} ({total_smo_fell/n_total:.1%})")

# McNemar test (paired binary outcomes)
b = int(((fell_r) & (~fell_s)).sum())  # raw fell, smo OK -> smoothed wins
c = int(((~fell_r) & (fell_s)).sum())  # raw OK, smo fell  -> raw wins
both = int((fell_r & fell_s).sum())
neither = int((~fell_r & ~fell_s).sum())
print(f"\nDiscordant pairs:  raw_fell&smo_ok = {b}   raw_ok&smo_fell = {c}")
print(f"Concordant:        both fell = {both}   neither fell = {neither}")
if b + c > 0:
    # McNemar's test (binomial exact)
    from scipy.stats import binomtest
    p = binomtest(min(b, c), b + c, p=0.5, alternative='two-sided').pvalue
    print(f"McNemar p-value (two-sided): {p:.4g}")
else:
    print("All pairs concordant; McNemar inapplicable")

# Time-to-fall comparison (only for envs that fell)
fell_both = fell_r & fell_s
print(f"\nEnvs where both fell: {int(fell_both.sum())}")
if fell_both.any():
    dt_steps = (fs_s[fell_both] - fs_r[fell_both])
    print(f"  fall_step delta (smo - raw): mean={dt_steps.mean():.2f} steps  "
          f"({dt_steps.mean()*float(R['dt'])*1000:.1f} ms)  positive => smoothed survived longer")
