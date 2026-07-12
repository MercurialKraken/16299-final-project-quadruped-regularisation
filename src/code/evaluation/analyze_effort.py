"""Paired analysis of effort eval: raw vs smoothed.
Metrics:
  action_rate_rms : RMS of (a_t - a_{t-1})/dt over time, then mean over joints
  jerk_rms        : RMS of (a_rate_t - a_rate_{t-1})/dt
  torque_rms      : RMS of applied torque per joint, summed
  torque_peak     : peak |torque| over episode
  power_mean      : mean of sum(|tau * q_dot|) over time
  energy          : integrated mech work = sum(|tau * q_dot|) * dt
"""
import numpy as np
from scipy import stats

R = np.load(r"A:\IsaacLab\effort_raw.npz")
S = np.load(r"A:\IsaacLab\effort_smoothed.npz")

dt = float(R["dt"])
N = int(R["num_envs"])
T = int(R["max_steps"])

def metrics(d):
    a = d["actions"]              # (T, N, 12)
    tau = d["torque"]             # (T, N, 12)
    qd = d["joint_vel"]           # (T, N, 12)
    # action rate
    a_rate = np.diff(a, axis=0) / dt  # (T-1, N, 12)
    # jerk = derivative of action rate
    jerk = np.diff(a_rate, axis=0) / dt  # (T-2, N, 12)
    # per-env metrics
    a_rate_rms = np.sqrt((a_rate ** 2).mean(axis=(0, 2)))  # (N,)
    jerk_rms   = np.sqrt((jerk ** 2).mean(axis=(0, 2)))
    tau_rms    = np.sqrt((tau ** 2).mean(axis=(0, 2)))
    tau_peak   = np.abs(tau).max(axis=(0, 2))
    power_inst = np.abs(tau * qd).sum(axis=2)  # (T, N)
    power_mean = power_inst.mean(axis=0)
    energy     = power_inst.sum(axis=0) * dt
    return dict(a_rate_rms=a_rate_rms, jerk_rms=jerk_rms, tau_rms=tau_rms,
                tau_peak=tau_peak, power_mean=power_mean, energy=energy)

mr = metrics(R)
ms = metrics(S)

def report(name, raw, smo):
    diff = smo - raw
    pct  = 100.0 * diff / raw
    n = len(raw)
    sem = diff.std(ddof=1) / np.sqrt(n)
    ci = 1.96 * sem
    t_stat, p_val = stats.ttest_rel(smo, raw)
    print(f"\n[{name}]")
    print(f"  raw : mean={raw.mean():.4f}  std={raw.std(ddof=1):.4f}")
    print(f"  smo : mean={smo.mean():.4f}  std={smo.std(ddof=1):.4f}")
    print(f"  diff (smo-raw): mean={diff.mean():+.4f}  95%CI=[{diff.mean()-ci:+.4f},{diff.mean()+ci:+.4f}]")
    print(f"  rel : {pct.mean():+.2f}%  (smoothed lower if negative)")
    print(f"  paired t={t_stat:.3f}  p={p_val:.4g}  wins(smo<raw)={(diff<0).sum()}/{n}")

print(f"=== Effort comparison: raw vs smoothed (n={N} paired envs, T={T} steps, dt={dt}s) ===")
report("action_rate_rms", mr["a_rate_rms"], ms["a_rate_rms"])
report("jerk_rms",        mr["jerk_rms"],   ms["jerk_rms"])
report("torque_rms",      mr["tau_rms"],    ms["tau_rms"])
report("torque_peak",     mr["tau_peak"],   ms["tau_peak"])
report("power_mean",      mr["power_mean"], ms["power_mean"])
report("energy_total",    mr["energy"],     ms["energy"])
