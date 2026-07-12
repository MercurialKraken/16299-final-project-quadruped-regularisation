import numpy as np
d = np.load(r"A:\IsaacLab\rollout_data.npz")
print("Keys:", list(d.keys()))
for k in d.keys():
    print(f"  {k}: {d[k].shape}")
ar = np.diff(d["actions"], axis=0)
jacc = d["joint_acc"]
print(f"\n-- Baseline Metrics --")
print(f"  Total timesteps  : {d['obs'].shape[0]}")
print(f"  action_rate_rms  : {np.sqrt(np.mean(ar**2)):.4f}")
print(f"  joint_acc_rms    : {np.sqrt(np.mean(jacc[1:]**2)):.4f}")
print(f"  mean_fwd_vel     : {np.mean(d['base_vel'][:,0]):.4f}")
print(f"  ep_lengths       : {d['ep_lengths']}")
print(f"  ep_returns       : {np.round(d['ep_returns'], 1)}")
