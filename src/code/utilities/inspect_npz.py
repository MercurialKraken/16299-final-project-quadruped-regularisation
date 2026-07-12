import numpy as np
r = np.load(r"A:\IsaacLab\ab_race_raw.npz")
s = np.load(r"A:\IsaacLab\ab_race_smoothed.npz")
for name, z in [("raw", r), ("sm", s)]:
    print(name)
    for k in z.files:
        print(f"  {k}: shape={z[k].shape} dtype={z[k].dtype}")
