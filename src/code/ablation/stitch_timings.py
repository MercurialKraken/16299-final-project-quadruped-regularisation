"""Copy timing.json from training log dirs into the ablation data dir."""
import json
import os
import shutil

ABL = r"A:\AllIsaac\flow_matching_project\data\ablation"
ROOTS = {
    "noreg": r"A:\AllIsaac\IsaacLab\logs\rsl_rl\unitree_go1_flat_noreg",
    "extremereg": r"A:\AllIsaac\IsaacLab\logs\rsl_rl\unitree_go1_flat_extremereg",
    "somereg": r"A:\AllIsaac\IsaacLab\logs\rsl_rl\unitree_go1_flat",
}

for v, root in ROOTS.items():
    if not os.path.isdir(root):
        print(f"[stitch] {v}: log root missing")
        continue
    runs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    runs.sort(key=os.path.getmtime, reverse=True)
    if not runs:
        print(f"[stitch] {v}: no run dirs")
        continue
    src = os.path.join(runs[0], "timing.json")
    dst = os.path.join(ABL, f"timing_{v}.json")
    if os.path.exists(src):
        shutil.copy(src, dst)
        with open(dst) as f:
            t = json.load(f)
        if isinstance(t, dict):
            wall = t.get("wall_total_s") or sum(i.get("wall_s", 0) for i in t.get("iters", []))
            print(f"[stitch] {v}: {wall:.1f}s -> {dst}")
    else:
        # fallback: synthesize from known runs
        if v == "somereg":
            # original training pre-instrumentation; mark as unknown
            with open(dst, "w") as f:
                json.dump({"variant": "somereg", "wall_total_s": None,
                           "iters_total": 300, "note": "pre-instrumentation"}, f)
            print(f"[stitch] {v}: synthesized stub")

# also copy timing for the smoothed variants from the corresponding base PPO
for derived, base in [("noreg_flow", "noreg"), ("noreg_lp", "noreg"),
                      ("noreg_flow_lp", "noreg"), ("somereg_lp", "somereg"),
                      ("somereg_flow", "somereg"), ("extremereg_flow", "extremereg")]:
    src = os.path.join(ABL, f"timing_{base}.json")
    dst = os.path.join(ABL, f"timing_{derived}.json")
    if os.path.exists(src):
        shutil.copy(src, dst)
