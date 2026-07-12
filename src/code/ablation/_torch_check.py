"""Quick env check: write torch + cuda info to a file."""
import sys
import os

out = r"A:\AllIsaac\flow_matching_project\_torch_check.txt"
os.makedirs(os.path.dirname(out), exist_ok=True)

lines = [f"sys={sys.version}"]
try:
    import torch
    lines.append(f"torch={torch.__version__}")
    lines.append(f"cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"gpu={torch.cuda.get_device_name(0)}")
        lines.append(f"vram_total_gb={round(torch.cuda.get_device_properties(0).total_memory/1e9,2)}")
        lines.append(f"vram_free_gb={round(torch.cuda.mem_get_info(0)[0]/1e9,2)}")
except Exception as e:
    lines.append(f"torch_error={e!r}")

try:
    import isaaclab
    lines.append(f"isaaclab={isaaclab.__version__ if hasattr(isaaclab,'__version__') else 'imported_ok'}")
except Exception as e:
    lines.append(f"isaaclab_error={e!r}")

open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
