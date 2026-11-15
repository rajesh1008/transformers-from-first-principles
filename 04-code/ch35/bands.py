"""Chapter 35, 35.3: finer bands from the per-token losses that smol_long.py saved (outputs/smol_long_MODE_16384.pt):
mean loss in bands of 1,024 positions from 6,144 to 16,383, for every mode that finished."""
from _common import *
import torch, glob
R = {}
for f in sorted(glob.glob(os.path.join(HERE, "outputs", "smol_long_*_16384.pt"))):
    mode = os.path.basename(f).split("_")[2]; L = torch.load(f)
    row = {f"{a}-{a + 1023}": float(L[a:a + 1024].mean()) for a in range(6144, 16384, 1024)}
    R[mode] = row
    print(f"{mode:6s} " + " ".join(f"{v:.3f}" for v in row.values()))
print("bands start at", ", ".join(str(a) for a in range(6144, 16384, 1024)))
save_json("smol_long_bands", R)
