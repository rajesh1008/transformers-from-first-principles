"""Chapter 38, 38.4 and 38.6: what fine-tuning changed in Tiny's tables.  Compares Chapter 24's checkpoint with the
checkpoint sft.py saved (default outputs/ckpt/sft_3e-4_600.pt; argument 1 to choose another).  For each kind of table:
relative change |W' - W| / |W| (Frobenius), averaged over layers; for every 128 x 128 and 128 x 512 matrix the share of
the change's squared size carried by its top 1, 4 and 8 singular directions (how low-rank the full fine-tune's update is);
and the arithmetic of low-rank adapters for Tiny (trainable numbers at rank r).  No training; CPU, 32-bit."""
from _common import *
import sys, torch, numpy as np, collections
A = torch.load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
B = torch.load(os.path.join(HERE, "outputs", "ckpt", sys.argv[1] if len(sys.argv) > 1 else "sft_3e-4_600.pt"))
groups = collections.defaultdict(list); top = collections.defaultdict(list); R = {}
for k in B:
    a, b = A[k].double(), B[k].double(); d = b - a
    kind = k.split(".", 1)[1].split(".", 1)[1] if k.startswith("h.") else k
    groups[kind].append(float(d.norm() / a.norm()))
    if d.dim() == 2 and k.startswith("h."):
        s = torch.linalg.svdvals(d) ** 2; s = s / s.sum()
        top[kind].append([float(s[:1].sum()), float(s[:4].sum()), float(s[:8].sum())])
print("relative change |W'-W|/|W|, mean over layers   (share of the change in its top 1 / 4 / 8 directions)")
for kind, v in groups.items():
    t = np.mean(top[kind], 0) if kind in top else None
    R[kind] = dict(rel=float(np.mean(v)), top=None if t is None else [float(x) for x in t])
    print(f"  {kind:12s} {np.mean(v):.4f}" + ("" if t is None else f"    {t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f}"), flush=True)
tot = sum(v.numel() for v in A.values())
for r in (1, 4, 8, 16):
    att = 12 * 4 * r * (128 + 128); ffn = 12 * 2 * r * (128 + 512)
    print(f"adapters at rank {r:2d}: attention maps {att:,} + feed-forward maps {ffn:,} = {att + ffn:,} trainable numbers, "
          f"{100 * (att + ffn) / tot:.1f} per cent of Tiny's {tot:,}", flush=True)
    R[f"lora_r{r}"] = att + ffn
save_json("weights" + ("" if len(sys.argv) < 2 else "_" + sys.argv[1].replace(".pt", "")), R)
