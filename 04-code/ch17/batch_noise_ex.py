"""Exercise 17.11 (added 16 November 2026): batch_noise.py extended to batches of 4 and 16 windows.  The reference nudge is
drawn exactly as in batch_noise.py (the first 128 window starts from seed 1865), so it is the same reference; the batches are
eight fresh draws per size from the same generator.  32-bit, eight threads; seed 1865.  -> outputs/batch_noise_ex.txt/.json"""
import _common
import math
import numpy as np
import torch
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from _common import novel_ids, save_json, SEED
torch.set_num_threads(8)
nov = np.array(novel_ids())
rng = np.random.default_rng(SEED)
tt = TorchTiny.from_numpy(Tiny(), dtype=torch.float32)
def flat_grad(starts):
    ids = torch.tensor(np.stack([nov[a:a + 129] for a in starts]))
    acc = None
    for chunk in torch.split(ids, 16):
        L, G = tt.grads(chunk)
        v = np.concatenate([g.ravel() for g in G.values()]) * len(chunk)
        acc = v if acc is None else acc + v
    return acc / len(starts)
ref = flat_grad(rng.integers(0, len(nov) - 129, size=128))
print(f"reference nudge: 128 windows, 16,384 predictions; length {np.linalg.norm(ref):.4f}")
r = 1 / 0.407 ** 2 - 1; out = {}
for b in (4, 16):
    cos = []
    for _ in range(8):
        g = flat_grad(rng.integers(0, len(nov) - 129, size=b)); cos.append(float(g @ ref / np.linalg.norm(g) / np.linalg.norm(ref)))
    out[b] = cos
    print(f"batch of {b:>2} windows ({128*b:>5,} predictions): cosine with the reference {np.mean(cos):.3f} (lowest {min(cos):.3f},"
          f" highest {max(cos):.3f}); predicted from one window's 0.407: {1 / math.sqrt(1 + r / b):.3f}")
save_json("batch_noise_ex", out)
