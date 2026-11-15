"""Chapter 17, section 17.11: the nudge from a batch is an estimate of the nudge from the whole text.
On the untrained Tiny, compute the gradient over 128 windows of 129 tokens from the novel (16,384 predictions,
the 'reference'), then over batches of 1, 2, 8 and 32 windows drawn from other places, and compare each with
the reference by its length and by the cosine of the angle between them (1 = same direction).
32-bit, eight threads (torch.set_num_threads(8)); seed 1865."""
import _common
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
out = {}
for b in (1, 2, 8, 32):
    cos, ln = [], []
    for _ in range(8):
        g = flat_grad(rng.integers(0, len(nov) - 129, size=b))
        cos.append(float(g @ ref / np.linalg.norm(g) / np.linalg.norm(ref))); ln.append(float(np.linalg.norm(g)))
    out[b] = dict(cos=cos, len=ln)
    print(f"batch of {b:>2} windows ({128*b:>5,} predictions): cosine with the reference {np.mean(cos):.3f}"
          f" (lowest {min(cos):.3f}, highest {max(cos):.3f}); length {np.mean(ln):.4f}")
save_json("batch_noise", out)
