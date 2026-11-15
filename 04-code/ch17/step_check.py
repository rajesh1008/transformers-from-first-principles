"""Chapter 17, section 17.8: how far one step can be trusted.  The nudges predict that a step of learning rate
lr lowers the loss by about lr x (sum of the squared nudges).  Compare the prediction with what one real step
does, on the hand Tiny (running sentence, 64-bit) and on the untrained Tiny (one batch of eight windows from the
novel, seed 1865, 32-bit, eight threads)."""
import _common
import numpy as np
import torch
from hand import hand_tiny, IDS
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from _common import novel_ids, SEED
torch.set_num_threads(8)

def check(make, ids, lrs, name):
    tt = make(); L0, G = tt.grads(ids)
    g2 = sum(float((g ** 2).sum()) for g in G.values())
    print(f"{name}: loss {L0:.4f}; sum of squared nudges {g2:.4f}")
    for lr in lrs:
        tt = make()
        with torch.no_grad():
            for k, v in tt.t.items():
                v -= lr * torch.as_tensor(G[k], dtype=v.dtype)
            L1 = float(tt.loss(ids))
        print(f"  lr {lr:<5}: predicted change {-lr * g2:+.4f}  actual change {L1 - L0:+.4f}  actual / predicted {(L1 - L0) / (-lr * g2):.3f}")

check(lambda: TorchTiny.from_numpy(hand_tiny()), IDS, (0.01, 0.1, 0.5, 2.0, 8.0), "hand Tiny, running sentence")
nov = np.array(novel_ids()); rng = np.random.default_rng(SEED)
batch = torch.tensor(np.stack([nov[a:a + 129] for a in rng.integers(0, len(nov) - 129, size=8)]))
T0 = Tiny()
check(lambda: TorchTiny.from_numpy(T0, dtype=torch.float32), batch, (0.01, 0.1, 0.3, 1.0, 3.0), "Tiny, one batch of 8 windows")
