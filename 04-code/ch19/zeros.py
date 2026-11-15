"""Chapter 19, section 19.1: starting from zeros, and from one repeated number.
Part 1: every learned number of Tiny set to 0 ("all_zeros").  Part 2: only the layers' tables set to 0.
Part 3: every layer table filled with the single number 0.02 ("constant"), then 20 steps of plain gradient
descent, counting how many of layer 0's 512 feed-forward neurons are still different from each other.
64-bit throughout so that "exactly zero" and "exactly equal" mean what they say."""
import _common
import numpy as np, torch
from tiny.model import Tiny
from tiny.init import initialise
from tiny.torch_tiny import TorchTiny
from tiny.train import train_torch
from _common import sentence_ids, novel_ids, tok, save_json, SEED, probe
torch.set_num_threads(1)
T = tok(); ids = sentence_ids(T)
out = {}
print(f"The running sentence, Tiny ids {ids}; uniform guess ln 2048 = {np.log(2048):.4f}")

print("\nPart 1: every learned number 0")
m = initialise(Tiny(), "all_zeros")
tt = TorchTiny.from_numpy(m)
L, G = tt.grads(ids)
nz = {k: int((v != 0).sum()) for k, v in G.items()}
print(f"  loss {L:.4f}; tables with any nonzero nudge: {sum(1 for v in nz.values() if v)} of {len(G)};"
      f" largest nudge anywhere {max(np.abs(v).max() for v in G.values()):.1f}")
out["all_zeros"] = dict(loss=L, nonzero_tables=[k for k, v in nz.items() if v])

print("\nPart 2: token and position tables as drawn, every layer table 0")
m = initialise(Tiny(), "zeros")
tt = TorchTiny.from_numpy(m)
L, G = tt.grads(ids)
moving = [k for k, v in G.items() if np.abs(v).max() > 0]
still = [k for k, v in G.items() if np.abs(v).max() == 0]
print(f"  loss {L:.4f}; tables that receive a nonzero nudge: {len(moving)} of {len(G)}")
kinds = sorted({k.split('.', 2)[-1] if k.startswith('h.') else k for k in moving})
print(f"  which kinds move: {', '.join(kinds)}")
kinds_still = sorted({k.split('.', 2)[-1] for k in still if k.startswith('h.')})
print(f"  which kinds get exactly 0 in every layer: {', '.join(kinds_still)}")
nov = np.array(novel_ids(T))
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(20, 2))
batches = lambda s: torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
tt = TorchTiny.from_numpy(m)
train_torch(tt, batches, 20, 0.6)
w = {k: float(tt.t[k].detach().abs().max()) for k in tt.t if k.startswith("h.") and (".W_" in k)}
print(f"  after 20 steps at learning rate 0.6: largest |number| in any W_q, W_k, W_v, W_o, W_1, W_2 of any layer: {max(w.values()):.1f}")
print(f"  largest |b_2| {max(float(tt.t[f'h.{l}.ff.b_2'].detach().abs().max()) for l in range(12)):.4f} (the bias still moves)")
out["zeros"] = dict(loss=L, moving=moving, still_kinds=kinds_still, max_W_after=max(w.values()))

print("\nPart 3: every layer table filled with 0.02")
m = initialise(Tiny(), "constant")
tt = TorchTiny.from_numpy(m)
def distinct(W, tol=1e-9):
    cols = W.T
    keep = []
    for c in cols:
        if not any(np.abs(c - k).max() <= tol * max(1.0, np.abs(k).max()) for k in keep):
            keep.append(c)
    return len(keep)
W1 = tt.t["h.0.ff.W_1"].detach().numpy()
P0 = probe(m, ids)
print(f"  what the branches add at layer 0: attention {P0['att'][0]:.3g}, feed-forward {P0['ff'][0]:.3g}"
      f" (a constant table times a normalised row, whose numbers sum to 0)")
print(f"  before training: distinct neurons in layer 0's W_1: {distinct(W1)} of {W1.shape[1]}")
hist = train_torch(tt, batches, 20, 0.6)
W1 = tt.t["h.0.ff.W_1"].detach().numpy()
Wq = tt.t["h.0.att.W_q"].detach().numpy()
heads = [Wq[:, 32 * i:32 * (i + 1)] for i in range(4)]
print(f"  after 20 steps (loss {hist[0][1]:.4f} -> {hist[-1][1]:.4f}): distinct neurons in W_1: {distinct(W1)} of 512;"
      f" distinct columns in W_q: {distinct(Wq)} of 128")
print(f"  largest difference between head 0's and head 3's query tables: {np.abs(heads[0] - heads[3]).max():.3g}")
print(f"  the 512 neurons' weights moved by {np.abs(W1 - 0.02).max():.3g} at most")
out["constant"] = dict(distinct_W1=distinct(W1), distinct_Wq=distinct(Wq), loss=[hist[0][1], hist[-1][1]])
save_json("zeros", out)
