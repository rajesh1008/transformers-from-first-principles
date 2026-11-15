"""Chapter 17, section 17.7: checking autograd's nudges on the real Tiny (2,766,592 numbers) by wiggling a
handful of them, one from each kind of table, on the running sentence (seven tokens, six predictions).
Wiggling uses the NumPy Tiny in 64-bit, h = 1e-6, central difference.  Also: how far 32-bit autograd is
from 64-bit.  Seed 1865 picks the numbers to check."""
import _common
import numpy as np
import torch
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from _common import sentence_ids, save_json, SEED
torch.set_num_threads(1)
m = Tiny()
ids = sentence_ids()
L, G = TorchTiny.from_numpy(m).grads(ids)
L32, G32 = TorchTiny.from_numpy(m, dtype=torch.float32).grads(ids)
print(f"Tiny, running sentence {ids}: loss {L:.6f} (NumPy {m.loss(ids):.6f}); {sum(v.size for v in G.values()):,} nudges from one backward pass")
rng = np.random.default_rng(SEED)
tables = m.tables()
names = ["E", "P", "h.0.ln_1.gain", "h.0.att.W_q", "h.0.att.W_k", "h.0.att.W_v", "h.0.att.W_o", "h.0.ff.W_1", "h.0.ff.b_1",
         "h.6.ff.W_2", "h.11.ln_2.bias", "ln_f.gain"]
rows = []
for name in names:
    T = tables[name]
    if name == "E":
        idx = (ids[3], int(rng.integers(T.shape[1])))           # a token of the sentence, so the input path is used too
    elif name == "P":
        idx = (2, int(rng.integers(T.shape[1])))
    else:
        idx = tuple(int(rng.integers(s)) for s in T.shape)
    old = T[idx]; h = 1e-6
    T[idx] = old + h; up = m.loss(ids); T[idx] = old - h; dn = m.loss(ids); T[idx] = old
    w = (up - dn) / (2 * h); a = G[name][idx]
    rel = abs(w - a) / max(abs(a), 1e-12)
    rows.append(dict(name=name, idx=list(idx), autograd=float(a), wiggle=float(w), rel=float(rel), a32=float(G32[name][idx])))
    print(f"  {name:<15} {str(list(idx)):<11} autograd {a:+.6e}  wiggled {w:+.6e}  relative difference {rel:.1e}  32-bit {G32[name][idx]:+.6e}")
print(f"largest relative difference: {max(r['rel'] for r in rows):.1e}")
d32 = max(float(np.abs(G[k] - G32[k]).max()) for k in G)
big = max(float(np.abs(G[k]).max()) for k in G)
print(f"32-bit autograd against 64-bit: largest difference {d32:.2e}, largest nudge {big:.3e}")
untouched = int(sum((np.abs(G['E'][i]).max() == 0) for i in range(G['E'].shape[0])))
print(f"token rows with a nudge of exactly zero: {untouched} of 2,048 (tied output: every row is scored at every seat, so none is zero)")
zp = int((np.abs(G['P']).max(1) == 0).sum())
print(f"position rows with a nudge of exactly zero: {zp} of 1,024 (only seats 0 to 5 are used as inputs for the six predictions)")
save_json("tiny_check", dict(loss=L, rows=rows, d32=d32, zero_E=untouched, zero_P=zp))
