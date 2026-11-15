"""Chapter 17, section 17.9: the height of the plateau every run in train_tiny.py sat on.  A model that ignores
context and gives every seat the same shares can do no better than the piece frequencies of the novel.
This prints that best context-free loss, on the novel itself and on the held-out text (add-one counts)."""
import _common
import json
import numpy as np
from _common import novel_ids, heldout_ids, tok, load_json
T = tok(); nov = np.array(novel_ids(T)); held = np.array(heldout_ids(T))
c = np.bincount(nov, minlength=2048).astype(float)
p = c / c.sum(); q = (c + 1) / (c + 1).sum()
on_novel = float(-(p[p > 0] * np.log(p[p > 0])).sum())
on_held = float(-np.log(q[held]).mean())
print(f"best context-free loss on the novel (its own piece frequencies): {on_novel:.4f}")
print(f"the same frequencies (add-one) on the held-out text: {on_held:.4f}")
top = np.argsort(-p)[:3]
print("the three commonest pieces: " + ", ".join(f"{T.decode([int(i)])!r} {p[i]:.3f}" for i in top))
d = load_json("train_tiny")
for lr, v in d.items():
    tr = v["train"]
    print(f"lr {lr}: highest batch loss {max(tr):.4g} at step {tr.index(max(tr))}; held-out at steps 100, 300, 599: "
          + ", ".join(f"{c[3]:.3f}" for c in v["curve"] if c[0] in (100, 300, 599)))
# the same 16 + 16 evaluation windows as train_tiny.py, before any step
import torch
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
rng = np.random.default_rng(1865); rng.integers(0, len(nov) - 129, size=(600, 8))
hs = rng.integers(0, len(held) - 129, size=16); ns = rng.integers(0, len(nov) - 129, size=16)
tt = TorchTiny.from_numpy(Tiny(), dtype=torch.float32)
with torch.no_grad():
    a = float(tt.loss(torch.tensor(np.stack([nov[s:s + 129] for s in ns]))))
    b = float(tt.loss(torch.tensor(np.stack([held[s:s + 129] for s in hs]))))
print(f"untrained Tiny on the same evaluation windows: novel {a:.4f}, held-out {b:.4f} (ln 2048 = {np.log(2048):.4f})")
