"""Chapter 20, section 20.5: how far each layer moves in one step, plain gradient descent against Adam.
Tiny's default start and the start ten times too large; 20 steps on Chapter 19's windows (seed 1865), then the
size (root mean square) of step 21's change to the attention and feed-forward tables of each layer.  32-bit, 2 threads."""
import _common
import numpy as np
import torch
from tiny.model import Tiny
from tiny.init import initialise
from tiny.torch_tiny import TorchTiny
from tiny.optim import SGD, Adam, train_opt
from _common import novel_ids, tok, save_json, SEED
torch.set_num_threads(2)
nov = np.array(novel_ids(tok()))
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(400, 8))
batches = lambda s: torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
KINDS = ("att.W_q", "att.W_k", "att.W_v", "att.W_o", "ff.W_1", "ff.W_2")
out = {}
print("RMS of the change made by step 21 to each layer's six tables (and to E), after 20 steps")
for scale in (1.0, 10.0):
    for label, mk in (("SGD lr 0.1", lambda: SGD(0.1)), ("Adam lr 3e-3", lambda: Adam(3e-3))):
        tt = TorchTiny.from_numpy(initialise(Tiny(), "gpt2", scale), dtype=torch.float32)
        opt = mk()
        train_opt(tt, batches, 20, opt)
        before = {k: v.detach().clone() for k, v in tt.t.items()}
        train_opt(tt, batches, 1, opt, start=20)
        d = {k: float(((tt.t[k].detach() - before[k]) ** 2).mean().sqrt()) for k in tt.t}
        per = [float(np.sqrt(np.mean([d[f"h.{l}.{k}"] ** 2 for k in KINDS]))) for l in range(12)]
        out[f"{label} x{scale:g}"] = dict(layers=per, E=d["E"])
        print(f"  start x{scale:<3g} {label:13s} layer 0 {per[0]:.2e}  layer 6 {per[6]:.2e}  layer 11 {per[11]:.2e}"
              f"  bottom/top {per[0] / per[11]:6.2f}  E {d['E']:.2e}")
save_json("layer_steps", out)
