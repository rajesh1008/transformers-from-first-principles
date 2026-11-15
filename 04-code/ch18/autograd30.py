"""Chapter 18, section 18.9: the thirty-line NumPy autograd of tiny/autograd.py, checked three ways.
(1) Its size.  (2) The hand Tiny: all 112 nudges against PyTorch and against Chapter 17's wiggled nudges.
(3) The real Tiny, 2,766,592 learned numbers, on the novel's first 65 tokens (64 predictions), against
PyTorch, 64-bit; timings one thread, fastest of five, 10 November 2026, the book's build machine.  (4) What it
stored: the number of recorded steps and the bytes of the values the backward pass reads."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import _common, inspect, json, time
import torch
torch.set_num_threads(1)
import numpy as np
import tiny.autograd as ag
from hand import hand_tiny, IDS
from tiny.model import Tiny
from tiny.train import torch_grads
from _common import novel_ids, best_of, save_json, CH17_OUT

src = inspect.getsource(ag.T).rstrip("\n").split("\n")
print(f"(1) class T in tiny/autograd.py: {len(src)} lines, {sum(1 for l in src if l.strip())} of them not blank")
m = hand_tiny()
L, G = ag.grads(m, IDS); Lt, Gt = torch_grads(m, IDS)
wig = json.load(open(os.path.join(CH17_OUT, "head_nudges.json")))["all_wig"]
mine = np.concatenate([G[k].ravel() for k in Gt]); theirs = np.concatenate([Gt[k].ravel() for k in Gt])
print(f"(2) hand Tiny: loss {L:.10f} (PyTorch {Lt:.10f}); {mine.size} nudges")
print(f"    largest difference from PyTorch {np.abs(mine - theirs).max():.1e}; from Chapter 17's wiggling {np.abs(mine - np.array(wig)).max():.1e}")
print(f"    exactly zero: {int((mine == 0).sum())} (PyTorch: {int((theirs == 0).sum())}):")
for k in Gt:
    z = np.argwhere(G[k] == 0)
    if len(z):
        print(f"      {k:<10} at {z.tolist()}")
mt = Tiny(); ids = novel_ids()[:65]
L, G = ag.grads(mt, ids); Lt, Gt = torch_grads(mt, ids)
worst = max(float(np.abs(G[k] - Gt[k]).max()) for k in Gt)
rel = max(float(np.abs(G[k] - Gt[k]).max() / np.abs(Gt[k]).max()) for k in Gt if np.abs(Gt[k]).max() > 0)
n = sum(v.size for v in Gt.values())
print(f"(3) Tiny: {n:,} nudges on 64 predictions; loss {L:.6f} (PyTorch {Lt:.6f})")
print(f"    largest difference {worst:.1e}; largest relative to its table's largest nudge {rel:.1e}")
t_np = best_of(lambda: ag.grads(mt, ids)); t_pt = best_of(lambda: torch_grads(mt, ids))
print(f"    time for loss and all nudges: thirty-line autograd {t_np:.0f} ms, PyTorch (with the copy into torch) {t_pt:.0f} ms")
tables = mt.tables(); t = {k: ag.T(v.copy()) for k, v in tables.items()}
loss = ag.tiny_loss(t, ids, mt.n_layers, mt.blocks[0].att.n_heads, act=mt.blocks[0].ff.act_name)
order, seen = [], set()
def visit(x):
    if id(x) not in seen:
        seen.add(id(x)); [visit(p) for p in x.parents]; order.append(x)
visit(loss)
steps = [x for x in order if x.back is not None]
stored = sum(x.data.nbytes for x in steps)
shares = sum(x.data.nbytes for x in steps if x.data.shape[-2:] == (64, 64))
print(f"(4) the record: {len(steps):,} steps; their results take {stored/1e6:.1f} MB in 64-bit, "
      f"{shares/1e6:.2f} MB of it in 64 x 64 tables; the learned numbers take {sum(v.nbytes for v in tables.values())/1e6:.1f} MB")
save_json("autograd30", dict(lines=len(src), hand_max=float(np.abs(mine - theirs).max()), tiny_max=worst, tiny_rel=rel,
                            t_np_ms=t_np, t_pt_ms=t_pt, steps=len(steps), stored_mb=stored / 1e6))
import copy
from tiny.train import train
h1, h2 = hand_tiny(), hand_tiny()
a = train(h1, lambda s: IDS, 3, 0.5, "numpy"); b = train(h2, lambda s: IDS, 3, 0.5, "torch")
print("(5) three steps of training the hand Tiny at learning rate 0.5: grad_fn='numpy'",
      ", ".join(f"{l:.4f}" for _, l in a), "| grad_fn='torch'", ", ".join(f"{l:.4f}" for _, l in b),
      f"| largest difference {max(abs(x[1] - y[1]) for x, y in zip(a, b)):.0e}")
