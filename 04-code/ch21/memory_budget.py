"""Chapter 21, sections 21.7 to 21.9: where the bytes of a training step go.
Part 1 counts the bytes kept between steps (weights, nudges, master copy, optimiser state) under five recipes.
Part 2 measures, with saved-tensor hooks, the bytes the forward pass keeps for the backward pass (the stored
activations) for Tiny on 8 windows of 128 predictions, in 32-bit, mixed BF16 and pure BF16, with and without
recomputing each layer in the backward pass (activation checkpointing).  Part 3 separates Tiny's per-layer
bytes into a part that grows with tokens and a part that grows with tokens squared, and Part 4 applies the
same counts to a 7-billion-number model.  Counts only; the time each choice costs is in budget.py."""
import _common
import numpy as np
import torch
torch.set_num_threads(1)
from torch.utils.checkpoint import checkpoint
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.precision import MixedTiny
from _common import SEED, save_json
N = 2_766_592
print("Part 1. bytes kept between steps, per learned number and for Tiny")
RECIPES = [
 ("32-bit, Adam (Chapter 20)", [("weights", 4), ("nudges", 4), ("Adam m and v", 8)]),
 ("mixed BF16, 32-bit master and Adam", [("16-bit weights", 2), ("16-bit nudges", 2), ("32-bit master", 4), ("Adam m and v", 8)]),
 ("mixed BF16, Adam m and v in BF16", [("16-bit weights", 2), ("16-bit nudges", 2), ("32-bit master", 4), ("Adam m and v", 4)]),
 ("mixed BF16, 8-bit Adam state", [("16-bit weights", 2), ("16-bit nudges", 2), ("32-bit master", 4), ("Adam m and v", 2)]),
 ("BF16 only, no master copy", [("16-bit weights", 2), ("16-bit nudges", 2), ("Adam m and v", 4)]),
]
for name, parts in RECIPES:
    b = sum(p for _, p in parts)
    print(f"  {name:38s} {' + '.join(f'{p} {k}' for k, p in parts):70s} = {b:2d} bytes; Tiny {b * N / 1e6:6.1f} MB; 7B {b * 7:4d} GB")

class Ckpt:
    """Wraps a TorchTiny or MixedTiny so that each layer's inside is recomputed in the backward pass;
    only each layer's input row is kept (torch.utils.checkpoint, reentrant form, so the hooks see what it keeps)."""
    def __init__(self, tt):
        self.tt = tt
    def loss(self, ids):
        tt = self.tt
        ids = torch.as_tensor(ids)
        inp, tgt = ids[..., :-1], ids[..., 1:]
        x = tt.t["E"][inp] + tt.t["P"][:inp.shape[-1]]
        for l in range(tt.config["n_layers"]):
            def layer(x, l=l):
                x = x + tt._attention(tt._norm(x, f"h.{l}.ln_1"), l)
                return x + tt._ffn(tt._norm(x, f"h.{l}.ln_2"), l)
            x = checkpoint(layer, x, use_reentrant=True)
        x = tt._norm(x, "ln_f")
        S = tt.mm(x, tt.t["E"].T) if hasattr(tt, "mm") else x @ tt.t["E"].T
        return -torch.log_softmax(S.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()

def saved_bytes(model, batch, layers=None):
    """Bytes of the distinct tensors the forward pass saves for the backward pass (learned tables excluded)."""
    tt = model.tt if isinstance(model, Ckpt) else model
    keep = tt.config["n_layers"]
    if layers is not None:
        tt.config["n_layers"] = layers
    seen = {}
    def pack(x):
        if not (x.requires_grad and x.is_leaf):
            key = (x.untyped_storage().data_ptr(), x.storage_offset(), tuple(x.shape), tuple(x.stride()), x.dtype)
            seen.setdefault(key, x.numel() * x.element_size())
        return x
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        L = model.loss(batch)
    L.backward()
    tt.config["n_layers"] = keep
    return sum(seen.values())

m = Tiny()
gen = np.random.default_rng(SEED)
def make_batch(b, n):
    return torch.tensor(gen.integers(0, 2048, size=(b, n + 1)))
batch = make_batch(8, 128)
print("\nPart 2. stored activations for one step of Tiny, 8 windows x 128 predictions (MB, measured)")
res = {}
for name, mk in [("32-bit", lambda: TorchTiny.from_numpy(m, dtype=torch.float32)),
                 ("mixed BF16 (products 16-bit, rest 32-bit)", lambda: MixedTiny.from_numpy(m)),
                 ("pure BF16", lambda: TorchTiny.from_numpy(m, dtype=torch.bfloat16))]:
    full = saved_bytes(mk(), batch)
    ck = saved_bytes(Ckpt(mk()), batch)
    res[name] = dict(full=full, ckpt=ck)
    print(f"  {name:44s} keep everything {full / 1e6:7.1f}   recompute each layer {ck / 1e6:6.1f}   ({full / ck:4.1f} times less)")
w32 = 4 * N / 1e6
print(f"  for scale: Tiny's 32-bit weights are {w32:.1f} MB, and weights + nudges + Adam {4 * w32:.1f} MB")

print("\nPart 3. one layer's stored activations, 32-bit, split into a part per token and a part per (token, token) pair")
per = {}
for n in (64, 128, 256):
    bt = make_batch(8, n)
    a = saved_bytes(TorchTiny.from_numpy(m, dtype=torch.float32), bt, layers=12)
    b_ = saved_bytes(TorchTiny.from_numpy(m, dtype=torch.float32), bt, layers=6)
    per[n] = (a - b_) / 6
b, d, h = 8, 128, 4
# per layer bytes = 4 * (alpha * b * n * d + beta * b * h * n * n)
A = np.array([[4 * b * n * d, 4 * b * h * n * n] for n in (64, 128, 256)], dtype=float)
y = np.array([per[n] for n in (64, 128, 256)])
(alpha, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
for n in (64, 128, 256):
    print(f"  n = {n:3d}: {per[n] / 1e6:6.2f} MB per layer; fit {(alpha * A[(64, 128, 256).index(n)][0] + beta * A[(64, 128, 256).index(n)][1]) / 1e6:6.2f}")
print(f"  each layer keeps {alpha:.2f} numbers per token per unit of width (b n d) and {beta:.2f} per attention cell (b h n^2)")

print("\nPart 4. the same count for a 7-billion-number model (32 layers, width 4,096, 32 heads, feed-forward 11,008 gated,")
print("        vocabulary 32,000; 4,096 tokens per sequence, one sequence per device), 16-bit activations")
L7, d7, h7, f7, V7, s7 = 32, 4096, 32, 11008, 32000, 4096
P7 = V7 * d7 * 2 + L7 * (4 * d7 * d7 + 3 * d7 * f7 + 2 * d7) + d7
print(f"  learned numbers {P7:,}; states at 16 bytes each {16 * P7 / 1e9:.1f} GB")
# Tiny's alpha counts its own layer (d_ff = 4d, plain); scale the feed-forward part to 11,008 gated (three tables)
act_bnd = alpha * s7 * d7 * 2 * L7
act_att = beta * h7 * s7 * s7 * 2 * L7
print(f"  with Tiny's per-layer counts: {act_bnd / 1e9:.1f} GB that grow with tokens + {act_att / 1e9:.1f} GB of attention tables = {(act_bnd + act_att) / 1e9:.1f} GB")
mg_bnd, mg_att = 34 * s7 * d7 * L7, 5 * h7 * s7 * s7 * L7
print(f"  with the 2022 count for a fused GPT layer (34 bytes per token per width + 5 per attention cell, in 16-bit):"
      f" {mg_bnd / 1e9:.1f} + {mg_att / 1e9:.1f} = {(mg_bnd + mg_att) / 1e9:.1f} GB")
print(f"  without storing attention tables (a fused attention kernel recomputes them): {act_bnd / 1e9:.1f} GB")
print(f"  recomputing every layer (keep each layer's input only): {L7 * s7 * d7 * 2 / 1e9:.2f} GB")
for nd in (1, 8, 64):
    z = [16 * P7 / 1e9, (4 + 12 / nd) * P7 / 1e9, (2 + 14 / nd) * P7 / 1e9, 16 * P7 / nd / 1e9]
    print(f"  states per device over {nd:2d} devices: none {z[0]:6.1f} GB | shard optimiser {z[1]:6.1f} | + nudges {z[2]:6.1f} | + weights {z[3]:6.1f}")
save_json("memory_budget", dict(recipes=[(n, p) for n, p in RECIPES], tiny=res, per_layer=per, alpha=alpha, beta=beta,
                                P7=P7, act_bnd=act_bnd, act_att=act_att))
