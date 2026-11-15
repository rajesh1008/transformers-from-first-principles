"""Chapter 12, section 4: where the learned numbers are.  GPT-2 small's counts are read from the shapes
in its weight file's header (nothing is assumed); Tiny's from its classes.  The gated sizing rule is
checked by arithmetic.  Run from 04-code/ch12:  python3 params.py"""
from _common import *
from math import prod
from _gpt2 import header
from tiny.attention import MultiHead
from tiny.ffn import FeedForward

h, _ = header()
groups = {"token table (wte)": 0, "position table (wpe)": 0, "attention tables": 0, "attention biases": 0,
          "feed-forward tables": 0, "feed-forward biases": 0, "normalisation": 0}
for name, info in h.items():
    if name == "__metadata__" or name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
        continue                                  # the stored causal mask is a constant, not a learned number
    n = prod(info["shape"])
    if name.startswith("wte"): groups["token table (wte)"] += n
    elif name.startswith("wpe"): groups["position table (wpe)"] += n
    elif ".attn." in name: groups["attention biases" if name.endswith("bias") else "attention tables"] += n
    elif ".mlp." in name: groups["feed-forward biases" if name.endswith("bias") else "feed-forward tables"] += n
    elif ".ln_" in name or name.startswith("ln_f"): groups["normalisation"] += n
    else: raise SystemExit("unclassified " + name)
total = sum(groups.values())
print("GPT-2 small, learned numbers by kind (from the weight file's shapes):")
for k, v in groups.items():
    print(f"  {k:<22} {v:>12,}  {v / total:6.1%}")
print(f"  {'total':<22} {total:>12,}")
inside = total - groups["token table (wte)"] - groups["position table (wpe)"]
ff = groups["feed-forward tables"] + groups["feed-forward biases"]
att = groups["attention tables"] + groups["attention biases"]
print(f"inside the 12 layers: {inside:,}; feed-forward {ff:,} = {ff / inside:.1%}, attention {att:,} = {att / inside:.1%}")
print(f"one layer's feed-forward block: c_fc {h['h.0.mlp.c_fc.weight']['shape']} + bias {h['h.0.mlp.c_fc.bias']['shape']}, "
      f"c_proj {h['h.0.mlp.c_proj.weight']['shape']} + bias {h['h.0.mlp.c_proj.bias']['shape']} = {ff // 12:,} numbers; 3,072 neurons")

mh, f = MultiHead(n_heads=4), FeedForward()
g = FeedForward(gated=True, activation="silu")
tiny_before = 2048 * 128 + 1024 * 128 + mh.n_params
print(f"\nTiny: token table 262,144 + position table 131,072 + attention {mh.n_params:,} = {tiny_before:,} (Chapter 11)")
print(f"  plus a feed-forward block 128 -> 512 -> 128 with biases: {f.n_params:,} -> {tiny_before + f.n_params:,} learned numbers")
print(f"  inside the layer: attention {mh.n_params:,}, feed-forward {f.n_params:,} ({f.n_params / (f.n_params + mh.n_params):.1%})")
print(f"  gated alternative (SiLU gate, d_ff = {g.d_ff}): {g.n_params:,}")

print("\nkeeping the gated block the same size as a plain one with d_ff = 4d: d_ff = 8d/3")
for d in (128, 768, 4096):
    plain = 2 * d * 4 * d
    exact = 8 * d / 3
    print(f"  d = {d:>5}: plain 2 x d x 4d = {plain:>13,} | gated 3 x d x 8d/3 = {round(3 * d * exact):>13,} | 8d/3 = {exact:,.1f}")

for n in (1, 1024):
    att_ops = 4 * n * 768 * 768 * 2 + 4 * n * n * 768 / 2            # tables, plus causal scores and blend (half the table)
    ff_ops = 2 * n * 768 * 3072 * 2
    print(f"GPT-2 small, one layer, {n:>5} tokens: feed-forward {ff_ops / 1e6:,.1f}M operations, attention {att_ops / 1e6:,.1f}M (tables plus causal half of the scores)")
