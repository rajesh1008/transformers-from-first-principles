"""Chapter 14, section 5: counting learned numbers by hand, for Tiny and for GPT-2 small, and checking
both counts against the tables themselves (Tiny's objects; GPT-2's weight-file header).
Run from 04-code/ch14:  python3 params.py"""
from _common import *
from tiny.model import Tiny
from _gpt2 import header

def by_hand(V, C, d, L, d_ff, bias_att, bias_ff, norm_bias=True, tied=True):
    norm = 2 * d if norm_bias else d
    att = 4 * d * d + (4 * d if bias_att else 0)
    ff = 2 * d * d_ff + ((d_ff + d) if bias_ff else 0)
    layer = 2 * norm + att + ff
    total = V * d + C * d + L * layer + norm + (0 if tied else V * d)
    return {"token table": V * d, "position table": C * d, "one layer": layer, "attention": att,
            "feed-forward": ff, "norms per layer": 2 * norm, "final norm": norm, "total": total}

print("Tiny: V = 2,048 tokens, C = 1,024 seats, d = 128, 12 layers, d_ff = 512; attention tables without")
print("biases (Chapter 9), feed-forward with biases (Chapter 12), layer norms with gain and bias, tied output")
h = by_hand(2048, 1024, 128, 12, 512, bias_att=False, bias_ff=True)
for k, v in h.items():
    print(f"  {k:<16} {v:>12,}")
print(f"  formula: V*d + C*d + L*(4d^2 + 2*d*d_ff + d_ff + d + 4d) + 2d = {h['total']:,}")
m = Tiny()
print(f"  counted from Tiny's own tables: {m.n_params:,}   match: {m.n_params == h['total']}")
for part, n in m.count().items():
    print(f"    {part:<22} {n:>10,}  {n / m.n_params:6.1%}")

print("\nGPT-2 small: V = 50,257, C = 1,024, d = 768, 12 layers, d_ff = 3,072; biases on every table, tied output")
g = by_hand(50257, 1024, 768, 12, 3072, bias_att=True, bias_ff=True)
for k, v in g.items():
    print(f"  {k:<16} {v:>12,}")
hd, _ = header()
names = [k for k in hd if k != "__metadata__"]
n_file = sum(int(np.prod(hd[k]["shape"])) for k in names)
n_learned = sum(int(np.prod(hd[k]["shape"])) for k in names if not k.endswith(".attn.bias"))
print(f"  tables in the weight file: {len(names)}; numbers in the file {n_file:,}")
print(f"  of which the causal masks stored as 'attn.bias' (not learned): {n_file - n_learned:,}")
print(f"  learned numbers in the file: {n_learned:,}   match with the hand count: {n_learned == g['total']}")
print(f"  share of GPT-2 in its token table: {g['token table'] / g['total']:.1%}; in its twelve layers: {12 * g['one layer'] / g['total']:.1%}")
print(f"  Tiny's layers: {12 * h['one layer'] / h['total']:.1%}")

print("\nthe rule of thumb 12 * L * d^2 for the layers (ignoring biases, norms and the tables at either end):")
for name, L, d, exact in [("Tiny", 12, 128, 12 * h["one layer"]), ("GPT-2 small", 12, 768, 12 * g["one layer"])]:
    print(f"  {name:<12} 12 x {L} x {d}^2 = {12 * L * d * d:>12,}   exact layers {exact:>12,}   ({12 * L * d * d / exact:.4f})")
