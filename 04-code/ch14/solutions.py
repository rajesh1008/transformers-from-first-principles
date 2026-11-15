"""Chapter 14 exercise solutions (Appendix A).  Seed 1865 throughout.
Run from 04-code/ch14:  python3 solutions.py   (about a minute; exercise 9 loads GPT-2 small)"""
from _common import *
from tiny.model import Tiny
from tiny.attention import softmax as tsoftmax

print("Exercise 1. top row (1, 0.5) scored against the toy rows")
rows = {"the": (1, 0), "cat": (0, 1), "sat": (1, 1), "on": (1, -1), "mat": (-1, 1)}
h = np.array([1.0, 0.5])
sc = {w: float(h @ np.array(r)) for w, r in rows.items()}
e = {w: np.exp(s) for w, s in sc.items()}; tot = sum(e.values())
for w in rows:
    print(f"  {w:4s} score {sc[w]:+.3f}  e^score {e[w]:.3f}  share {e[w] / tot:.3f}")
print(f"  sum of e^score {tot:.3f}")

print("Exercise 2. equal unrelated writes, after 24 writes")
print(f"  cos with the row below sqrt(24/25) = {np.sqrt(24 / 25):.4f}; cos with the start 1/sqrt(25) = {1 / np.sqrt(25):.4f}")

print("Exercise 3. Tiny with six layers")
m6 = Tiny(n_layers=6)
print(f"  by hand 262,144 + 131,072 + 6 x 197,760 + 256 = {262144 + 131072 + 6 * 197760 + 256:,}; counted {m6.n_params:,}")

print("Exercise 4. GPT-2 medium: V 50,257, C 1,024, d 1,024, 24 layers, d_ff 4,096, biases everywhere, tied")
V, C, d, Lm, ff = 50257, 1024, 1024, 24, 4096
att = d * 3 * d + 3 * d + d * d + d
ffn = d * ff + ff + ff * d + d
nrm = 4 * d
layer = att + ffn + nrm
total = V * d + C * d + Lm * layer + 2 * d
print(f"  attention {att:,}; feed-forward {ffn:,}; norms {nrm:,}; one layer {layer:,}; total {total:,}")

print("Exercise 5. rule of thumb, 32 layers of width 4,096")
print(f"  12 x 32 x 4096^2 = {12 * 32 * 4096 ** 2:,}")

print("Exercise 6. Tiny's operations per token at a 256-token text")
L, d, ff, n, Vt = 12, 128, 512, 256, 2048
lin = 2 * L * (4 * d * d + 2 * d * ff); att6 = 2 * 2 * d * (n + 1) / 2 * L; out = 2 * d * Vt
print(f"  tables {lin:,}; scores and blends {att6:,.0f}; output {out:,}; total {lin + att6 + out:,.0f}; attention part / tables {att6 / lin:.3f}")

t = tok(); ids = sentence_ids(t)
print("Exercise 7. untrained Tiny with an untied output table, Logit Lens on ' sat' (row 2)")
mu = Tiny(tied=False)
S, stream = mu.forward(ids, keep=True)
for l in [0, 1, 2, 12]:
    sc7 = mu.scores_from(mu.top_rows(stream[l]))[2]
    top = [piece(t, int(i)) for i in np.argsort(sc7)[::-1][:3]]
    print(f"  enter {l if l < 12 else 'top':>3}: top three {top}; rank of ' sat' {int((sc7 > sc7[ids[2]]).sum()) + 1}")
mt = Tiny()
S, stream = mt.forward(ids, keep=True)
sc0 = mt.scores_from(mt.top_rows(stream[0]))[2]
print(f"  tied, for comparison, enter 0: rank of ' sat' {int((sc0 > sc0[ids[2]]).sum()) + 1}")

print("Exercise 10. learned numbers: untied and without ln_f")
print(f"  Tiny(tied=False, final_norm=False): {Tiny(tied=False, final_norm=False).n_params:,}")

print("Exercise 9. GPT-2 small after 'the cat sat on the': layers (1, 6) removed and layers (6, 7) removed")
from _gpt2run import weights, block, norm, L as GL, decoder
W = weights(); dec = decoder()
g = np.array([1169, 3797, 3332, 319, 262, 2603])
def top3(skip):
    x = W["wte.weight"][g] + W["wpe.weight"][:len(g)]
    for l in range(GL):
        if l not in skip:
            x, _ = block(x, l)
    s = norm(x, W["ln_f.weight"], W["ln_f.bias"]) @ W["wte.weight"].T
    s4 = s[4]
    return [dec(int(i)) for i in np.argsort(s4)[::-1][:3]], int((s4 > s4[2603]).sum()) + 1
for sk in [(), (1, 6), (6, 7)]:
    tt, r = top3(sk)
    print(f"  removed {sk if sk else 'none'}: top three {tt}; rank of ' mat' {r}")
