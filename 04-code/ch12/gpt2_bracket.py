"""Chapter 12, section 7: one neuron followed up.  Neuron 10.888 of GPT-2 small pushes up closing
brackets (gpt2_neurons.py, part B).  Where does it fire?  GPT-2 reads the same 1,024 tokens of the novel;
each position is marked as inside or outside an open round bracket, by counting '(' and ')' in the text
read so far.  Run from 04-code/ch12:  python3 gpt2_bracket.py"""
from _common import *
from _gpt2run import weights, norm, gelu, softmax, decoder, H, D, L
from _bpe import encode
from tiny.corpus import load_text

W = weights(); dec = decoder()
alice = load_text(); i = alice.index("Alice was beginning")
ids = [50256] + encode(alice[i:i + 12000].replace("\r", ""))[:1023]
n = len(ids); x = W["wte.weight"][ids] + W["wpe.weight"][:n]; mask = np.tril(np.ones((n, n), dtype=bool))
for l in range(11):
    p = f"h.{l}."
    a = norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
    q, k, v = (t.reshape(n, H, D).transpose(1, 0, 2) for t in np.split(a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"], 3, axis=-1))
    s = np.where(mask, q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(D)), np.float32(-1e9))
    x = x + (softmax(s) @ v).transpose(1, 0, 2).reshape(n, H * D) @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
    m = norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
    h = gelu(m @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
    if l == 10:
        act = h[:, 888]
    x = x + h @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
depth = []; d = 0
for t in ids:
    s = dec(t) if t != 50256 else ""
    d += s.count("(") - s.count(")"); depth.append(d)
inside = np.array([dd > 0 for dd in depth])
print(f"neuron 10.888 on {n:,} tokens of the novel: positions inside an open bracket {inside.sum()} of {n}")
print(f"  mean activation inside {act[inside].mean():.3f} | outside {act[~inside].mean():.3f}")
print(f"  activations above 1: {int((act > 1).sum())}, of which inside a bracket {int((act[inside] > 1).sum())}")
top = np.argsort(-act)[:20]
print(f"  of its 20 strongest positions, inside a bracket: {int(inside[top].sum())}")
close = [j for j in range(n - 1) if ")" in dec(ids[j + 1])]
print(f"  positions just before a ')' token: {len(close)}; activation there: {[round(float(act[j]), 2) for j in close]}")
np.save(os.path.join(HERE, "outputs", "bracket_act.npy"), np.stack([act, inside.astype(np.float32)]))   # for fig12-04
