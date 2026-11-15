"""Chapter 15, section 7: checking every hand number against Tiny's own code.
The pencil trace (pencil.py: plain Python, no eps) against Tiny built from its classes with the hand
tables (hand.py), step by step; then the same with eps set to 0; then the pencil trace rounded to three
places at every step; then the things the code does that the pencil does not.
Run from 04-code/ch15:  python3 check.py > outputs/check.txt"""
from _common import *
import math
import pencil
from hand import hand_tiny, WORDS, IDS, SEAT_WORDS, W_q, W_k, W_v, W_o, W_1, b_1, W_2, b_2, CONFIG
from tiny.embed import TOY_EMBED, TOY_POS
from tiny.attention import softmax

TAB = {k: v.tolist() for k, v in dict(W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, W_1=W_1, b_1=b_1, W_2=W_2, b_2=b_2).items()}
E, P = TOY_EMBED.tolist(), TOY_POS.tolist()
pen = pencil.trace(E, P, IDS, TAB)
pen3 = pencil.trace(E, P, IDS, TAB, rounded=True)

m = hand_tiny()
print("Tiny built from its own classes:", ", ".join(f"{k}={v}" for k, v in CONFIG.items()))
print("  learned numbers by part (Tiny.count):", m.count(), "total", m.n_params)
print("  every table loaded through the shape check of Chapter 14: yes (Tiny.load -> Block.set_tables)")

def code_trace(m, ids):
    X = m.embed(ids)
    out = {"x0": X}
    for l, b in enumerate(m.blocks):
        st = b.explode(X)
        _, sh = b.att(b.ln_1(X), return_shares=True)
        out[f"layer{l}"] = {"ln_1": st["ln_1"], "shares": sh[0], "att": st["attention"], "add_1": st["add_1"],
                            "ln_2": st["ln_2"], "ff": st["feed_forward"], "add_2": st["add_2"]}
        X = st["add_2"]
    out["ln_f"] = m.top_rows(X)
    S = m.forward(ids)
    out["out_scores"], out["out_shares"] = S, softmax(S)
    return out

steps = [("x0", None), ("ln_1", 0), ("shares", 0), ("att", 0), ("add_1", 0), ("ln_2", 0), ("ff", 0), ("add_2", 0),
         ("ln_1", 1), ("shares", 1), ("att", 1), ("add_1", 1), ("ln_2", 1), ("ff", 1), ("add_2", 1),
         ("ln_f", None), ("out_scores", None), ("out_shares", None)]

def get(T, name, l):
    v = T[name] if l is None else T[f"layer{l}"][name]
    return np.array(v, dtype=float)

def compare(code, ref, label):
    print(label)
    worst = 0.0; rows = []
    for name, l in steps:
        d = float(np.abs(get(code, name, l) - get(ref, name, l)).max())
        worst = max(worst, d); rows.append((name, l, d))
        print(f"  {name + ('' if l is None else f' (layer {l})'):<22} largest difference {d:.2e}")
    return worst, rows

ct = code_trace(m, IDS)
w1, rows1 = compare(ct, pen, "\n(a) Tiny's code against the pencil trace at full precision:")
for l in range(2):
    for nm in ("ln_1", "ln_2"):
        getattr(m.blocks[l], nm).eps = 0.0
m.ln_f.eps = 0.0
ct0 = code_trace(m, IDS)
w0, rows0 = compare(ct0, pen, "\n(b) the same with eps = 0 inside every RMSNorm (the pencil has no eps):")
w3, rows3 = compare(ct, pen3, "\n(c) Tiny's code against the pencil trace rounded to three places after every step:")

print("\nwhere the eps difference comes from, the 'the' row at seat 4, (0.5, 0.5):")
x = np.array([0.5, 0.5])
print(f"  pencil: rms = sqrt(0.25) = 0.5, row / rms = (1, 1) exactly")
print(f"  code:   rms = sqrt(0.25 + 0.00001) = {math.sqrt(0.25 + 1e-5):.8f}, row / rms = {r3(x / math.sqrt(0.25 + 1e-5), 8)}")

print("\ndoes rounding to three places change any decision?")
for i in range(6):
    a, b = np.array(pen["out_shares"][i]), np.array(pen3["out_shares"][i])
    print(f"  after {SEAT_WORDS[i]}@{i}: top word {WORDS[int(a.argmax())]} / {WORDS[int(b.argmax())]}; "
          f"largest share difference {np.abs(a - b).max():.4f}")
nxt = IDS[1:]
L = lambda T: sum(-math.log(T["out_shares"][i][nxt[i]]) for i in range(5)) / 5
print(f"  average surprise: full precision {L(pen):.4f}, rounded {L(pen3):.4f}, Tiny's code {float(np.mean([-np.log(ct['out_shares'][i][nxt[i]]) for i in range(5)])):.4f}")

print("\nthe stable softmax: the code subtracts each row's largest score before exp; the pencil does not")
big = [1000.0, 999.0]
try:
    e = [math.exp(s) for s in big]; print("  pencil:", e)
except OverflowError as err:
    print(f"  pencil on scores {big}: OverflowError ({err})")
print(f"  code on the same scores: {r3(softmax(np.array(big)), 6)}; on the hand scores of sat@2 in layer 0 the two agree to "
      f"{np.abs(softmax(np.array(pen['layer0']['scaled'][2][:3])) - np.array(pen['layer0']['shares'][2][:3])).max():.1e}")

print("\nthe mask: the code writes -inf and lets exp turn it into 0; the pencil skips the cell")
sc = np.array(pen["layer0"]["scaled"][2])
masked = np.where(np.arange(6) <= 2, sc, -np.inf)
print(f"  sat@2's row with -inf: {['-inf' if not np.isfinite(v) else round(float(v), 3) for v in masked]} -> shares {r3(softmax(masked))}")

print("\nnormalisation at width 2: Tiny's default LayerNorm in place of RMSNorm")
from tiny.norm import layer_norm, rms_norm
for i in range(6):
    xr = np.array(pen["x0"][i])
    print(f"  {SEAT_WORDS[i]}@{i} {r3(xr)}: layer norm -> {r3(layer_norm(xr), 4)} | RMSNorm -> {r3(rms_norm(xr), 4)}")
ml = hand_tiny(norm="layer")
Sl = softmax(ml.forward(IDS))
print("  the whole hand Tiny with layer norm: top word after each seat", [WORDS[int(k)] for k in Sl.argmax(1)],
      "| largest share", round(float(Sl.max()), 3))
_, st = ml.forward(IDS, keep=True)
print("  rows reaching ln_f:", [r3(r) for r in st[-1]])
print("  after ln_f (layer norm):", [r3(r, 3) for r in ml.top_rows(st[-1])])

print("\nthe hand Tiny without the position table (seat rows all zero):")
mn = hand_tiny(positions=False)
_, stn = mn.forward(IDS, keep=True)
print(f"  the@0 and the@4 entering layer 0: {r3(stn[0][0])} and {r3(stn[0][4])}; after layer 0: {r3(stn[1][0])} and {r3(stn[1][4])}")
Sn = softmax(mn.forward(IDS))
print("  top word after each seat:", [WORDS[int(k)] for k in Sn.argmax(1)])

print("\nthe same trace at Tiny's real size (seed 1865, twelve layers, width 128), names and shapes only:")
from tiny.model import Tiny
from _common import TINY_BPE
from tiny.tokenizer import load_bpe
tok = load_bpe(TINY_BPE); ids = tok.encode(SENTENCE)
big = Tiny()
X = big.embed(ids)
n = len(ids)
total = X.size
print(f"  rows in: {X.shape}")
names = []
for l, b in enumerate(big.blocks):
    st = b.explode(X)
    Q = b.ln_1(X) @ b.att.W_q
    _, sh = b.att(b.ln_1(X), return_shares=True)
    per = {"ln_1": st["ln_1"].size, "queries, keys, values": 3 * Q.size, "scores and shares": 2 * sh.size,
           "blends": Q.size, "attention write (after W_o)": st["attention"].size, "add_1": st["add_1"].size,
           "ln_2": st["ln_2"].size, "matches (512 per row)": n * b.ff.d_ff, "activations": n * b.ff.d_ff,
           "feed-forward write": st["feed_forward"].size, "add_2": st["add_2"].size}
    if l == 0:
        for k, v in per.items():
            print(f"    layer 0 {k:<28} {v:>7,}")
    total += sum(per.values())
    X = st["add_2"]
S = big.forward(ids)
total += big.top_rows(X).size + 2 * S.size
print(f"  intermediates in one forward pass of the running sentence, all twelve layers and the output: {total:,} numbers")
hand_total = sum(np.array(v).size for k, v in pen.items() if k in ("x0", "ln_f", "out_scores", "out_shares"))
for l in range(2):
    Lh = pen[f"layer{l}"]
    hand_total += sum(np.array(Lh[k]).size for k in ("ln_1", "q", "k", "v", "scaled", "shares", "blend", "att", "add_1", "ln_2", "match", "active", "ff", "add_2"))
print(f"  the hand trace writes down {hand_total:,} numbers (masked cells included as they are printed)")
save_json("check", {"max_diff_eps": w1, "max_diff_eps0": w0, "max_diff_rounded": w3,
                    "rows_eps": rows1, "rows_eps0": rows0, "rows_rounded": rows3, "tiny_intermediates": total, "hand_numbers": hand_total})

print("\ncontinuity with Chapter 13 (no positions, eps 0): the hand Tiny reproduces residual_hand.py's pre-norm rows")
for l in range(2):
    for nm in ("ln_1", "ln_2"):
        getattr(mn.blocks[l], nm).eps = 0.0
_, stc = mn.forward(IDS, keep=True)
print(f"  after one layer: sat {r3(stc[1][2])}, mat {r3(stc[1][5])}; after two: sat {r3(stc[2][2])}, mat {r3(stc[2][5])}"
      "  (Chapter 13 printed sat [3.052, 2.262], mat [0.701, 0.779]; sat [5.331, 3.603], mat [2.961, 2.114])")

print("\nneuron 3 after a norm: key (1, 1), bias -2, rows of RMS 1 (length sqrt(2))")
ang = np.linspace(0, 2 * np.pi, 3601)
rows = np.sqrt(2) * np.c_[np.cos(ang), np.sin(ang)]
match3 = rows @ np.array([1.0, 1.0]) - 2.0
print(f"  largest match over every direction (3,601 directions): {match3.max():.6f}, reached at {r3(rows[match3.argmax()], 4)}")
print("  so after RMSNorm with gain 1, neuron 3 can never fire: its largest possible match is 0")

print("\nwhy every seat picks 'looked': the score is (length of the top row) x (length of the word's row) x cosine")
from tiny.norm import token_similarity
z = np.array(pen["ln_f"][5])
for k in np.argsort(-np.array(pen["out_scores"][5]))[:6]:
    e = TOY_EMBED[k]
    c = float(z @ e / (np.linalg.norm(z) * np.linalg.norm(e))) if np.linalg.norm(e) else float("nan")
    print(f"  {WORDS[k]:<7} row {r3(e)} length {np.linalg.norm(e):.3f}, cosine with mat@5's top row {c:.3f}, score {pen['out_scores'][5][k]:+.3f}")
print("  token similarity of the six rows: entering layer 0 "
      f"{token_similarity(np.array(pen['x0'])):.3f}, after layer 0 {token_similarity(np.array(pen['layer0']['add_2'])):.3f}, "
      f"after layer 1 {token_similarity(np.array(pen['top'])):.3f}, after ln_f {token_similarity(np.array(pen['ln_f'])):.3f}")
