"""Chapter 15, 'what the hand showed that the code hid': the order of adding and the width of the numbers.
(1) the same three numbers added in two orders; (2) Tiny (seed 1865, twelve layers) on the running sentence,
computed with its own tables in float64, float32 and float16, and in float32 with every sum inside the
output head taken in reverse order; (3) the hand Tiny in float16.  Run from 04-code/ch15:
python3 order.py > outputs/order.txt"""
from _common import *
from tiny.model import Tiny
from tiny.tokenizer import load_bpe
from hand import hand_tiny, IDS, WORDS

print("(1) adding in a different order")
a, b, c = 0.1, 0.2, 0.3
print(f"  float64: (0.1 + 0.2) + 0.3 = {(a + b) + c!r}; 0.1 + (0.2 + 0.3) = {a + (b + c)!r}; equal: {(a + b) + c == a + (b + c)}")
a32, b32, c32 = np.float32(1e8), np.float32(1.0), np.float32(-1e8)
print(f"  float32: (1e8 + 1) - 1e8 = {float((a32 + b32) + c32)}; (1e8 - 1e8) + 1 = {float((a32 + c32) + b32)}")

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi).astype(x.dtype) * (x + x.dtype.type(0.044715) * x ** 3)))

def forward(m, ids, dt, reverse_head=False):
    """Tiny's forward pass written out with every table cast to dtype dt (pre-norm, four heads, GELU, ln_f, tied)."""
    eps = dt(1e-5)
    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(v + eps) * g.astype(dt) + b.astype(dt)
    E = m.E.table.astype(dt)
    X = E[ids] + m.P.table[:len(ids)].astype(dt)
    n = len(ids)
    mask = np.tril(np.ones((n, n), bool))
    for blk in m.blocks:
        t = {k: v.astype(dt) for k, v in blk.tables().items()}
        h = ln(X, t["ln_1.gain"], t["ln_1.bias"])
        H, dh = blk.att.n_heads, blk.att.d_head
        Q = (h @ t["att.W_q"]).reshape(n, H, dh).transpose(1, 0, 2)
        K = (h @ t["att.W_k"]).reshape(n, H, dh).transpose(1, 0, 2)
        V = (h @ t["att.W_v"]).reshape(n, H, dh).transpose(1, 0, 2)
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(dt(dh))
        S = np.where(mask, S, -np.inf).astype(dt)
        S = S - S.max(-1, keepdims=True)
        A = np.exp(S); A = A / A.sum(-1, keepdims=True)
        O = (A @ V).transpose(1, 0, 2).reshape(n, H * dh) @ t["att.W_o"]
        X = X + O
        h = ln(X, t["ln_2.gain"], t["ln_2.bias"])
        X = X + (gelu(h @ t["ff.W_1"] + t["ff.b_1"]) @ t["ff.W_2"] + t["ff.b_2"])
    Z = ln(X, m.ln_f.gain, m.ln_f.bias)
    if not reverse_head:
        return (Z @ E.T).astype(np.float64)
    out = np.zeros((n, E.shape[0]), dtype=dt)
    for j in range(Z.shape[1] - 1, -1, -1):           # add the 128 products in the opposite order
        out = (out + np.outer(Z[:, j], E[:, j])).astype(dt)
    return out.astype(np.float64)

tok = load_bpe(TINY_BPE); ids = tok.encode(SENTENCE)
m = Tiny()
ref = m.forward(ids)
s64 = forward(m, ids, np.float64)
print("\n(2) untrained Tiny (seed 1865) on the running sentence, scores (7, 2048)")
print(f"  this file's float64 pass against Tiny.forward: largest difference {np.abs(s64 - ref).max():.1e}")
s32 = forward(m, ids, np.float32)
s32r = forward(m, ids, np.float32, reverse_head=True)
s16 = forward(m, ids, np.float16)
top = lambda s: [int(k) for k in np.argsort(-s[-1])[:5]]
for name, s in (("float32", s32), ("float32, output sums reversed", s32r), ("float16", s16)):
    d = np.abs(s - ref).max()
    same5 = top(s) == top(ref)
    print(f"  {name:<30} largest score difference {d:.2e}; top five after 'at' unchanged: {same5}")
print(f"  float32 against float32 with the output sums reversed: largest difference {np.abs(s32 - s32r).max():.2e}; "
      f"scores that differ in any bit: {int((s32 != s32r).sum()):,} of {s32.size:,}")
gap = np.sort(ref[-1])[::-1]
print(f"  gap between Tiny's first and second choice after 'at' (float64): {gap[0] - gap[1]:.4f}; between the 5th and 6th: {gap[4] - gap[5]:.6f}")
r16 = np.argsort(-s16[-1])[:5]
import importlib.util
spec = importlib.util.spec_from_file_location("c14", os.path.join(HERE, "..", "ch14", "_common.py")); c14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c14)
print("  float16 top five after 'at':", [c14.piece(tok, k) for k in r16], "against", [c14.piece(tok, k) for k in top(ref)])

print("\n(3) the hand Tiny in float16")
hm = hand_tiny()
h64 = hm.forward(IDS)
# the hand Tiny uses RMSNorm and ReLU; compute it directly with float16 tables
def hand16(m, ids, dt=np.float16):
    eps = dt(1e-5)
    def rms(x, g):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g.astype(dt)
    E = m.E.table.astype(dt); X = E[ids] + m.P.table[:len(ids)].astype(dt)
    n = len(ids); mask = np.tril(np.ones((n, n), bool))
    for blk in m.blocks:
        t = {k: v.astype(dt) for k, v in blk.tables().items()}
        h = rms(X, t["ln_1.gain"])
        Q, K, V = h @ t["att.W_q"], h @ t["att.W_k"], h @ t["att.W_v"]
        S = np.where(mask, Q @ K.T / np.sqrt(dt(2)), -np.inf).astype(dt)
        A = np.exp(S - S.max(-1, keepdims=True)); A = A / A.sum(-1, keepdims=True)
        X = X + (A @ V) @ t["att.W_o"]
        h = rms(X, t["ln_2.gain"])
        X = X + (np.maximum(h @ t["ff.W_1"] + t["ff.b_1"], dt(0)) @ t["ff.W_2"] + t["ff.b_2"])
    return (rms(X, m.ln_f.gain) @ E.T).astype(np.float64)
h16 = hand16(hm, IDS)
print(f"  largest score difference float16 against float64: {np.abs(h16 - h64).max():.4f}; top word at every seat unchanged: "
      f"{bool((h16.argmax(1) == h64.argmax(1)).all())}")
save_json("order", {"tiny_f32": float(np.abs(s32 - ref).max()), "tiny_f32_rev": float(np.abs(s32r - ref).max()),
                    "tiny_f16": float(np.abs(s16 - ref).max()), "hand_f16": float(np.abs(h16 - h64).max())})
