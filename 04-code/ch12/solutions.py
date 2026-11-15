"""Chapter 12 exercise solutions (numerical ones).  Hand exercises use the 2 -> 3 -> 2 block of section 12.2;
Exercise 9 runs GPT-2 small 50 times (about 8 minutes on the book's CPU).  Run from 04-code/ch12:  python3 solutions.py"""
from _common import *
from tiny.ffn import relu, gelu, silu
W1 = np.array([[1., 0., 1.], [0., 1., 1.]]); b1 = np.array([0., 0., -2.])
W2 = np.array([[0., 0.5], [0.5, 0.], [-0.5, 0.5]])
def block(x): return relu(x @ W1 + b1) @ W2
def r(v): return [round(float(t), 3) + 0.0 for t in v]

x = np.array([1., 1.]); m = x @ W1 + b1
print(f"1. row (1, 1): matches {r(m)}, activations {r(relu(m))}, block adds {r(block(x))}, new row {r(x + block(x))}")
x = np.array([2., 3.]); m = x @ W1 + b1
print(f"2. row (2, 3): matches {r(m)}, activations {r(relu(m))}, block adds {r(block(x))}")
d, f = 1024, 4096
plain = 2 * d * f + f + d
fg = int(np.ceil(8 * d / 3 / 256) * 256)
print(f"3. plain block d={d}, d_ff={f}: {plain:,} numbers; gated, d_ff = 8d/3 = {8 * d / 3:.1f} rounded up to a multiple of 256 = {fg}: "
      f"{3 * d * fg:,} numbers (no biases); 24 plain layers: {24 * plain:,}")
for g, u in [(1, 2), (-1, 2), (3, 0.5)]:
    print(f"4. SwiGLU gate {g}, up {u}: silu({g}) = {silu(np.array(g, float)):.3f}, times {u} = {silu(np.array(g, float)) * u:.3f}")
C = W1 @ W2; c = b1 @ W2
mat = np.array([-0.335, 0.703])
print(f"5. without the bend: table W1 W2 = {C.tolist()}, constant b1 W2 = {r(c)}; mat -> {r(mat @ C + c)} (Lab 12's Breakage 1 printed [1.167, -0.983])")
gm = gelu(mat @ W1 + b1) @ W2
print(f"6. hand block with GELU: mat adds {r(gm)} against ReLU's {r(block(mat))}; matches {r(mat @ W1 + b1)}, GELU activations {r(gelu(mat @ W1 + b1))}")

print("\n9. GPT-2 small: each layer's feed-forward output replaced by its mean over the text (mean removal), "
      "against zero removal; change in loss")
from _gpt2run import weights, norm, softmax, token_losses, H, D, L
from _gpt2run import gelu as ggelu
from _bpe import encode
from tiny.corpus import load_text
Wt = weights()
alice = load_text()
glass = open(os.path.join(HERE, "..", "data", "looking_glass_gutenberg_raw.txt"), encoding="utf-8").read()
def opening(txt, marker, n=1024):
    i = txt.index(marker); return [50256] + encode(txt[i:i + 12000].replace("\r", ""))[:n - 1]
def forward(ids, drop=None, how="zero", means=None):
    x = Wt["wte.weight"][ids] + Wt["wpe.weight"][:len(ids)]; n = len(ids); outs = {}
    mask = np.tril(np.ones((n, n), dtype=bool))
    for l in range(L):
        p = f"h.{l}."
        a = norm(x, Wt[p + "ln_1.weight"], Wt[p + "ln_1.bias"])
        q, k, v = (t.reshape(n, H, D).transpose(1, 0, 2) for t in np.split(a @ Wt[p + "attn.c_attn.weight"] + Wt[p + "attn.c_attn.bias"], 3, axis=-1))
        s = np.where(mask, q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(D)), np.float32(-1e9))
        x = x + (softmax(s) @ v).transpose(1, 0, 2).reshape(n, H * D) @ Wt[p + "attn.c_proj.weight"] + Wt[p + "attn.c_proj.bias"]
        mm = norm(x, Wt[p + "ln_2.weight"], Wt[p + "ln_2.bias"])
        o = ggelu(mm @ Wt[p + "mlp.c_fc.weight"] + Wt[p + "mlp.c_fc.bias"]) @ Wt[p + "mlp.c_proj.weight"] + Wt[p + "mlp.c_proj.bias"]
        outs[l] = o.mean(0)
        if l == drop:
            o = np.zeros_like(o) if how == "zero" else np.broadcast_to(means[l], o.shape)
        x = x + o
    return x, outs
for name, ids in [("Alice", opening(alice, "Alice was beginning")), ("Looking-Glass", opening(glass, "One thing was certain"))]:
    xb, means = forward(ids); base = token_losses(xb, ids).mean()
    print(f"   {name}: baseline {base:.3f}")
    for l in range(L):
        z = token_losses(forward(ids, l, "zero")[0], ids).mean() - base
        mu = token_losses(forward(ids, l, "mean", means)[0], ids).mean() - base
        print(f"   layer {l:>2}: zero {z:+.3f} | mean {mu:+.3f}")
