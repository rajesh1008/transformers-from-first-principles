"""Chapter 33, 33.3: why rounding GPT-2's token table hurts.  The table's largest |value| per dimension; the final
norm's output per dimension on held-out text; then the table rounded to 4 bits per 128 for the input lookup only, for
the output scores only, and for both (the 48 maps left in 32 bits), and the same with the largest dimensions kept in
32 bits.  Bits per character on held-out Carroll as in quant_gpt2.py.  CPU, one thread, 11 Nov 2026."""
from _common import *
import torch, math
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny import quant as Q
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
E = gm.W["wte.weight"]
def forward_split(ids, Ein, Eout):
    W = gm.W; saved = W["wte.weight"]
    W["wte.weight"] = Ein
    try:
        T = len(ids); H, d = gm.H, gm.d; hd = d // H
        x = Ein[torch.as_tensor(ids)] + W["wpe.weight"][:T]
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        for l in range(gm.L):
            p = f"h.{l}."
            q, k, v = (gm._ln(x, p + "ln_1") @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).split(d, -1)
            q, k, v = (z.view(T, H, hd).transpose(0, 1) for z in (q, k, v))
            s = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~mask, float("-inf"))
            o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
            x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
            f = gm._gelu(gm._ln(x, p + "ln_2") @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
            x = x + f @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        h = gm._ln(x, "ln_f")
        return h @ Eout.T, h
    finally:
        W["wte.weight"] = saved
def bpc(Ein, Eout):
    tot, ch = 0.0, 0; hs = []
    with torch.no_grad():
        for a in range(0, len(g) - 1, 1024):
            w = g[a:a + 1024]
            if len(w) < 2: break
            S, h = forward_split(w, Ein, Eout); hs.append(h)
            lp = torch.log_softmax(S.double(), -1)
            tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); ch += len(tk.decode(w[1:]))
    return tot / math.log(2) / ch, torch.cat(hs)
base, h = bpc(E, E)
am = E.abs().amax(0); hm = h.abs().amax(0)
print(f"32-bit: {base:.4f}")
print(f"token table, largest |value| per dimension: top 5 dims {am.topk(5).indices.tolist()} {[round(float(x), 3) for x in am.topk(5).values]}; median {float(am.median()):.3f}")
print(f"final norm output on held-out text, largest |value| per dimension: top 5 dims {hm.topk(5).indices.tolist()} {[round(float(x), 1) for x in hm.topk(5).values]}; median {float(hm.median()):.2f}")
def q4(T, keep=None):
    Qt = Q.quantize(T.t().contiguous(), 4, 128).t().contiguous()
    if keep is not None: Qt[:, keep] = T[:, keep]
    return Qt
E4 = q4(E)
for lab, Ein, Eout in (("4-bit table for the input lookup only", E4, E), ("4-bit table for the output scores only", E, E4), ("4-bit table for both", E4, E4)):
    v, _ = bpc(Ein, Eout); print(f"  {lab:42s}: {v:.4f} ({v - base:+.4f})", flush=True)
for n in (2, 8):
    keep = hm.topk(n).indices
    E4k = q4(E, keep); v, _ = bpc(E4k, E4k)
    print(f"  4-bit table, final-norm top {n} dims {keep.tolist()} kept in 32 bits: {v:.4f} ({v - base:+.4f})", flush=True)
keep = am.topk(8).indices; E4k = q4(E, keep); v, _ = bpc(E4k, E4k)
print(f"  4-bit table, table's own top 8 dims {keep.tolist()} kept in 32 bits: {v:.4f} ({v - base:+.4f})", flush=True)
v, _ = bpc(Q.quantize(E.t().contiguous(), 8, None).t().contiguous(), Q.quantize(E.t().contiguous(), 8, None).t().contiguous())
print(f"  8-bit table (one scale per token row), maps in 32 bits: {v:.4f} ({v - base:+.4f})")
