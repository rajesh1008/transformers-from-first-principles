"""Chapter 34, 34.3 to 34.5: pruning GPT-2 small without retraining.  Bits per character on the held-out Carroll text of
Chapters 25 to 33 (3,422 GPT-2 tokens, windows of up to 1,024), 32-bit; 32-bit score 1.5640.
Part A  unstructured sparsity in the 48 linear maps: smallest |w| within each table, smallest |w| within each output
        channel, and Wanda's score |w| x (norm of the input channel over 16 x 256 calibration tokens) within each output
        channel; and a random-mask control; at 10 / 30 / 50 / 60 / 70 / 80 per cent of the weights zeroed.
Part B  structured: whole feed-forward hidden units removed (column of mlp.c_fc, its bias, row of mlp.c_proj), ranked by the
        product of the two norms, at 10 / 25 / 50 per cent of the units; against unstructured pruning of the two feed-forward
        maps at the same share of those two maps' weights.
CPU, one thread, seed 1865, 11 Nov 2026."""
from _common import *
import torch, math, time
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.load import GPT2
from tiny import quant as Q
from transformers import AutoTokenizer
from _pool import corpora
from tiny.tokenizer import load_bpe
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
tb = load_bpe(TINY_BPE); C = corpora()
def bpc(W=None):
    tot, ch = 0.0, 0
    with torch.no_grad():
        for a in range(0, len(g) - 1, 1024):
            w = g[a:a + 1024]
            if len(w) < 2: break
            lp = torch.log_softmax(Q.forward_q(gm, w, W).double(), -1)
            tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); ch += len(tk.decode(w[1:]))
    return tot / math.log(2) / ch
base = bpc(); print(f"32-bit: {base:.4f}", flush=True)
pool_ids = tk.encode(tb.decode([int(i) for i in C["pool"][100000:112000]]))
cal = [pool_ids[i * 256:(i + 1) * 256] for i in range(16)]
XtX, amax = Q.collect_inputs(gm, cal)
NAMES = [(l, n) for l in range(12) for n in Q.LINEARS]
rng = torch.Generator().manual_seed(SEED)
def mask_table(w, frac, how, xn=None):
    """w [in, out].  Returns pruned copy with `frac` of its weights zeroed."""
    if frac == 0: return w.clone()
    if how == "table":
        k = int(w.numel() * frac); thr = w.abs().flatten().kthvalue(k).values
        return w * (w.abs() > thr)
    if how == "random":
        return w * (torch.rand(w.shape, generator=rng) >= frac)
    s = w.abs() if how == "channel" else w.abs() * xn[:, None]          # score per weight
    k = int(w.shape[0] * frac)                                           # per output channel: prune k of the `in` weights
    idx = s.argsort(0)[:k]                                               # [k, out] smallest scores per column
    m = torch.ones_like(w); m.scatter_(0, idx, 0.0)
    return w * m
A = {"base": base}
print("\nPart A: unstructured sparsity in the 48 linear maps")
for how in ("table", "channel", "wanda", "random"):
    for frac in ((0.1, 0.3, 0.5, 0.6, 0.7, 0.8) if how != "random" else (0.3, 0.5)):
        W = {}
        for l, n in NAMES:
            xn = XtX[f"h.{l}.{n}"].diag().sqrt().float()
            W[f"h.{l}.{n}.weight"] = mask_table(gm.W[f"h.{l}.{n}.weight"], frac, how, xn)
        v = bpc(W); A[f"{how} {frac}"] = v
        print(f"  {how:8s} {int(frac * 100):2d}% zeroed: {v:.4f} ({v - base:+.4f})", flush=True)
B = {}
print("\nPart B: whole feed-forward units against single weights in the two feed-forward maps")
for frac in (0.1, 0.25, 0.5):
    W = {}
    for l in range(12):
        p = f"h.{l}.mlp."
        cf, cb, cp = gm.W[p + "c_fc.weight"], gm.W[p + "c_fc.bias"], gm.W[p + "c_proj.weight"]
        score = cf.norm(dim=0) * cp.norm(dim=1)                          # [3072] per hidden unit
        k = int(cf.shape[1] * frac); drop = score.argsort()[:k]
        a, b, c = cf.clone(), cb.clone(), cp.clone(); a[:, drop] = 0; b[drop] = 0; c[drop, :] = 0
        W[p + "c_fc.weight"], W[p + "c_fc.bias"], W[p + "c_proj.weight"] = a, b, c
    v = bpc(W); B[f"units {frac}"] = v
    print(f"  remove {int(frac * 100):2d}% of the 3,072 feed-forward units in every layer: {v:.4f} ({v - base:+.4f})", flush=True)
    W = {}
    for l in range(12):
        for n in ("mlp.c_fc", "mlp.c_proj"):
            W[f"h.{l}.{n}.weight"] = mask_table(gm.W[f"h.{l}.{n}.weight"], frac, "channel")
    v = bpc(W); B[f"weights {frac}"] = v
    print(f"  zero {int(frac * 100):2d}% of the weights of both feed-forward maps (smallest per output channel): {v:.4f} ({v - base:+.4f})", flush=True)
save_json("prune_gpt2", {"A": A, "B": B})
