"""Chapter 30, sections 30.5 to 30.7: prefix (prompt) caching, and compressing the cache, on GPT-2 small.
Part F: a 600-token shared prefix cached once and reused for three different 16-token endings (time and equality).
Part G: the cache's quality cost.  Keys and values rounded to 8, 4 and 2 bits, and evicted (keep the first 4 positions
and the most recent ones, or only recent ones), simulated in one full pass per window with the same arithmetic a
compressed cache would give; bits per character on Chapter 29's held-out Carroll in 1,024-token windows.
CPU, one thread, 32-bit; timings 11 Nov 2026, best of 3."""
from _common import *
import torch, time, math
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny import cache as C
from transformers import AutoTokenizer
from _pool import corpora
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2"))
c = corpora(); tt = tok()
text = tt.decode([int(i) for i in c["held_b"]]); g = tk.encode(text)
out = {}
def best(f, k=3):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)

print("Part F: one shared 600-token prefix, three different 16-token endings")
prefix = g[:600]; ends = [g[1000 + 100 * i:1016 + 100 * i] for i in range(3)]
with torch.no_grad():
    pc = C.new_cache(gm); gm.forward(prefix, pc, 0)
    t_prefix = best(lambda: gm.forward(prefix, C.new_cache(gm), 0))
    def reuse(e):
        cc = [(k.clone(), v.clone()) for k, v in pc]
        return gm.forward(e, cc, 600)
    def full(e):
        return gm.forward(prefix + e)[600:]
    d = max(float((torch.log_softmax(reuse(e).double(), -1) - torch.log_softmax(full(e).double(), -1)).abs().max()) for e in ends)
    t_full = sum(best(lambda: full(e)) for e in ends); t_reuse = sum(best(lambda: reuse(e)) for e in ends)
print(f"  prefix pass {t_prefix:.2f} s; three endings from scratch {t_full:.2f} s; three endings on the cached prefix {t_reuse:.3f} s "
      f"(plus one prefix pass {t_prefix:.2f} s); largest |log-prob difference| {d:.2e}")
print(f"  the cached prefix holds {C.cache_bytes(pc):,} bytes (32-bit)")
out["F"] = dict(prefix=t_prefix, full=t_full, reuse=t_reuse, diff=d, bytes=C.cache_bytes(pc))

def quant(x, bits, axis):
    if bits is None: return x
    q = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=axis, keepdim=True).clamp_min(1e-8) / q
    return torch.round(x / s).clamp(-q, q) * s

def forward_c(ids, kbits=None, vbits=None, kaxis=-1, keep=None, sinks=0):
    """GPT2.forward with the keys and values rounded (per position over each head's 64 numbers if axis -1; per channel
    over positions if axis -2) and, if keep is set, each query seeing only the first `sinks` positions and the most recent
    keep - sinks positions: what a cache that evicted everything else would give."""
    W, H, d = gm.W, gm.H, gm.d; T = len(ids); hd = d // H
    x = W["wte.weight"][torch.as_tensor(ids)] + W["wpe.weight"][:T]
    m = torch.ones(T, T, dtype=torch.bool).tril()
    if keep is not None:
        i = torch.arange(T)[:, None]; j = torch.arange(T)[None]
        m &= (j < sinks) | (j > i - (keep - sinks))
    for l in range(gm.L):
        p = f"h.{l}."
        q, k, v = (gm._ln(x, p + "ln_1") @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).split(d, -1)
        q, k, v = (z.view(T, H, hd).transpose(0, 1) for z in (q, k, v))
        k, v = quant(k, kbits, kaxis), quant(v, vbits, -1)
        s = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~m, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        f = gm._gelu(gm._ln(x, p + "ln_2") @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + f @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    return gm._ln(x, "ln_f") @ W["wte.weight"].T

print("\nPart G: compressing the cache, bits per character on held-out Carroll (1,024-token windows; 10,919 characters)")
def bpc(**kw):
    tot, ch = 0.0, 0
    with torch.no_grad():
        for a in range(0, len(g) - 1, 1024):
            w = g[a:a + 1024]
            if len(w) < 2: break
            lp = torch.log_softmax(forward_c(w, **kw).double(), -1)
            tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); ch += len(tk.decode(w[1:]))
    return tot / math.log(2) / ch
cfgs = [("32-bit (no compression)", {}, 1.0),
        ("8-bit keys and values, per position", dict(kbits=8, vbits=8), 4.0),
        ("4-bit keys and values, per position", dict(kbits=4, vbits=4), 8.0),
        ("4-bit, keys per channel", dict(kbits=4, vbits=4, kaxis=-2), 8.0),
        ("2-bit keys and values, per position", dict(kbits=2, vbits=2), 16.0),
        ("2-bit, keys per channel", dict(kbits=2, vbits=2, kaxis=-2), 16.0),
        ("keep first 4 + recent 252 (256)", dict(keep=256, sinks=4), 4.0),
        ("keep recent 256 only", dict(keep=256, sinks=0), 4.0),
        ("keep first 4 + recent 60 (64)", dict(keep=64, sinks=4), 16.0),
        ("keep recent 64 only", dict(keep=64, sinks=0), 16.0)]
res = {}
for lab, kw, ratio in cfgs:
    b = bpc(**kw); res[lab] = dict(bpc=b, ratio=ratio)
    print(f"  {lab:<40} {b:.4f} bits per character; cache {ratio:4.1f} times smaller at 1,024 tokens (scales of the rounding not counted)", flush=True)
out["G"] = res
save_json("cache_more", out)
