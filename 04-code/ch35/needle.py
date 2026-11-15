"""Chapter 35, 35.5: a pass key hidden in a haystack.  A five-digit key is written once, as the sentence " The pass key is
NNNNN.", at a chosen depth inside a stretch of Through the Looking-Glass; the input ends with "\nWhat is the pass key? The pass
key is" and we read the model's table at each token of the true key (teacher forcing): exact = every answer token is the model's top
choice (SmolLM2 writes ' 32008' as a bare space and five digits; the space is fed but not scored, see needle_probe.py); logp = mean log-probability per answer token.  Six keys per cell (seed 1865).  The same haystack text for every key.
  python3 needle.py MODEL LEN      MODEL = smol | gpt2      LEN = haystack tokens
Conditions: depth 0, 0.25, 0.5, 0.75, 1.0; 'none' (no key written: the model must guess); with LEN 2048 also two rivals:
a second sentence with a different key at depth 0.25 (true key at 0.75) and the reverse.  CPU, one thread, 32-bit."""
from _common import *
import sys, time, random, torch
torch.set_num_threads(1)
from transformers import AutoTokenizer
from tiny.load import Llama, GPT2
from longfwd import token_losses
MODEL, LEN = sys.argv[1], int(sys.argv[2]); NK = int(os.environ.get('NK', '6'))
if MODEL == "smol":
    tk = AutoTokenizer.from_pretrained(CKPT("smollm2")); m = Llama.from_dir(CKPT("smollm2"))
else:
    tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); m = GPT2.from_dir(CKPT("gpt2"))
E = lambda s: tk.encode(s, add_special_tokens=False)
hay = E(glass_text())[2000:2000 + LEN]
rng = random.Random(SEED); keys = [str(rng.randint(10000, 99999)) for _ in range(NK)]
rivals = [str(rng.randint(10000, 99999)) for _ in range(NK)]
Q = E("\nWhat is the pass key? The pass key is")
def table(ids):
    with torch.no_grad():
        if MODEL == "smol":
            return full_logits(ids)
        return torch.log_softmax(m.scores(ids).double(), -1)
def full_logits(ids):
    # last 8 positions only are needed; run the blocked forward and keep the final hidden rows
    import math
    W, T, hd, Hq, Hkv = m.W, len(ids), m.hd, m.Hq, m.Hkv
    inv = 1.0 / m.theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None]; c, s = torch.cos(ang), torch.sin(ang)
    rope = lambda x: torch.cat([x[..., :hd // 2] * c - x[..., hd // 2:] * s, x[..., :hd // 2] * s + x[..., hd // 2:] * c], -1)
    x = m.emb[torch.as_tensor(ids)]
    for l in range(m.L):
        p = f"model.layers.{l}."
        a = m._rms(x, W[p + "input_layernorm.weight"])
        q = rope((a @ W[p + "self_attn.q_proj.weight"].T).view(T, Hq, hd).transpose(0, 1))
        k = rope((a @ W[p + "self_attn.k_proj.weight"].T).view(T, Hkv, hd).transpose(0, 1))
        v = (a @ W[p + "self_attn.v_proj.weight"].T).view(T, Hkv, hd).transpose(0, 1)
        kk, vv = k.repeat_interleave(Hq // Hkv, 0), v.repeat_interleave(Hq // Hkv, 0)
        o = torch.empty(Hq, T, hd)
        for b0 in range(0, T, 512):
            b1 = min(T, b0 + 512)
            sc = q[:, b0:b1] @ kk[:, :b1].transpose(-1, -2) / math.sqrt(hd)
            o[:, b0:b1] = torch.softmax(sc.masked_fill(~torch.ones(b1 - b0, b1, dtype=torch.bool).tril(b0), float("-inf")), -1) @ vv[:, :b1]
        x = x + o.transpose(0, 1).reshape(T, m.d) @ W[p + "self_attn.o_proj.weight"].T
        h = m._rms(x, W[p + "post_attention_layernorm.weight"])
        x = x + (torch.nn.functional.silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
    h = m._rms(x[-8:], W["model.norm.weight"])
    out = torch.full((T, m.out.shape[0]), float("nan"), dtype=torch.float64)
    out[-8:] = torch.log_softmax((h @ m.out.T).double(), -1)
    return out
ENDS = [i + 1 for i, t in enumerate(hay) if tk.decode([t]).rstrip().endswith((".", "!", "?", "\n")) or tk.decode([t]).endswith("\n")]
def snap(at):
    """Insert only after a token that ends a sentence or a line, the first such point at or after the target offset
    (or the last one before the end), so the key sentence never splits a word or a sentence."""
    if at <= 0: return 0
    later = [e for e in ENDS if e >= at]
    return later[0] if later else len(hay)
def build(depth_items):
    """depth_items: list of (depth, key); sentences inserted at the first sentence end at or after depth x LEN, later ones first."""
    ids = list(hay)
    for d, k in sorted(depth_items, key=lambda t: -t[0]):
        at = snap(int(d * LEN)); ids[at:at] = E(f" The pass key is {k}.")
    return ids + Q
def score(ids, k):
    A = E(f" {k}")
    full = ids + A
    lp = table(full)
    n = len(A); pos = list(range(len(full) - n - 1, len(full) - 1))
    if tk.decode([A[0]]).strip() == "":            # SmolLM2 writes " 32008" as a bare space then five digits: score the digits only
        pos = pos[1:]; n -= 1
    lps = [float(lp[p, full[p + 1]]) for p in pos]; hits = [int(lp[p].argmax()) == full[p + 1] for p in pos]
    return sum(lps) / n, all(hits)
R = {"model": MODEL, "len": LEN, "keys": keys, "answer tokens": [len(E(f' {k}')) for k in keys], "cells": {}}
print(f"{MODEL}, haystack {LEN} tokens, keys {keys}; answer tokens per key {R['answer tokens']}", flush=True)
conds = [("none", [])] + [(f"depth {d}", [(d, None)]) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
if LEN == 2048: conds += [("rival at 0.25, key at 0.75", "rb"), ("key at 0.25, rival at 0.75", "ra")]
t0 = time.perf_counter()
for name, spec in conds:
    ex, lp_sum, rex = 0, 0.0, 0
    for k, r in zip(keys, rivals):
        if spec == "rb": items = [(0.25, r), (0.75, k)]
        elif spec == "ra": items = [(0.25, k), (0.75, r)]
        else: items = [(d, k) for d, _ in spec]
        ids = build(items)
        lp, hit = score(ids, k); ex += hit; lp_sum += lp
        if spec in ("rb", "ra"): rex += score(ids, r)[1]
    cell = dict(exact=ex, n=len(keys), logp=lp_sum / len(keys)); 
    if spec in ("rb", "ra"): cell["rival exact"] = rex
    R["cells"][name] = cell
    print(f"  {name:28s} exact {ex} of {len(keys)}; mean log-prob per answer token {lp_sum / len(keys):7.3f}" + (f"; rival's key exact {rex} of {len(keys)}" if spec in ("rb", "ra") else "") + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
save_json(f"needle_{MODEL}_{LEN}", R)
