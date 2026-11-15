"""Chapter 48, 48.8 to 48.10: evaluate, quantise, serve and sample one checkpoint.  Usage: python3 eval48.py ckpt/main_step2000.pt
CPU, one thread, 32-bit.  Writes outputs/eval_<checkpoint name>.json.
  1. Bits per character on all of The Time Machine (held out) and all of Through the Looking-Glass, in non-overlapping windows of
     256 predicted tokens (each window starts with its first token as context only).
  2. The same after rounding every 2-D table to 8 bits (symmetric, one scale per output row, as in Chapters 33 and 46) and back.
  3. Generation with a KV cache (Chapter 31's loop written for this model): the cached logits are checked against the full
     forward pass, then 3 seeded samples of 120 tokens at temperature 0.8 from the running sentence, timed in tokens per second."""
import os, sys, json, time, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from tiny.capstone.model import Capstone
from tokenizers import Tokenizer
torch.set_num_threads(1)
CK = sys.argv[1]; NAME = os.path.splitext(os.path.basename(CK))[0]
C = os.path.join(HERE, "corpus"); tk = Tokenizer.from_file(os.path.join(C, "tok8192.json"))
ck = torch.load(CK, map_location="cpu"); cfg = ck["cfg"]
m = Capstone(**cfg); m.load_state_dict(ck["state"]); m.eval(); CTX = cfg["ctx"]

def bpc(name):
    a = torch.from_numpy(np.fromfile(os.path.join(C, f"{name}.u16"), dtype=np.uint16).astype(np.int64))
    k = (len(a) - 1) // CTX; w = torch.stack([a[i * CTX: i * CTX + CTX + 1] for i in range(k)])
    chars = sum(len(tk.decode(r[1:].tolist())) for r in w); nats = 0.0
    with torch.no_grad():
        for i in range(0, k, 16):
            b = w[i:i + 16]
            nats += float(F.cross_entropy(m(b[:, :-1]).reshape(-1, cfg["vocab"]), b[:, 1:].reshape(-1), reduction="sum"))
    return dict(windows=k, tokens=k * CTX, chars=chars, loss=nats / (k * CTX), bpc=nats / math.log(2) / chars)

R = dict(checkpoint=NAME, step=ck.get("step"), params=m.count())
R["fp32"] = {k: bpc(k) for k in ("heldout", "glass")}
print(f"{NAME} (step {ck.get('step')}, {m.count():,} numbers)")
for k, v in R["fp32"].items(): print(f"  32-bit {k:8s} {v['windows']} windows, {v['chars']:,} characters: loss {v['loss']:.4f}, {v['bpc']:.4f} bits per character")
keep = {k: v.clone() for k, v in m.state_dict().items()}; maxerr = 0.0; nq = 0
with torch.no_grad():
    for n, p in m.named_parameters():
        if p.dim() == 2:
            s = p.abs().amax(1, keepdim=True) / 127.0; s[s == 0] = 1.0
            q = torch.clamp(torch.round(p / s), -127, 127) * s; maxerr = max(maxerr, float((q - p).abs().max())); p.copy_(q); nq += p.numel()
R["int8"] = {k: bpc(k) for k in ("heldout", "glass")}; R["int8_numbers"] = nq; R["int8_maxerr"] = maxerr
R["bytes_fp32"] = 4 * m.count(); R["bytes_int8"] = nq + 4 * (m.count() - nq) + 4 * sum(p.shape[0] for p in m.parameters() if p.dim() == 2)
for k, v in R["int8"].items(): print(f"  8-bit  {k:8s} loss {v['loss']:.4f}, {v['bpc']:.4f} bits per character (change {v['bpc'] - R['fp32'][k]['bpc']:+.4f})")
print(f"  8-bit: {nq:,} numbers rounded, largest change {maxerr:.2e}; weights {R['bytes_fp32']:,} bytes at 32 bits, {R['bytes_int8']:,} at 8 bits with scales")
m.load_state_dict(keep)

def step_cached(ids, cache):
    """One position (or a prompt) through the model, keeping each layer's keys and values."""
    pos0 = cache[0][0].shape[2] if cache else 0; n = ids.shape[1]
    x = m.E(ids) + m.P(torch.arange(pos0, pos0 + n)); new = []
    for li, b in enumerate(m.blocks):
        d = x.shape[-1]; q, k, v = b.qkv(b.ln1(x)).split(d, -1)
        q, k, v = (t.view(1, n, b.h, d // b.h).transpose(1, 2) for t in (q, k, v))
        if cache: k = torch.cat([cache[li][0], k], 2); v = torch.cat([cache[li][1], v], 2)
        new.append((k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(pos0 == 0)).transpose(1, 2).reshape(1, n, d)
        x = x + b.o(a); x = x + b.down(F.gelu(b.up(b.ln2(x)), approximate="tanh"))
    return m.lnf(x) @ m.E.weight.T, new

PROMPT = "the cat sat on the mat"; pid = tk.encode(PROMPT).ids
with torch.no_grad():
    ref = torch.tensor([pid + tk.encode(" and the dog sat by the fire").ids])
    full = m(ref)[0]; lg, cache = step_cached(ref[:, :len(pid)], None); outs = [lg[0]]
    for t in range(len(pid), ref.shape[1]):
        lg, cache = step_cached(ref[:, t:t + 1], cache); outs.append(lg[0])
    diff = float((torch.cat(outs) - full).abs().max())
R["cache_maxdiff"] = diff; print(f"  KV cache against full forward pass, {ref.shape[1]} positions: largest logit difference {diff:.2e}")
R["samples"] = []; NEW = 120
for seed in (1, 2, 3):
    g = torch.Generator().manual_seed(seed); ids = list(pid)
    with torch.no_grad():
        t0 = time.time(); lg, cache = step_cached(torch.tensor([ids]), None)
        for _ in range(NEW):
            p = torch.softmax(lg[0, -1] / 0.8, -1); nxt = int(torch.multinomial(p, 1, generator=g)); ids.append(nxt)
            if len(ids) >= CTX: break
            lg, cache = step_cached(torch.tensor([[nxt]]), cache)
        dt = time.time() - t0
    text = tk.decode(ids); R["samples"].append(dict(seed=seed, text=text, new_tokens=len(ids) - len(pid), seconds=dt, tps=(len(ids) - len(pid)) / dt))
    print(f"  seed {seed}: {len(ids) - len(pid)} tokens in {dt:.2f} s ({(len(ids) - len(pid)) / dt:.0f} tokens/s)\n    {text!r}")
json.dump(R, open(os.path.join(HERE, "outputs", f"eval_{NAME}.json"), "w"), indent=1)
