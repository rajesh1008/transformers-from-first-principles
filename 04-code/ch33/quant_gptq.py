"""Chapter 33, 33.7 and 33.8, continuing quant_gpt2.py (same set-up, same held-out text and windows).
Part F  GPTQ against round-to-nearest at 4 and 3 bits (group 128) and 3 bits per channel, calibrated on 16 x 256 tokens
        of the Carroll pool; then 3-bit GPTQ per 128 calibrated on Through the Looking-Glass and on random token ids.
Part G  greedy text: 64 new tokens after four held-out prompts, positions identical to 32-bit and the first difference.
CPU, one thread, 32-bit arithmetic on dequantized tables, seed 1865, 11 Nov 2026.  Reads outputs/quant_gpt2.json (Part B)."""
from _common import *
import torch, math, time, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.load import GPT2
from tiny import quant as Q
from transformers import AutoTokenizer
from _pool import corpora
from tiny.tokenizer import load_bpe
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
tt = load_bpe(TINY_BPE); C = corpora()
out = {}; B = load_json("quant_gpt2")["B"]; base = B["32"]
def bpc(W=None, act=None):
    tot, ch = 0.0, 0
    with torch.no_grad():
        for a in range(0, len(g) - 1, 1024):
            w = g[a:a + 1024]
            if len(w) < 2: break
            lp = torch.log_softmax(Q.forward_q(gm, w, W, act).double(), -1)
            tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); ch += len(tk.decode(w[1:]))
    return tot / math.log(2) / ch
NAMES = [f"h.{l}.{n}.weight" for l in range(12) for n in Q.LINEARS]
def qall(bits, group, names=NAMES, sym=True):
    return {n: Q.quantize(gm.W[n], bits, group, sym) for n in names}
def mb(bits, group, sym=True, names=NAMES):
    lin = sum(Q.storage_bits(tuple(gm.W[n].shape), bits, group, sym) for n in names)
    rest = sum(v.numel() for k, v in gm.W.items() if k not in names and not k.endswith("attn.bias")) * 32
    return (lin + rest) / 8 / 1e6
gname = lambda gr: {"tensor": "one scale per table", None: "per output channel", 128: "per 128 inputs", 32: "per 32 inputs"}[gr]

pool_ids = tk.encode(tt.decode([int(i) for i in C["pool"][100000:112000]]))
cal = [pool_ids[i * 256:(i + 1) * 256] for i in range(16)]
XtX, amax = Q.collect_inputs(gm, cal)
W8 = qall(8, None)
print("Part F: GPTQ against round-to-nearest (calibration: 16 x 256 tokens of the Carroll pool)")
def gptq_all(bits, group, H):
    return {f"h.{l}.{n}.weight": Q.gptq(gm.W[f"h.{l}.{n}.weight"], H[f"h.{l}.{n}"], bits, group) for l in range(12) for n in Q.LINEARS}
F = {}
for bits, gr in ((4, 128), (3, 128), (3, None)):
    t0 = time.perf_counter(); Wg = gptq_all(bits, gr if gr else 768 * 4, XtX); tq = time.perf_counter() - t0
    v = bpc(Wg); F[f"{bits} {gr}"] = v
    rtn = B[f"{bits} {gr}"]
    print(f"  {bits}-bit {gname(gr):20s}: round-to-nearest {rtn:.4f} ({rtn - base:+.4f}); GPTQ {v:.4f} ({v - base:+.4f}); GPTQ took {tq:.0f} s", flush=True)
    if (bits, gr) == (3, 128): W3g = Wg
glass_ids = tk.encode(tt.decode([int(i) for i in C["glass"][:12000]]))
calg = [glass_ids[i * 256:(i + 1) * 256] for i in range(16)]
rng = np.random.default_rng(SEED); calr = [rng.integers(0, 50257, 256).tolist() for _ in range(16)]
for lab, cs in (("Through the Looking-Glass", calg), ("random token ids", calr)):
    Hx, _ = Q.collect_inputs(gm, cs); v = bpc(gptq_all(3, 128, Hx)); F[f"3 128 cal {lab}"] = v
    print(f"  3-bit per 128 inputs, GPTQ calibrated on {lab}: {v:.4f} ({v - base:+.4f})", flush=True)
out["F"] = F; save_json("quant_gptq", out)

print("\nPart G: greedy, 64 new tokens after four held-out prompts of 64 tokens; agreement with 32-bit")
prompts = [g[i:i + 64] for i in (800, 1600, 2400, 3000)]
def greedy_q(p, W):
    ids = list(p)
    with torch.no_grad():
        for _ in range(64): ids.append(int(Q.forward_q(gm, ids, W)[-1].argmax()))
    return ids[64:]
ref = [greedy_q(p, None) for p in prompts]
G = {}
for lab, W in (("8-bit per channel", W8), ("4-bit per 128, round-to-nearest", qall(4, 128)), ("3-bit per 128, round-to-nearest", qall(3, 128)),
               ("3-bit per 128, GPTQ", W3g), ("2-bit per 32, round-to-nearest", qall(2, 32))):
    o = [greedy_q(p, W) for p in prompts]
    first = [next((i for i in range(64) if a[i] != b[i]), 64) for a, b in zip(o, ref)]
    same = sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(o, ref))
    G[lab] = dict(first=first, same=same)
    print(f"  {lab:34s}: tokens before first difference {first}; positions equal {same}/256", flush=True)
    if lab.startswith("3-bit per 128, round"):
        print(f"    prompt 1, 32-bit: {tk.decode(ref[0][:24])!r}\n    prompt 1, {lab}: {tk.decode(o[0][:24])!r}")
out["G"] = G; save_json("quant_gptq", out)
