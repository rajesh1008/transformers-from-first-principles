"""Chapter 33: quantizing GPT-2 small's weights (and activations) with tiny/quant.py, scored in bits per character.
Part A  by hand: one row of eight weights at 8 and 4 bits, then the same row with one outlier.
Part B  round-to-nearest, weights of the twelve layers' linear maps only: 8 / 6 / 4 / 3 / 2 bits x one scale per table /
        per output channel / per 128 inputs / per 32 inputs.  Bits per character on held-out Carroll (the text and
        1,024-token windows of Chapter 30, where 32-bit scored 1.5640).  Storage in MB including scales.
Part C  the token table (shared input and output) too.
Part D  which maps and which layers are fragile: one kind of map at a time, then one layer at a time, at 3 bits.
Part E  activations: the largest |value| per input channel, and int8 activations per tensor / per token / per token
        with outlier channels (calibrated |value| > 6) left in float, on 8-bit weights.
(Parts F and G, GPTQ and greedy text, are in quant_gptq.py.)
CPU, one thread, 32-bit arithmetic on dequantized tables (fake quantization), seed 1865, 11 Nov 2026."""
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
out = {}
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

print("Part A: one row of eight weights (layer 0, feed-forward input map, output channel 0, inputs 0 to 7)")
row = gm.W["h.0.mlp.c_fc.weight"][:8, 0].clone()
def show(r, bits):
    qm = 2 ** (bits - 1) - 1; s = float(r.abs().max()) / qm; ints = (r / s).round().clamp(-qm - 1, qm)
    deq = ints * s
    print(f"  {bits}-bit: scale {s:.6f}; integers {ints.int().tolist()}; largest error {float((deq - r).abs().max()):.6f}; "
          f"mean |error| {float((deq - r).abs().mean()):.6f}")
print(f"  weights {[round(float(x), 4) for x in row]}")
show(row, 8); show(row, 4)
row2 = row.clone(); row2[3] = 1.0
print(f"  the same row with weight 3 set to 1.0: {[round(float(x), 4) for x in row2]}")
show(row2, 8); show(row2, 4)

print("\nPart B: round-to-nearest on the 48 linear maps (85.0M numbers), symmetric; bits per character on held-out Carroll")
base = bpc(); print(f"  32-bit: {base:.4f} bits per character; {mb(32, 'tensor'):.1f} MB as stored here (all tables 32-bit)", flush=True)
B = {"32": base}
for bits in (8, 6, 4, 3, 2):
    for gr in ("tensor", None, 128, 32):
        if bits == 8 and gr in (128, 32): continue
        v = bpc(qall(bits, gr)); B[f"{bits} {gr}"] = v
        print(f"  {bits}-bit {gname(gr):20s}: {v:8.4f} ({v - base:+.4f}); {mb(bits, gr):6.1f} MB", flush=True)
for bits in (4, 3):
    v = bpc(qall(bits, 128, sym=False)); B[f"{bits} 128 asym"] = v
    print(f"  {bits}-bit per 128 inputs, with zero point: {v:8.4f} ({v - base:+.4f}); {mb(bits, 128, False):6.1f} MB", flush=True)
out["B"] = B; save_json("quant_gpt2", out)

print("\nPart C: the token table as well (it is also the output map)")
wte = gm.W["wte.weight"]
for bits, gr in ((8, None), (4, 128), (3, 128)):
    W = qall(4, 128); W["wte.weight"] = Q.quantize(wte.t().contiguous(), bits, gr).t().contiguous()
    v = bpc(W); out.setdefault("C", {})[f"{bits} {gr}"] = v
    print(f"  maps 4-bit per 128 + token table {bits}-bit {gname(gr)}: {v:.4f} ({v - base:+.4f}); against maps alone {B['4 128']:.4f}", flush=True)
save_json("quant_gpt2", out)

print("\nPart D: 3-bit per output channel, one kind of map at a time, then one layer at a time")
D = {}
for n in Q.LINEARS:
    v = bpc(qall(3, None, [f"h.{l}.{n}.weight" for l in range(12)])); D[n] = v
    print(f"  only {n:12s} (12 maps): {v:.4f} ({v - base:+.4f})", flush=True)
for l in range(12):
    v = bpc(qall(3, None, [f"h.{l}.{n}.weight" for n in Q.LINEARS])); D[f"layer {l}"] = v
    print(f"  only layer {l:2d} (4 maps): {v:.4f} ({v - base:+.4f})", flush=True)
out["D"] = D; save_json("quant_gpt2", out)

print("\nPart E: activations entering the linear maps, calibrated on 16 x 256 tokens of the Carroll pool")
pool_ids = tk.encode(tt.decode([int(i) for i in C["pool"][100000:112000]]))
cal = [pool_ids[i * 256:(i + 1) * 256] for i in range(16)]
XtX, amax = Q.collect_inputs(gm, cal)
for n in Q.LINEARS:
    mx = [float(amax[f"h.{l}.{n}"].max()) for l in range(12)]; md = [float(amax[f"h.{l}.{n}"].median()) for l in range(12)]
    big = [int((amax[f"h.{l}.{n}"] > 6).sum()) for l in range(12)]
    print(f"  {n:12s}: largest |input| per layer {[round(x, 1) for x in mx]}")
    print(f"  {'':12s}  median channel's largest {[round(x, 2) for x in md]}; channels above 6: {big}")
top = torch.stack([amax[f"h.{l}.attn.c_attn"] for l in range(12)]).amax(0)
print(f"  attention-input channels with the largest |value| over all layers: {top.topk(5).indices.tolist()} ({[round(float(x), 1) for x in top.topk(5).values]})")
keep = {k: (v > 6).nonzero().flatten() for k, v in amax.items()}
W8 = qall(8, None)
E = {}
for lab, act in (("8-bit weights, 32-bit activations", None),
                 ("int8 activations, one scale per tensor", lambda x, n: Q.quant_act(x, 8, "tensor")),
                 ("int8 activations, one scale per token", lambda x, n: Q.quant_act(x, 8, "token")),
                 ("int8 per token, outlier channels (> 6) in float", lambda x, n: Q.quant_act(x, 8, "token", keep[n])),
                 ("int4 activations, per token", lambda x, n: Q.quant_act(x, 4, "token")),
                 ("int4 per token, outlier channels in float", lambda x, n: Q.quant_act(x, 4, "token", keep[n]))):
    v = bpc(W8, act); E[lab] = v
    print(f"  {lab:50s}: {v:8.4f} ({v - base:+.4f})", flush=True)
nk = sum(len(v) for v in keep.values()); tot = sum(len(v) for v in amax.values())
print(f"  outlier channels kept in float: {nk} of {tot} input channels ({100 * nk / tot:.2f} per cent)")
out["E"] = E; save_json("quant_gpt2", out)
