"""Chapter 33, 33.9 and 33.10.
Part H  does an 8-bit table run faster here?  One GPT-2-sized map (768 -> 3,072 and back) as a 32-bit nn.Linear, a
        BF16 matmul, and PyTorch's dynamically quantized int8 Linear (int8 weights, activations quantized per call),
        for 1 and 64 rows; best of 5 timings of 200 calls each.
Part I  Tiny small: Chapter 24's checkpoint (the one Chapters 25 and 29 scored), every 2-D table rounded to 8 / 4 / 3 /
        2 bits per output channel and per 32 inputs, with and without the token table; bits per character on held-out
        Carroll in windows of 129 tokens, as Chapter 29 measured it (1.9812); storage in bytes.
CPU, one thread, seed 1865, 11 Nov 2026."""
from _common import *
import torch, time, math
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny import quant as Q
from tiny.eval import load
from _pool import corpora
print("Part H: one 768 x 3,072 map and its 3,072 x 768 partner, 200 calls, best of 5")
import torch.ao.nn.quantized.dynamic as nnqd
H = {}
for fin, fout in ((768, 3072), (3072, 768)):
    lin = torch.nn.Linear(fin, fout); qlin = nnqd.Linear(fin, fout)
    w = lin.weight.detach()
    qlin.set_weight_bias(torch.quantize_per_tensor(w, float(w.abs().max()) / 127, 0, torch.qint8), lin.bias.detach())
    wb = lin.weight.detach().t().contiguous().bfloat16()
    for rows in (1, 64):
        x = torch.randn(rows, fin)
        def tm(f):
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(200): f()
                ts.append((time.perf_counter() - t0) / 200)
            return min(ts)
        a = tm(lambda: lin(x)); b = tm(lambda: x.bfloat16() @ wb); c = tm(lambda: qlin(x))
        H[f"{fin}x{fout} r{rows}"] = dict(fp32=a, bf16=b, int8=c)
        print(f"  {fin:4d} -> {fout:4d}, {rows:2d} rows: 32-bit {a * 1e6:8.1f} us | BF16 {b * 1e6:8.1f} us ({a / b:4.2f} x) | "
              f"int8 dynamic {c * 1e6:8.1f} us ({a / c:4.2f} x)", flush=True)
save_json("tiny_quant", {"H": H})

print("\nPart I: Tiny, every 2-D table rounded; bits per character on held-out Carroll, windows of 129")
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok(); ids = [int(i) for i in corpora()["held_b"]]
orig = {k: v.clone() for k, v in tt.t.items()}
def bpc():
    tot, chars = 0.0, 0
    for a in range(0, len(ids) - 1, 129):
        w = ids[a:a + 129]
        if len(w) < 2: break
        with torch.no_grad(): lp = torch.log_softmax(tt.scores(w[:-1]).double(), -1)
        tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); chars += len(t.decode(w[1:]))
    return tot / math.log(2) / chars
mats = [k for k, v in orig.items() if v.dim() == 2 and k not in ("E", "P")]
nmat = sum(orig[k].numel() for k in mats); nall = sum(v.numel() for v in orig.values())
base = bpc(); print(f"  32-bit: {base:.4f}; {nall:,} numbers, {nall * 4:,} bytes; linear maps {nmat:,} numbers, token table {orig['E'].numel():,}")
I = {"32": base}
for bits in (8, 4, 3, 2):
    for gr in (None, 32):
        for withE in (False, True):
            for k in orig: tt.t[k] = orig[k].clone()
            for k in mats: tt.t[k] = Q.quantize(orig[k], bits, gr)
            sb = sum(Q.storage_bits(tuple(orig[k].shape), bits, gr) for k in mats)
            if withE:
                tt.t["E"] = Q.quantize(orig["E"].t().contiguous(), bits, gr).t().contiguous(); sb += Q.storage_bits((128, 2048), bits, gr)
            rest = sum(v.numel() for k, v in orig.items() if k not in mats and not (withE and k == "E")) * 32
            v = bpc(); byt = (sb + rest) / 8
            I[f"{bits} {gr} {withE}"] = dict(bpc=v, bytes=byt)
            print(f"  {bits}-bit {'per output channel' if gr is None else 'per 32 inputs':18s} {'maps + token table' if withE else 'maps only':18s}: "
                  f"{v:.4f} ({v - base:+.4f}); {byt:,.0f} bytes ({nall * 4 / byt:4.1f} x smaller)", flush=True)
save_json("tiny_quant", {"H": H, "I": I})
