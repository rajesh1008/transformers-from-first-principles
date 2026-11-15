"""Chapter 33, tick 25 additions (Exercises 11 and 10 of the chapter, run so that the text can use them).
Part J  Tiny's final norm output on held-out Carroll: largest |value| per dimension (as quant_table.py prints for GPT-2),
        and the table's own largest |value| per dimension.
Part K  GPTQ at 4 and 3 bits per 128 inputs calibrated on as many windows of 256 tokens of the Carroll pool as the slice gives,
        up to 64 (37 were available; quant_gptq.py used 16), scored on the same held-out text as quant_gptq.py.
CPU, one thread, seed 1865, 11 Nov 2026."""
from _common import *
import torch, math, time
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny import quant as Q
from tiny.eval import load
from tiny.load import GPT2
from transformers import AutoTokenizer
from _pool import corpora
from tiny.tokenizer import load_bpe

print("Part J: Tiny's final norm output, held-out Carroll windows of 129")
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
ids = [int(i) for i in corpora()["held_b"]]
print(f"  tied token table: {tt.config['tied']}; final norm: {tt.config['final_norm']}; d = {tt.t['E'].shape[1]}; layers {tt.config['n_layers']}")
hs = []
with torch.no_grad():
    for a in range(0, len(ids) - 1, 129):
        w = ids[a:a + 129]
        if len(w) < 2: break
        x = tt.t["E"][torch.as_tensor(w[:-1])] + tt.t["P"][:len(w) - 1]
        for l in range(tt.config["n_layers"]):
            x = x + tt._attention(tt._norm(x, f"h.{l}.ln_1"), l)
            x = x + tt._ffn(tt._norm(x, f"h.{l}.ln_2"), l)
        hs.append(tt._norm(x, "ln_f"))
h = torch.cat(hs); hm = h.abs().amax(0); E = tt.t["E"]; am = E.abs().amax(0)
print(f"  rows scored: {h.shape[0]}")
print(f"  final norm output, largest |value| per dimension: top 5 dims {hm.topk(5).indices.tolist()} {[round(float(x), 2) for x in hm.topk(5).values]}; median {float(hm.median()):.2f}")
print(f"  token table, largest |value| per dimension: top 5 dims {am.topk(5).indices.tolist()} {[round(float(x), 3) for x in am.topk(5).values]}; median {float(am.median()):.3f}")

print("\nPart K: GPTQ calibrated on 64 windows of 256 tokens (quant_gptq.py: 16)")
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
tb = load_bpe(TINY_BPE); C = corpora()
B = load_json("quant_gpt2")["B"]; base = B["32"]
def bpc(W=None):
    tot, ch = 0.0, 0
    with torch.no_grad():
        for a in range(0, len(g) - 1, 1024):
            w = g[a:a + 1024]
            if len(w) < 2: break
            lp = torch.log_softmax(Q.forward_q(gm, w, W).double(), -1)
            tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); ch += len(tk.decode(w[1:]))
    return tot / math.log(2) / ch
pool_ids = tk.encode(tb.decode([int(i) for i in C["pool"][100000:112000]]))
print(f"  pool tokens available from offset 100000: {len(pool_ids)}")
n = min(64, len(pool_ids) // 256)
cal = [pool_ids[i * 256:(i + 1) * 256] for i in range(n)]
XtX, amax = Q.collect_inputs(gm, cal)
K = {"windows": n}
for bits in (4, 3):
    t0 = time.perf_counter()
    Wg = {f"h.{l}.{nm}.weight": Q.gptq(gm.W[f"h.{l}.{nm}.weight"], XtX[f"h.{l}.{nm}"], bits, 128) for l in range(12) for nm in Q.LINEARS}
    tq = time.perf_counter() - t0; v = bpc(Wg); K[str(bits)] = v
    print(f"  {bits}-bit per 128 inputs, {n} windows: GPTQ {v:.4f} ({v - base:+.4f}); round-to-nearest {B[f'{bits} 128']:.4f} ({B[f'{bits} 128'] - base:+.4f}); took {tq:.0f} s", flush=True)
save_json("extra25", K)
