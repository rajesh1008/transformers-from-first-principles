"""Chapter 31, 31.1 to 31.2: what one decode step costs as the batch grows, on GPT-2 small (tiny/load.py, tiny/serve.py).
Part A: batched decode equals one-at-a-time decode.  Part B: decode step time against batch size at two contexts.
Part C: prefill time against prompt length.  Part D: a cost model fitted to B and C (used by the simulations).
CPU, one thread, 32-bit; timings best of 3, 11 Nov 2026."""
from _common import *
import torch, time
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.load import GPT2
from tiny import serve as S
from transformers import AutoTokenizer
import numpy as np
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2"))
g = held_text_ids(tk)
out = {}
def best(f, k=3):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)

print("Part A: four conversations decoded together against one at a time, 8 greedy steps each")
prompts = [tk.encode(SENTENCE), g[0:40], g[100:117], g[300:390]]
solo = []
for p in prompts:
    s, c = S.prefill(gm, p); ids = list(p); rows = []
    for _ in range(8):
        n = int(s.argmax()); ids.append(n); rows.append(s)
        with torch.no_grad():
            s = gm.forward([n], c, len(ids) - 1)[-1]
    solo.append((ids, rows))
caches, last, ids = [], [], [list(p) for p in prompts]
for p in prompts:
    s, c = S.prefill(gm, p); caches.append(c); last.append(s)
rowsb = [[] for _ in prompts]
for step in range(8):
    toks = []
    for i in range(4):
        rowsb[i].append(last[i]); n = int(last[i].argmax()); ids[i].append(n); toks.append(n)
    sc = S.decode_step(gm, caches, toks, [len(x) - 1 for x in ids])
    last = [sc[i] for i in range(4)]
same = all(ids[i] == solo[i][0] for i in range(4))
dd = max(float((torch.log_softmax(a.double(), -1) - torch.log_softmax(b.double(), -1)).abs().max())
         for i in range(4) for a, b in zip(rowsb[i], solo[i][1]))
print(f"  contexts {[len(p) for p in prompts]}; same tokens: {same}; largest |difference| in log-probabilities {dd:.3e}")
print(f"  running sentence continues: {tk.decode(ids[0][len(prompts[0]):])!r}")
out["A"] = dict(same=same, maxdiff=dd)

print("\nPart B: one decode step for a batch of B conversations (time per step and per token)")
resB = []
for ctx in (128, 512):
    base_s, base_c = S.prefill(gm, g[:ctx])
    for B in (1, 2, 4, 8, 16, 32, 64):
        def step():
            cs = [[(k.clone(), v.clone()) for k, v in base_c] for _ in range(B)]
            t0 = time.perf_counter(); S.decode_step(gm, cs, [g[ctx]] * B, [ctx] * B); return time.perf_counter() - t0
        t = min(step() for _ in range(3))
        resB.append(dict(ctx=ctx, B=B, step=t))
        print(f"  context {ctx:4d}  B {B:3d}: step {t * 1000:8.1f} ms; {t * 1000 / B:6.1f} ms per token; {B / t:7.1f} tokens/s", flush=True)
out["B"] = resB

print("\nPart C: prefill (one pass over the prompt, filling the cache)")
resC = []
for P in (16, 64, 128, 256, 512, 768):
    t = best(lambda: S.prefill(gm, g[:P]))
    resC.append(dict(P=P, t=t))
    print(f"  prompt {P:4d}: {t * 1000:8.1f} ms; {t * 1000 / P:5.2f} ms per prompt token", flush=True)
out["C"] = resC

print("\nPart D: fitted cost model")
X = np.array([[1, r["B"], r["B"] * r["ctx"]] for r in resB]); y = np.array([r["step"] for r in resB])
a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
err = np.abs(X @ [a, b, c] - y) / y
print(f"  decode step = {a * 1000:.1f} ms + {b * 1000:.2f} ms x B + {c * 1e6:.3f} us x (sum of contexts); worst relative miss {err.max():.2f}")
Xp = np.array([[1, r["P"], r["P"] ** 2] for r in resC]); yp = np.array([r["t"] for r in resC])
pa, pb, pc = np.linalg.lstsq(Xp, yp, rcond=None)[0]
errp = np.abs(Xp @ [pa, pb, pc] - yp) / yp
print(f"  prefill = {pa * 1000:.1f} ms + {pb * 1000:.3f} ms x P + {pc * 1e6:.4f} us x P^2; worst relative miss {errp.max():.2f}")
out["D"] = dict(dec=[a, b, c], pre=[pa, pb, pc], dec_miss=float(err.max()), pre_miss=float(errp.max()))
save_json("step_cost", out)
