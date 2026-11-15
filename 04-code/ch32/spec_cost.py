"""Chapter 32, 32.7: when speculation stops paying.
Part G: k sweep for the prompt-lookup draft, greedy, 64 new tokens, on a repeating prompt (the separator) and a
  non-repeating one (held-out 1600): tokens per verify pass and time against plain greedy.
Part H: what verifying five tokens costs inside a batch.  tiny/serve.py's decode_step at context 128 for B
  conversations with one row each, against B x 5 rows (each conversation's five rows run as five single-token rows over
  copies of its cache: a cost stand-in for a batched verify pass, since the linear maps see B x 5 rows either way).
CPU, one thread, 32-bit, greedy; timings best of 3, 11 Nov 2026."""
from _common import *
import torch, time, numpy as np
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny.cache import greedy
from tiny import speculate as SP, serve as SV
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
THE = tk.encode(" the")[0]; out = {}
def best(f, r=3, setup=None):
    ts = []
    for _ in range(r):
        a = setup() if setup else None
        t0 = time.perf_counter(); o = f(a) if setup else f(); ts.append(time.perf_counter() - t0)
    return min(ts), o
print("Part G: prompt lookup, greedy, 64 new tokens; k sweep")
G = {}
for name, p in (("separator (token 0)", g[0:64]), ("held-out 1600", g[1600:1664])):
    tpl, (ref, _) = best(lambda: greedy(gm, p, 64, True))
    print(f"  {name}: plain greedy {tpl:.2f} s")
    for k in (1, 2, 4, 8):
        t, (ids, st) = best(lambda: SP.speculate(gm, p, 64, lambda ids, kk: (SP.lookup_draft(ids, kk, THE), None), k))
        G[f"{name} k{k}"] = dict(tpp=64 / (st["passes"] - 1), acc=st["accepted"] / st["proposed"], time=t, speedup=tpl / t, same=ids == ref)
        print(f"    k {k}: same {ids == ref}; accepted {st['accepted']}/{st['proposed']} ({st['accepted'] / st['proposed']:.3f}); "
              f"{64 / (st['passes'] - 1):4.2f} tokens per pass; {t:5.2f} s ({tpl / t:4.2f} x)", flush=True)
out["G"] = G; save_json("spec_cost", out)
print("\nPart H: decode_step at context 128, B conversations x 1 row against B x 5 rows")
H = {}
ps = [g[i * 130:i * 130 + 128] for i in range(16)]
caches = [SV.prefill(gm, p)[1] for p in ps]
def clone(c): return [(a.clone(), b.clone()) for a, b in c]
for B in (1, 4, 16):
    for rows in (1, 5):
        def setup(): return [clone(caches[i]) for i in range(B) for _ in range(rows)]
        toks = [ps[i][-1] for i in range(B) for _ in range(rows)]; pos = [128] * (B * rows)
        t, _ = best(lambda cs: SV.decode_step(gm, cs, toks, pos), 3, setup)
        H[f"B{B} r{rows}"] = t
    r = H[f"B{B} r5"] / H[f"B{B} r1"]
    print(f"  B {B:2d}: one row each {H[f'B{B} r1'] * 1000:6.1f} ms; five rows each {H[f'B{B} r5'] * 1000:6.1f} ms ({r:4.2f} x); "
          f"break-even {r:4.2f} tokens per verify pass", flush=True)
out["H"] = H; save_json("spec_cost", out)
