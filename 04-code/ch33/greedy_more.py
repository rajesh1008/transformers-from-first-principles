"""Chapter 33 (session 25, Exercise 12's idea at ten prompts): greedy text under rounding on ten held-out prompts of 64 tokens,
64 new tokens each, agreement with 32-bit.  Four prompts are Part G's (offsets 800, 1600, 2400, 3000); six are new (200, 500, 1000,
1300, 2000, 2700).  Full recomputation each step (no cache), as Part G.  python3 greedy_more.py CONFIG   CONFIG = 8ch | 6g32 | 4g32
CPU, one thread, seed 1865, 11 Nov 2026."""
from _common import *
import torch, sys, json
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny import quant as Q
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
cfg = sys.argv[1]
bits, gr = {"8ch": (8, None), "6g32": (6, 32), "4g32": (4, 32)}[cfg]
names = [f"h.{l}.{n}.weight" for l in range(12) for n in Q.LINEARS]
W = {n: Q.quantize(gm.W[n], bits, gr) for n in names}
offs = (200, 500, 800, 1000, 1300, 1600, 2000, 2400, 2700, 3000)
prompts = [g[i:i + 64] for i in offs]
def greedy_q(p, W):
    ids = list(p)
    with torch.no_grad():
        for _ in range(64): ids.append(int(Q.forward_q(gm, ids, W)[-1].argmax()))
    return ids[64:]
ref = [greedy_q(p, None) for p in prompts]; o = [greedy_q(p, W) for p in prompts]
first = [next((i for i in range(64) if a[i] != b[i]), 64) for a, b in zip(o, ref)]
same = sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(o, ref))
print(f"{cfg}: bits {bits}, group {gr}: tokens before first difference {first}; identical for all 64: {sum(f == 64 for f in first)} of 10; positions equal {same}/640")
json.dump(dict(cfg=cfg, first=first, same=same), open(os.path.join(HERE, "outputs", f"greedy_more_{cfg}.json"), "w"))
