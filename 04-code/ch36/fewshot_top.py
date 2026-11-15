"""Chapter 36, 36.4: what GPT-2 small and SmolLM2-135M put first when they miss, at k = 4 (arrow format, the same prompts
as fewshot.py).  Prints the greedy first token for each of the 24 queries and counts how often it copies one of the four
examples' answers, repeats the query word, or is something else.  CPU, one thread, 32-bit, seed 1865."""
from _common import *
import torch, numpy as np
torch.set_num_threads(1)
from fewshot_lib import PAIRS, build, full_text
from transformers import AutoTokenizer, AutoModelForCausalLM
R = {}
for name in ("gpt2", "smollm2"):
    tk = AutoTokenizer.from_pretrained(CKPT(name)); m = AutoModelForCausalLM.from_pretrained(CKPT(name), dtype=torch.float32).eval()
    rng = np.random.default_rng(SEED); cnt = {"right": 0, "copies an example's answer": 0, "repeats the query word": 0, "other": 0}; rows = []
    for qi in range(len(PAIRS)):
        segs = build(qi, 4, rng); text = full_text(segs, "arrow", 4)
        with torch.no_grad(): t = int(m(torch.tensor([tk.encode(text)])).logits[0, -1].argmax())
        w = tk.decode([t]).strip(); outs = [s.strip() for s, k in segs if k == "demo_out"]
        kind = "right" if t == tk.encode(" " + PAIRS[qi][1])[0] else ("copies an example's answer" if w in outs else ("repeats the query word" if w == PAIRS[qi][0] else "other"))
        cnt[kind] += 1; rows.append((PAIRS[qi][0], w, kind))
    print(f"{name}: " + ", ".join(f"{k} {v}" for k, v in cnt.items()), flush=True)
    print("   " + "  ".join(f"{q}->{w!r}" for q, w, _ in rows), flush=True)
    R[name] = dict(counts=cnt, rows=rows)
save_json("fewshot_top", R)
