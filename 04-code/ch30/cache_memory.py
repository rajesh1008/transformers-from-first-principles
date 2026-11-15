"""Chapter 30, section 30.3: the cache's size by hand for four models, checked against a measured cache where the model is
on the build machine (GPT-2 small and SmolLM2-135M through tiny/load.py).  Hand formula: 2 x layers x key-value heads x
numbers per head x bytes per number, per token."""
from _common import *
import torch
torch.set_num_threads(1)
from tiny.load import GPT2, Llama
from tiny import cache as C
from transformers import AutoTokenizer
rows = [("Tiny (Chapter 24)", 12, 4, 32, 4, 128, "32-bit"), ("GPT-2 small", 12, 12, 64, 4, 1024, "32-bit"),
        ("SmolLM2-135M", 30, 3, 64, 2, 8192, "BF16"), ("Qwen3-0.6B (config only)", 28, 8, 128, 2, 40960, "BF16")]
print("model                     layers kv-heads x per-head bytes  per token     at full context        (context)")
for name, L, h, hd, b, ctx, fmt in rows:
    pt = C.kv_bytes_per_token(L, h, hd, b)
    print(f"  {name:<24} {L:4d}   {h:3d} x {hd:3d}   {fmt:<6} {pt:9,} B   {pt * ctx / 2**20:10.1f} MiB  ({ctx:,} tokens)")
print("\nmeasured (the running sentence, then per token):")
tk = AutoTokenizer.from_pretrained(CKPT("smollm2")); sm = Llama.from_dir(CKPT("smollm2"))
ids = tk.encode(SENTENCE); cache = C.new_cache(sm)
with torch.no_grad():
    S = sm.forward(ids, cache, 0); full = sm.forward(ids)
print(f"  SmolLM2-135M: per layer K {list(cache[0][0].shape)}; {C.cache_numbers(cache):,} numbers; {C.cache_bytes(cache) // len(ids):,} bytes per token "
      f"in 32-bit = by hand {C.kv_bytes_per_token(30, 3, 64, 4):,}; (9 query heads would need {C.kv_bytes_per_token(30, 9, 64, 4):,})")
print(f"  cached prefill equals the plain pass: max |score difference| {float((S - full).abs().max()):.2e}")
import time
pr = ids
def best(f, k=3):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)
a, _ = C.greedy(sm, pr, 32, True); b, _ = C.greedy(sm, pr, 32, False)
print(f"  SmolLM2 greedy 32 tokens cached == uncached: {a == b}; cached {best(lambda: C.greedy(sm, pr, 32, True)):.2f} s, uncached {best(lambda: C.greedy(sm, pr, 32, False)):.2f} s")
