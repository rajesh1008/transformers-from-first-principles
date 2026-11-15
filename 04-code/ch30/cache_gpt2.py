"""Chapter 30, sections 30.1 to 30.4: the key-value cache on GPT-2 small (tiny/load.py's forward pass with tiny/cache.py).
Part A: what the cache holds after the running sentence.  Part B: cached and uncached greedy decoding agree.
Part C: time per new token with and without the cache, at two prompt lengths.  Part D: prefill against decode.
Part E: three ways to break a cache.  CPU, one thread, 32-bit; greedy (no draws); timings 11 Nov 2026, best of 3."""
from _common import *
import torch, time, math
torch.set_num_threads(1)
from tiny.load import GPT2, compare
from tiny import cache as C
from transformers import AutoTokenizer
from _pool import corpora
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2"))
out = {}

print("Part A: the cache after the running sentence")
ids = tk.encode(SENTENCE); cache = C.new_cache(gm)
with torch.no_grad():
    S = gm.forward(ids, cache, 0)
print(f"  {len(ids)} tokens; per layer K {list(cache[0][0].shape)} and V {list(cache[0][1].shape)} (heads, positions, numbers per head)")
print(f"  {len(cache)} layers; numbers held {C.cache_numbers(cache):,}; bytes {C.cache_bytes(cache):,} (32-bit); "
      f"per token {C.cache_bytes(cache) // len(ids):,} = by hand {C.kv_bytes_per_token(12, 12, 64, 4):,}")
out["A"] = dict(numbers=C.cache_numbers(cache), bytes=C.cache_bytes(cache))

print("\nPart B: cached against uncached greedy decoding, 64 tokens after 'The rabbit looked at'")
pr = tk.encode("The rabbit looked at")
a, ra = C.greedy(gm, pr, 64, cached=True); b, rb = C.greedy(gm, pr, 64, cached=False)
dd = max(float((torch.log_softmax(x.double(), -1) - torch.log_softmax(y.double(), -1)).abs().max()) for x, y in zip(ra, rb))
print(f"  same 64 tokens: {a == b}; largest |difference| in log-probabilities over the 64 steps {dd:.3e}")
print(f"  text: {tk.decode(a[len(pr):])[:110]!r}")
out["B"] = dict(same=a == b, maxdiff=dd)

def best(f, k=3):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)

c = corpora()
from tiny.tokenizer import load_bpe
tt = load_bpe(os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt"))
g = tk.encode(tt.decode([int(i) for i in c["held_b"]]))
print("\nPart C: time per new token, 32 new tokens, one thread")
res = {}
for P in (16, 256, 768):
    p = g[:P]
    tc = best(lambda: C.greedy(gm, p, 32, True)); tu = best(lambda: C.greedy(gm, p, 32, False), 1 if P > 256 else 3)
    res[P] = dict(cached=tc, uncached=tu)
    print(f"  prompt {P:4d}: cached {tc:7.2f} s ({32 / tc:6.1f} tokens/s, includes the prefill); uncached {tu:7.2f} s ({32 / tu:5.1f} tokens/s); ratio {tu / tc:5.1f}", flush=True)
out["C"] = res

print("\nPart D: prefill against decode")
rows = []
for P in (16, 64, 256, 768):
    p = g[:P]
    with torch.no_grad():
        tp = best(lambda: gm.forward(p, C.new_cache(gm), 0))
        cache = C.new_cache(gm); gm.forward(p, cache, 0)
        def one():
            cc = [(k.clone(), v.clone()) for k, v in cache]
            gm.forward([g[P]], cc, P)
        td = best(one, 5)
    rows.append((P, tp, td))
    print(f"  context {P:4d}: prefill {tp * 1000:8.1f} ms ({tp / P * 1000:6.2f} ms per prompt token); one decode step {td * 1000:6.1f} ms", flush=True)
out["D"] = rows

print("\nPart E: three ways to break a cache (largest |difference| in log-probabilities against the full pass, 24 tokens of held-out text)")
seq = g[:24]
with torch.no_grad():
    ref = torch.log_softmax(gm.forward(seq).double(), -1)
    def run(mode):
        cache = C.new_cache(gm); rows = []
        for i, t in enumerate(seq):
            start = 0 if mode == "pos" else i
            if mode == "overwrite" and i > 0:
                cache = C.new_cache(gm)
            rows.append(gm.forward([t], cache, start)[-1])
        L = torch.log_softmax(torch.stack(rows).double(), -1)
        return float((L - ref).abs().max()), bool((L.argmax(-1) == ref.argmax(-1)).all())
    for lab, m in [("correct cache", "ok"), ("every new token at position 0", "pos"), ("cache replaced, not appended", "overwrite")]:
        d_, s_ = run(m); print(f"  {lab:<36} {d_:.3e}  same top token everywhere: {s_}")
    # stale cache: the prompt edited after its keys and values were cached
    edited = list(seq); edited[3] = tk.encode(" dog")[0]
    cache = C.new_cache(gm); gm.forward(seq[:20], cache, 0)
    stale = torch.log_softmax(gm.forward(edited[20:], cache, 20).double(), -1)
    fresh = torch.log_softmax(gm.forward(edited).double(), -1)[20:]
    print(f"  {'prompt edited at token 4, cache kept':<36} {float((stale - fresh).abs().max()):.3e}  same top token everywhere: {bool((stale.argmax(-1) == fresh.argmax(-1)).all())}")
save_json("cache_gpt2", out)
