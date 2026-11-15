"""Lab 30: add a cache, time it, break it by invalidating it wrongly, and run the checkpoints.  GPT-2 small through
tiny/load.py and tiny/cache.py; CPU, one thread, 32-bit, greedy.  Reads the API key from ../../.env into this process only to
confirm that its string appears in no file under 04-code/ch30 (no hosted-model call is made)."""
from _common import *
import torch, time, glob
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny import cache as C
from transformers import AutoTokenizer
from _pool import corpora
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2"))
g = tk.encode(tok().decode([int(i) for i in corpora()["held_b"]]))
ok = []
def check(label, cond):
    ok.append(bool(cond)); print(f"[{'pass' if cond else 'FAIL'}] {label}")

print("Part 1. predict, then read")
ids = tk.encode(SENTENCE); cache = C.new_cache(gm)
with torch.no_grad(): gm.forward(ids, cache, 0)
print(f"  cache after the running sentence: {C.cache_bytes(cache):,} bytes; by hand 2 x 12 x 12 x 64 x 4 x 6 = {2 * 12 * 12 * 64 * 4 * 6:,}")
print(f"  a 4,096-token conversation would hold {C.kv_bytes_per_token(12, 12, 64, 4) * 4096 / 2**20:.0f} MiB (beyond GPT-2's 1,024 positions; arithmetic only)")

class PrefixStore:
    """Keeps one cached prefix.  key='tokens' (correct) matches on the exact token tuple; key='length' (the breakage)
    matches on the prefix's length only."""
    def __init__(self, key): self.key, self.k, self.cache = key, None, None
    def _id(self, toks): return tuple(toks) if self.key == "tokens" else len(toks)
    def get(self, toks):
        return [(a.clone(), b.clone()) for a, b in self.cache] if self.k == self._id(toks) else None
    def put(self, toks, cache): self.k, self.cache = self._id(toks), cache

print("\nPart 2. two breakages")
pre = g[:200]; q = g[400:416]
edited = list(pre); edited[10] = tk.encode(" dog")[0]
with torch.no_grad():
    fresh = torch.log_softmax(gm.forward(edited + q)[200:].double(), -1)
    for key in ("tokens", "length"):
        st = PrefixStore(key); c0 = C.new_cache(gm); gm.forward(pre, c0, 0); st.put(pre, c0)
        hit = st.get(edited)
        if hit is None:
            hit = C.new_cache(gm); gm.forward(edited, hit, 0); used = "miss, prefix recomputed"
        else:
            used = "hit, old prefix reused"
        out = torch.log_softmax(gm.forward(q, hit, 200).double(), -1)
        print(f"  Breakage 1, prefix store keyed on {key:<6}: {used:<24} max |log-prob diff| against a fresh pass {float((out - fresh).abs().max()):.3e}")
    a, _ = C.greedy(gm, tk.encode("The rabbit looked at"), 16)
    cache = C.new_cache(gm); ids2 = tk.encode("The rabbit looked at"); s = gm.forward(ids2, cache, 0)[-1]
    for _ in range(16):
        nx = int(s.argmax()); ids2.append(nx); s = gm.forward([nx], cache, 0)[-1]     # position never advanced
print(f"  Breakage 2, position never advanced: {tk.decode(ids2[len(tk.encode('The rabbit looked at')):])!r}")
print(f"     correct:                         {tk.decode(a[len(tk.encode('The rabbit looked at')):])!r}")
print("\nPart 3. checkpoints")
p = g[:256]
a, _ = C.greedy(gm, p, 16, True); b, _ = C.greedy(gm, p, 16, False)
check("cached and uncached greedy decoding write the same 16 tokens after a 256-token prompt", a == b)
t0 = time.perf_counter(); C.greedy(gm, p, 16, True); tc = time.perf_counter() - t0
t0 = time.perf_counter(); C.greedy(gm, p, 16, False); tu = time.perf_counter() - t0
print(f"       tokens per second: cached {16 / tc:.1f}, uncached {16 / tu:.1f}")
check("the cache is faster than recomputation after a 256-token prompt", tc < tu)
check("the cache's bytes equal the hand formula for the running sentence", C.cache_bytes(cache if False else (lambda c: (gm.forward(tk.encode(SENTENCE), c, 0), c)[1])(C.new_cache(gm))) == 2 * 12 * 12 * 64 * 4 * 6)
st = PrefixStore("tokens"); c0 = C.new_cache(gm); gm.forward(pre, c0, 0); st.put(pre, c0)
check("a prefix store keyed on tokens misses after a one-token edit and hits on the same tokens", st.get(edited) is None and st.get(list(pre)) is not None)
key = None
for line in open(os.path.join(HERE, "..", "..", ".env")):
    if "=" in line and "ANTHROPIC" in line.split("=")[0]:
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
hit = [f for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f) and key and key in open(f, "rb").read().decode("latin-1")]
check("the API key was read from .env and its string appears in no file under 04-code/ch30", key and not hit)
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
if not all(ok): raise SystemExit(1)
