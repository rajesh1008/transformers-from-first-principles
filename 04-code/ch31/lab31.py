"""Lab 31: predict, break, check.  Run from 04-code/ch31 after step_cost.py, serve_real.py and experiments.py.
Part 1: predict the pages a request needs.  Part 2: two breakages (a static batch that returns together; admission
without reservation).  Part 3: checkpoints.  No API call; the key is read from .env only to check its absence."""
from _common import *
import torch, glob, math
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny import serve as S
from transformers import AutoTokenizer
ok = []
def check(cond, text):
    ok.append(bool(cond)); print(("[pass] " if cond else "[FAIL] ") + text)
print("Part 1. predict, then read")
kv = S.PagedKV(100, 16); kv.alloc("s", 6)
print(f"  the running sentence, 6 tokens: {len(kv.tables['s'])} page of 16 slots, {kv.reserved_slots() - kv.used_slots()} slots empty")
for _ in range(16): kv.alloc("s", 1)
print(f"  after 16 new tokens (22 in all): {len(kv.tables['s'])} pages, block table {kv.tables['s']}")
kv.free("s"); print(f"  after free: {len(kv.free_pages)} of 100 pages free")
print("\nPart 2. breakages (from outputs/serve_real.json)")
R = load_json("serve_real")
print(f"  static batch returning together: mean reply {R['static']['back_mean']:.2f} s against continuous {R['continuous']['back_mean']:.2f} s")
print(f"  admission without reservation, 384 slots: finished {R['break']['done']} of 20; deadlock {R['break']['deadlock']}")
print("\nPart 3. checkpoints")
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2"))
ps = [tk.encode(SENTENCE), tk.encode("The rabbit looked at the clock and")]
caches = []; last = []
for p in ps:
    s, c = S.prefill(gm, p); caches.append(c); last.append(s)
toks = [int(s.argmax()) for s in last]
sb = S.decode_step(gm, caches, toks, [len(p) for p in ps])
solo = []
for p, t in zip(ps, toks):
    with torch.no_grad(): solo.append(gm.forward(p + [t])[-1])
d = max(float((torch.log_softmax(a, -1) - torch.log_softmax(b, -1)).abs().max()) for a, b in zip(sb, solo))
check(d < 1e-3 and all(int(a.argmax()) == int(b.argmax()) for a, b in zip(sb, solo)),
      f"a batched decode step matches full passes for two conversations of different lengths (max diff {d:.2e})")
kv = S.PagedKV(4, 16); check(kv.alloc("a", 40) and not kv.alloc("b", 30) and (kv.free("a") or kv.alloc("b", 30)),
                              "the page allocator refuses when out of pages and reuses freed pages")
check(R["continuous"]["ttft_mean"] < R["static"]["ttft_mean"] and R["continuous"]["tok_s"] > R["static"]["tok_s"],
      "continuous batching beat static on first-token time and throughput for the twenty clients")
check(R["break"]["deadlock"], "admission that reserves nothing deadlocks once the cache is full")
key = [l.split("=", 1)[1].strip().strip('"') for l in open(os.path.join(HERE, "..", "..", ".env")) if l.startswith("ANTHROPIC_API_KEY")][0]
clean = all(key not in open(f, errors="ignore").read() for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f))
check(clean, "the API key was read from .env and its string appears in no file under 04-code/ch31")
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
assert all(ok)
