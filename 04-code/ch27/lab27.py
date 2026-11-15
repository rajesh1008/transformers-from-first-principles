"""Lab 27: predict, break decoding five ways, and run the checkpoints.  Chapter 24's 2,500-step Tiny, 32-bit, one thread.
Stops with an error if any checkpoint fails.  No hosted-model call."""
from _common import *
import torch, re, math
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G

tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok()
P = t.encode("the cat sat on the")

print("Part 1. predict, then read")
s = G.next_scores(tt, P)
for T in (1.0, 0.5):
    q, _ = G.shares(s, T)
    e = float(-(q[q > 0] * np.log(q[q > 0])).sum())
    print(f"  T {T}: entropy {e:.3f} nats = {math.exp(e):.0f} equally likely choices" if False else
          f"  T {T}: entropy {e:.3f} nats, about {np.exp(e):.0f} equally likely choices")
q, k = G.shares(s, min_p=0.1)
print(f"  min-p 0.1 keeps {k} tokens after the prompt (largest share {np.exp(s - s.max()).max() / np.exp(s - s.max()).sum():.4f})")

print("\nPart 2. five breakages")
# 1 temperature applied after the cut instead of before: min-p threshold moves
q_before, k1 = G.shares(s, T=1.5, min_p=0.1)
z = np.exp(s - s.max()); z /= z.sum(); keep = z >= 0.1 * z.max()
q_after = np.where(keep, np.exp((s - s.max()) / 1.5), 0); q_after /= q_after.sum()
print(f"  1 temperature after the min-p cut: keeps {int(keep.sum())} tokens instead of {k1}")
# 2 a shared generator: one extra draw elsewhere moves every later sample
rng = streams(SEED)["sample"]; a = G.generate(tt, P, 10, rng)[0]
rng = streams(SEED)["sample"]; rng.random(); b = G.generate(tt, P, 10, rng)[0]
print(f"  2 one extra draw from the sampling generator: same 10 tokens? {a == b}")
# 3 greedy with no stop and no penalty
ids, info = G.generate(tt, P, 120, None)
print(f"  3 greedy 120 tokens: repeated 4-runs {G.repeated_runs(ids[len(P):]):.3f}")
# 4 top-p computed on unsorted shares
z = np.exp(s - s.max()); z /= z.sum()
wrong = int(np.searchsorted(np.cumsum(z), 0.9) + 1)
_, right = G.shares(s, top_p=0.9)
print(f"  4 top-p 0.9 on unsorted shares keeps the first {wrong} token ids; sorted keeps {right}")
# 5 option scores by mean per token instead of total
names = [" Alice", " Bruno", " Sylvie", " Balbus"]
opts = [t.encode(n) for n in names]
lp = G.option_logprobs(tt, t.encode('"Come here, my dear," said'), opts)
per = lp / np.array([len(o) for o in opts])
print(f"  5 choosing a name by total log-prob picks {names[int(np.argmax(lp))].strip()}; by mean per token {names[int(np.argmax(per))].strip()}")

print("\nPart 3. checkpoints")
checks = []
q0, _ = G.shares(s, T=0.01)
checks.append(("temperature 0.01 puts the largest share on the greedy token", int(np.argmax(q0)) == int(np.argmax(s))))
rng = streams(SEED)["sample"]
checks.append(("top-k 1 sampling equals greedy for 20 tokens", G.generate(tt, P, 20, rng, top_k=1)[0] == G.generate(tt, P, 20, None)[0]))
r1 = G.generate(tt, P, 20, streams(SEED)["sample"], min_p=0.1)[0]; r2 = G.generate(tt, P, 20, streams(SEED)["sample"], min_p=0.1)[0]
checks.append(("a fresh 'sample' stream from seed 1865 repeats a min-p continuation", r1 == r2))
checks.append(("beam width 1 equals greedy for 20 tokens", G.beam_search(tt, P, 20, 1)[0][1] == G.generate(tt, P, 20, None)[0]))
lm = G.local_mask_probs(tt, t.encode('"Come here, my dear," said'), opts)
checks.append(("token-by-token masking chances over four names sum to 1", abs(lm.sum() - 1) < 1e-9))
nr = G.generate(tt, P, 120, None, no_repeat=4)[0]
checks.append(("no-repeat 4 leaves no repeated 4-token run in 120 greedy tokens", G.repeated_runs(nr[len(P):]) == 0))
key = None
envp = os.path.join(HERE, "..", "..", ".env")
if os.path.exists(envp):
    for line in open(envp):
        if line.startswith("ANTHROPIC_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"')
bad = False
if key:
    for root, _, files in os.walk(HERE):
        for f in files:
            try:
                bad |= key in open(os.path.join(root, f), errors="ignore").read()
            except IsADirectoryError:
                pass
checks.append(("the API key was read from .env and its string appears in no file under 04-code/ch27", bool(key) and not bad))
for name, ok in checks:
    print(f"[{'pass' if ok else 'FAIL'}] {name}")
n = sum(ok for _, ok in checks)
print(f"\n{n} of {len(checks)} checkpoints pass")
if n != len(checks):
    sys.exit(1)
