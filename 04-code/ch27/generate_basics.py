"""Chapter 27, sections 27.1 to 27.3: Tiny writes.  Chapter 24's 2,500-step model (WSD 1e-3, order 1865), 32-bit, one thread.
Part A: the shares after "the cat sat on the"; Part B: greedy continuation; Part C: temperature and the shares;
Part D: ten sampled continuations at temperature 1 from tiny/seed.py's 'sample' stream; Part E: repeat check."""
from _common import *
import torch, math
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams

CK = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
tt = load(CK)
t = tok()
CTX = tt.t["P"].shape[0]
PROMPT = "the cat sat on the"

def shares(ids, T=1.0):
    with torch.no_grad():
        s = tt.scores(ids[-CTX:])[-1].double()
    return torch.softmax(s / T, -1).numpy()

def show(i):
    return repr(t.decode([int(i)]))

def generate(ids, n, rule, rng=None, T=1.0):
    ids = list(ids)
    for _ in range(n):
        p = shares(ids, T)
        if rule == "greedy":
            nxt = int(np.argmax(p))
        else:
            nxt = int(rng.choice(len(p), p=p / p.sum()))
        ids.append(nxt)
    return ids

out = {}
ids = t.encode(PROMPT)
print(f"model {os.path.basename(CK)}; context {CTX} tokens; vocabulary {tt.t['E'].shape[0]}")
print(f"prompt {PROMPT!r} -> {len(ids)} tokens {ids}")

print("\nPart A: shares for the next token")
p = shares(ids)
order = np.argsort(-p)
ent = float(-(p * np.log(p)).sum())
for r in order[:10]:
    print(f"  {show(r):>12}  {p[r]:.4f}")
print(f"  top 10 hold {p[order[:10]].sum():.4f}; entropy {ent:.3f} nats (uniform {math.log(len(p)):.3f}); "
      f"tokens needed for 0.9 of the share {int(np.searchsorted(np.cumsum(p[order]), 0.9) + 1)}")
out["A"] = {"top": [[t.decode([int(r)]), float(p[r])] for r in order[:10]], "entropy": ent}

print("\nPart B: greedy, 40 tokens")
g_ids = generate(ids, 40, "greedy")
txt = t.decode(g_ids)
print("  " + repr(txt))
new = g_ids[len(ids):]
# first repeat of a 4-token run
rep = None
for i in range(len(new) - 4):
    for j in range(i + 1, len(new) - 3):
        if new[i:i + 4] == new[j:j + 4]:
            rep = (i, j); break
    if rep: break
print(f"  first repeated 4-token run: {rep} (positions in the generated part)")
g2 = generate(ids, 40, "greedy")
print(f"  second greedy run identical: {g2 == g_ids}")
out["B"] = {"text": txt, "repeat": rep}

print("\nPart C: temperature")
for T in (0.25, 0.5, 0.7, 1.0, 1.5, 2.0):
    q = shares(ids, T); o = np.argsort(-q)
    e = float(-(q * np.log(q)).sum())
    print(f"  T {T:<4}  top share {q[o[0]]:.4f} ({show(o[0])})  top-5 hold {q[o[:5]].sum():.4f}  entropy {e:.3f}  "
          f"tokens for 0.9: {int(np.searchsorted(np.cumsum(q[o]), 0.9) + 1)}")
    out.setdefault("C", []).append([T, float(q[o[0]]), e])

print("\nPart D: ten samples at T 1.0, 20 tokens each, seed 1865 ('sample' stream)")
rng = streams(SEED)["sample"]
samples = []
for k in range(10):
    s_ids = generate(ids, 20, "sample", rng)
    samples.append(t.decode(s_ids[len(ids):]))
    print(f"  {k + 1:>2} {PROMPT}|{samples[-1]!r}")
first = [s for s in samples]
print(f"  distinct continuations {len(set(samples))} of 10")
out["D"] = samples

print("\nPart E: repeat with a fresh 'sample' stream from the same seed")
rng = streams(SEED)["sample"]
again = [t.decode(generate(ids, 20, "sample", rng)[len(ids):]) for _ in range(10)]
print(f"  same ten continuations: {again == samples}")
rng = streams(7)["sample"]
other = [t.decode(generate(ids, 20, "sample", rng)[len(ids):]) for _ in range(10)]
print(f"  master seed 7: same as 1865 in {sum(a == b for a, b in zip(other, samples))} of 10")
out["E"] = {"repeat": again == samples}
save_json("generate_basics", out)
