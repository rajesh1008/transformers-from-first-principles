"""Lab 46: one small check per misconception, on Tiny (Chapter 24's checkpoint wsd_1e-3_2000_b2000.pt).
CPU, one thread, 32-bit, seed 1865.  Run from 04-code/ch46:  python3 -u lab46.py > outputs/lab46.txt
Checks
 A  "it sees words": how Tiny's tokenizer cuts the running sentence.
 B  "the same prompt gives the same answer": greedy twice; sampling at T=1 under two seeds.
 C  "it learns while you chat": SHA-256 of every learned number before and after generating 64 tokens.
 D  "temperature is randomness": top token after "the cat sat on the" at T = 0.25, 0.5, 1, 2.
 E  "quantisation is free": every 2-D table rounded to 8 bits (symmetric, per output column) and back; loss on Through the Looking-Glass.
 F  "a longer prompt is always better": loss on the same 64 Looking-Glass tokens read with 64, 128, 256 and 512 tokens before them.
 G  breakage, "it looks things up": closed-book and open-book capital questions, greedy 6 tokens.
"""
from _common import *
import torch, numpy as np, hashlib
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); T = tok(); c = corpora()
held = [int(i) for i in c["glass"]]   # Through the Looking-Glass, held out from Tiny's training pool
R = {}; passed = 0; total = 0
def check(name, ok):
    global passed, total
    total += 1; passed += bool(ok); print(f"  check {name}: {'pass' if ok else 'FAIL'}", flush=True)
def probs(ids, temp=1.0):
    with torch.no_grad(): return torch.softmax(tt.scores(ids[-1024:])[-1].double() / temp, -1)
def gen(ids, n, temp=None, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None; ids = list(ids)
    for _ in range(n):
        p = probs(ids[-128:]) if temp is None else probs(ids[-128:], temp)
        t = int(p.argmax()) if temp is None else int(torch.multinomial(p, 1, generator=g))
        ids.append(t)
    return ids
def digest():
    h = hashlib.sha256()
    for k in sorted(tt.t): h.update(tt.t[k].detach().numpy().tobytes())
    return h.hexdigest()

print("A  tokens of the running sentence")
ids = T.encode(SENTENCE); pieces = [T.decode([i]) for i in ids]
print(f"  {len(SENTENCE.split())} words -> {len(ids)} tokens: {pieces}"); R["A"] = pieces
check("A, ' mat' is not one token", " mat" not in pieces)

print("B  same prompt, same answer?")
p0 = T.encode("the cat sat on the")
g1, g2 = gen(p0, 20), gen(p0, 20)
s1, s2 = gen(p0, 20, 1.0, 1), gen(p0, 20, 1.0, 2)
div = next((i for i, (a, b) in enumerate(zip(s1[len(p0):], s2[len(p0):])) if a != b), None)
print(f"  greedy run 1: {T.decode(g1[len(p0):])!r}\n  greedy run 2 identical: {g1 == g2}")
print(f"  sampled, seed 1: {T.decode(s1[len(p0):])!r}\n  sampled, seed 2: {T.decode(s2[len(p0):])!r}\n  first differing generated token: {div}")
R["B"] = dict(greedy=T.decode(g1[len(p0):]), same=g1 == g2, s1=T.decode(s1[len(p0):]), s2=T.decode(s2[len(p0):]), first_diff=div)
check("B, greedy repeats and samples differ", g1 == g2 and div is not None)

print("C  does chatting change the weights?")
d0 = digest(); _ = gen(T.encode("Alice said to the Hatter"), 64, 1.0, 3); d1 = digest()
print(f"  before {d0[:16]}...  after 64 generated tokens {d1[:16]}...  equal: {d0 == d1}"); R["C"] = dict(before=d0, after=d1)
check("C, weights unchanged", d0 == d1)

print("D  temperature and the top token")
R["D"] = {}
for temp in (0.25, 0.5, 1.0, 2.0):
    p = probs(p0, temp); top = int(p.argmax()); srt = torch.sort(p, descending=True).values
    n90 = int((torch.cumsum(srt, 0) < 0.9).sum()) + 1
    R["D"][temp] = dict(top=T.decode([top]), share=float(p[top]), n90=n90)
    print(f"  T={temp:<4}: top {T.decode([top])!r} share {float(p[top]):.4f}; tokens to cover 0.9: {n90}")
check("D, same top token at every temperature", len({v["top"] for v in R["D"].values()}) == 1)

print("E  8-bit round trip of every 2-D table")
W = [held[i * 128: i * 128 + 129] for i in range(64)]
def held_loss():
    with torch.no_grad(): return float(np.mean([float(tt.loss(w)) for w in W]))
base = held_loss(); keep = {k: v.clone() for k, v in tt.t.items()}; maxerr = 0.0
for k, v in tt.t.items():
    if v.dim() == 2:
        s = v.abs().amax(0, keepdim=True) / 127.0; s[s == 0] = 1.0
        q = torch.clamp(torch.round(v / s), -127, 127) * s; maxerr = max(maxerr, float((q - v).abs().max())); tt.t[k] = q
q8 = held_loss(); tt.t.update(keep)
assert held_loss() == base
print(f"  held-out loss on 64 windows of 129 tokens: 32-bit {base:.4f}, 8-bit {q8:.4f}, difference {q8 - base:+.4f}; largest weight error {maxerr:.2e}")
R["E"] = dict(fp32=base, int8=q8, maxerr=maxerr)
check("E, 8-bit changes the loss (not free, though small)", q8 != base)

print("F  the same 64 tokens read with longer and longer contexts")
R["F"] = {}
for L in (64, 128, 256, 512):
    tot = []
    for j in range(8):
        end = 600 + j * 700; ctx = held[end - 64 - L: end]
        with torch.no_grad():
            lp = torch.log_softmax(tt.scores(ctx[:-1]).double(), -1)
        tgt = torch.tensor(ctx[1:]); per = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tot.append(float(per[-64:].mean()))
    R["F"][L] = float(np.mean(tot)); print(f"  context {L:3d} tokens before the scored 64: loss {R['F'][L]:.4f} nats per token (8 places)")
check("F, 512 tokens of context is not better than 128", R["F"][512] >= R["F"][128])

print("G  breakage: try to make Tiny look something up")
R["G"] = {}
for q in ("The capital of France is", "Paris is the capital of France. The capital of France is",
          "The Hatter lives in", "The Hatter lives in Cheshire. The Hatter lives in"):
    ids = T.encode(q); out = gen(ids, 6); p = probs(ids)
    want = " Paris" if "France" in q else " Cheshire"; w = T.encode(want)[0]
    rank = int((p > p[w]).sum()) + 1
    R["G"][q] = dict(out=T.decode(out[len(ids):]), first_piece=T.decode([w]), share=float(p[w]), rank=rank)
    print(f"  {q!r} -> {T.decode(out[len(ids):])!r}; first piece of{want!r} = {T.decode([w])!r}, share {float(p[w]):.4f}, rank {rank}")
closed = R["G"]["The capital of France is"]["out"]
check("G, closed book does not answer Paris", "Paris" not in closed)
save_json("lab46", R)
print(f"{passed} of {total} checks pass")
