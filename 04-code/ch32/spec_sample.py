"""Chapter 32, 32.5 to 32.6: speculative sampling on GPT-2 small, and its speed under sampling.
Part C: one position.  Target p = GPT-2 at T 1, draft q = GPT-2's first 6 layers + ln_f + tied output (an untrained
  early exit), after a held-out prompt.  The rule's output distribution computed exactly, then 100,000 draws of the rule
  against 100,000 draws of plain sampling; two broken rules (resample from p on rejection; accept only q's top token).
Part D: two speculated tokens end to end (k = 2, the 6-layer draft sampled at T 1) after the running sentence:
  3,000 speculative runs against 3,000 + 3,000 plain-sampling runs; total-variation distance of each token's shares.
Part F: acceptance by temperature for early exits at 2 / 4 / 6 / 8 / 10 layers, from one pass over 512 held-out tokens:
  greedy agreement and the expected acceptance sum(min(p, q)) at T 0.7 and 1.0.
Part E: prompt lookup under sampling (T 0.7, 1.0) and greedy, per prompt, k = 4, 64 new tokens, timed against plain.
CPU, one thread, 32-bit, seed 1865; timings best of 3, 11 Nov 2026."""
from _common import *
import torch, time, numpy as np
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny.cache import new_cache
from tiny import speculate as SP
from spec_greedy_lib import fwd_layers
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
THE = tk.encode(" the")[0]; V = 50257
out = {}
def tv(a, b): return 0.5 * float(np.abs(a - b).sum())
def shares(xs): c = np.bincount(xs, minlength=V).astype(float); return c / c.sum()
def top(d, n=5): i = np.argsort(-d)[:n]; return ", ".join(f"{tk.decode([int(j)])!r} {d[j]:.3f}" for j in i)

print("Part C: one position after a held-out prompt, target GPT-2 (12 layers), draft its first 6 layers, T 1")
pr = g[800:864]
with torch.no_grad():
    p = SP.probs(gm.forward(pr)[-1], 1.0); q = SP.probs(fwd_layers(gm, pr, [None] * 6, 0, 6)[-1], 1.0)
a = float(np.minimum(p, q).sum())
res = np.maximum(p - q, 0); res /= res.sum()
exact = np.minimum(p, q) + (1 - a) * res
broken1 = q * np.minimum(1, p / np.maximum(q, 1e-300)) + (1 - a) * p
qa = int(q.argmax()); broken2 = np.zeros(V); broken2[qa] = p[qa]; broken2 += (1 - p[qa]) * np.where(np.arange(V) == qa, 0, p) / (1 - p[qa])
print(f"  acceptance sum(min(p, q)) = {a:.4f}; TV(p, q) = {tv(p, q):.4f}")
print(f"  target p top 5: {top(p)}")
print(f"  draft  q top 5: {top(q)}")
print(f"  rule's output, computed exactly: max |difference from p| {np.abs(exact - p).max():.2e}; TV {tv(exact, p):.2e}")
print(f"  broken rule 1 (on rejection, draw from p): TV from p {tv(broken1, p):.4f} (exact)")
print(f"  broken rule 2 (draft guesses its top token; accept with p(g), else draw from p without g): TV {tv(broken2, p):.2e} (exact)")
rng = np.random.default_rng(SEED); N = 100_000
plain = rng.choice(V, size=N, p=p)
gs = rng.choice(V, size=N, p=q); u = rng.random(N); acc = u < np.minimum(1, p[gs] / q[gs])
fix = rng.choice(V, size=N, p=res); spec = np.where(acc, gs, fix)
fix1 = rng.choice(V, size=N, p=p); b1 = np.where(acc, gs, fix1)
print(f"  {N:,} draws each (seed {SEED}): accepted {acc.mean():.4f} of draft guesses")
print(f"    plain sampling    TV from p {tv(shares(plain), p):.4f}   (sampling noise alone)")
print(f"    speculative rule  TV from p {tv(shares(spec), p):.4f};  TV from plain draws {tv(shares(spec), shares(plain)):.4f}")
print(f"    broken rule 1     TV from p {tv(shares(b1), p):.4f}")
for name, d in (("p", p), ("speculative draws", shares(spec)), ("broken 1 draws", shares(b1))):
    print(f"    {name:18s} top 5: {top(d)}")
out["C"] = dict(acc=a, tv_pq=tv(p, q), exact_maxdiff=float(np.abs(exact - p).max()), broken1=tv(broken1, p), broken2=tv(broken2, p),
                emp_acc=float(acc.mean()), tv_plain=tv(shares(plain), p), tv_spec=tv(shares(spec), p), tv_b1=tv(shares(b1), p))
save_json("spec_sample", out)

print("\nPart F: acceptance by temperature for untrained early exits, 512 held-out positions")
w = g[1000:1512]
with torch.no_grad():
    P = gm.forward(w)
    F = {}
    for L in (2, 4, 6, 8, 10):
        Q = fwd_layers(gm, w, [None] * L, 0, L)
        row = dict(greedy=float((P.argmax(-1) == Q.argmax(-1)).float().mean()))
        for T in (0.7, 1.0):
            pp = torch.softmax(P.double() / T, -1); qq = torch.softmax(Q.double() / T, -1)
            row[str(T)] = float(torch.minimum(pp, qq).sum(-1).mean())
        F[L] = row
        print(f"  first {L:2d} layers: greedy agreement {row['greedy']:.3f}; expected acceptance T 0.7 {row['0.7']:.3f}, T 1.0 {row['1.0']:.3f}", flush=True)
out["F"] = F; save_json("spec_sample", out)

print("\nPart D: two speculated tokens after the running sentence (k = 2, 6-layer draft sampled at T 1)")
sp = tk.encode(SENTENCE); base = new_cache(gm)
with torch.no_grad(): s0 = gm.forward(sp, base, 0)[-1]
t1 = int(s0.argmax())                                  # fix the first token (the target's greedy choice) so runs share a prefill
pre = sp + [t1]
def clone(c): return [(a.clone(), b.clone()) for a, b in c]
def plain_run(rng):
    c = clone(base); ids = list(pre); outt = []
    with torch.no_grad():
        for _ in range(2):
            s = gm.forward([ids[-1]], c, len(ids) - 1)[-1]; t = int(rng.choice(V, p=SP.probs(s, 1.0))); ids.append(t); outt.append(t)
    return outt
dbase = [None] * 6
with torch.no_grad(): fwd_layers(gm, sp, dbase, 0, 6)
def spec_run(rng):
    c = clone(base); dc = clone(dbase); ids = list(pre); guess = []; qs = []
    with torch.no_grad():
        cur = [ids[-1]]; n = len(ids) - 1
        for _ in range(2):
            qq = SP.probs(fwd_layers(gm, cur, dc, n, 6)[-1], 1.0); n += 1
            t = int(rng.choice(V, p=qq)); guess.append(t); qs.append(qq); cur = [t]
        rows = SP.verify(gm, c, ids[-1], guess, len(ids) - 1)
    new = SP.accept_sample(guess, qs, [SP.probs(r, 1.0) for r in rows], rng)
    return new[:2] if len(new) >= 2 else None, len(new) - 1, new
M = 3000; rng = np.random.default_rng(SEED)
t0 = time.perf_counter(); A1 = [plain_run(rng) for _ in range(M)]; A2 = [plain_run(rng) for _ in range(M)]; tp = time.perf_counter() - t0
S = []; accs = []; t0 = time.perf_counter()
for _ in range(M):
    two, m, new = spec_run(rng); accs.append(m)
    if two is None:                                   # first guess rejected: the pass gave one token; take the second from the target
        c = clone(base); ids = pre + new
        with torch.no_grad():
            gm.forward([pre[-1]], c, len(pre) - 1)
            s = gm.forward([ids[-1]], c, len(ids) - 1)[-1]
        two = new + [int(rng.choice(V, p=SP.probs(s, 1.0)))]
    S.append(two)
ts = time.perf_counter() - t0
D = {}
for j in range(2):
    a1 = shares([x[j] for x in A1]); a2 = shares([x[j] for x in A2]); ss = shares([x[j] for x in S])
    D[j] = dict(tv_plain_plain=tv(a1, a2), tv_spec_plain=tv(ss, a1), tv_spec_plain2=tv(ss, a2))
    print(f"  token {j + 2} after the prompt: TV plain vs plain {tv(a1, a2):.4f}; speculative vs plain {tv(ss, a1):.4f} / {tv(ss, a2):.4f}")
    print(f"    plain top 5: {top(a1)}\n    spec  top 5: {top(ss)}")
print(f"  first token fixed to greedy {tk.decode([t1])!r}; guesses accepted per pass: mean {np.mean(accs):.3f} of 2; "
      f"distribution {np.bincount(accs, minlength=3).tolist()}; plain {tp:.0f} s for 2 x {M}, speculative {ts:.0f} s")
out["D"] = dict(D=D, acc_mean=float(np.mean(accs)), acc_hist=np.bincount(accs, minlength=3).tolist()); save_json("spec_sample", out)

print("\nPart E: prompt lookup, k = 4, 64 new tokens per prompt, against plain cached decoding with the same settings")
prompts = {"separator (token 0)": g[0:64], "held-out 800": g[800:864], "held-out 1600": g[1600:1664], "held-out 2400": g[2400:2464]}
def plain_dec(p, n, T, seed):
    rng = np.random.default_rng(seed); ids = list(p); c = new_cache(gm)
    with torch.no_grad():
        s = gm.forward(ids, c, 0)[-1]
        for _ in range(n):
            t = int(s.argmax()) if T is None else int(rng.choice(V, p=SP.probs(s, T))); ids.append(t)
            s = gm.forward([t], c, len(ids) - 1)[-1]
    return ids
def best(f, r=3):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); o = f(); ts.append(time.perf_counter() - t0)
    return min(ts), o
E = {}
for T in (None, 0.7, 1.0):
    for name, p in prompts.items():
        tpl, _ = best(lambda: plain_dec(p, 64, T, SEED))
        tsp, (ids, st) = best(lambda: SP.speculate(gm, p, 64, lambda ids, k: (SP.lookup_draft(ids, k, THE), None), 4,
                                                  None if T is None else (T, np.random.default_rng(SEED))))
        same = ids == plain_dec(p, 64, None, SEED) if T is None else None
        tag = "greedy" if T is None else f"T {T}"
        E[f"{tag} | {name}"] = dict(acc=st["accepted"] / st["proposed"], tpp=64 / (st["passes"] - 1), plain=tpl, spec=tsp, speedup=tpl / tsp, same=same)
        print(f"  {tag:7s} {name:20s}: accepted {st['accepted']:3d}/{st['proposed']:3d} ({st['accepted'] / st['proposed']:.3f}); "
              f"{64 / (st['passes'] - 1):4.2f} tokens per verify pass; plain {tpl:5.2f} s, lookup {tsp:5.2f} s ({tpl / tsp:4.2f} x)"
              + (f"; same as greedy {same}" if same is not None else ""), flush=True)
out["E"] = E; save_json("spec_sample", out)
