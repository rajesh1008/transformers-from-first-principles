"""Chapter 22, s22-5 (Interlude: hashing and MinHash).  Part 1: exact duplicates by fingerprint.  Part 2: Jaccard
similarity of 5-word shingle sets, and MinHash's estimate of it from signatures of 16, 64 and 256 numbers.
Part 3: the banding rule, the chance two documents of similarity s become a candidate pair.  Part 4: near-duplicate
search over the novel's paragraphs plus 40 altered copies (SYNTHETIC: each copy has a few words changed by this
script, to give the search something to find), with MinHash in 32 bands of 4 against exact Jaccard on every pair."""
import _common
import time, itertools, random
import numpy as np
from tiny import data as D
from _common import SEED, save_json
out = {}
novel = D.load_source("alice")
paras = [p for p in D.paragraphs(novel)]
print("Part 1. exact duplicates among the novel's 817 paragraphs (normalised text, 64-bit fingerprints)")
keys = {}
for i, p in enumerate(paras):
    keys.setdefault(D.exact_key(p), []).append(i)
dups = [v for v in keys.values() if len(v) > 1]
print(f"  {len(keys):,} distinct fingerprints; {len(dups)} groups of identical paragraphs")
for g in dups[:4]:
    print(f"    {len(g)} copies: {paras[g[0]][:60]!r}")
out["exact_groups"] = len(dups)

print("\nPart 2. Jaccard similarity and its MinHash estimate (5-word shingles)")
rng = random.Random(SEED)
long = [p for p in paras if len(p.split()) >= 60]
base = long[3]
w = base.split()
def alter(words, k, r):
    w = list(words)
    for j in r.sample(range(len(w)), k):
        w[j] = "jabberwock"
    return " ".join(w)
for k in (1, 3, 8):
    cp = alter(w, k, rng)
    a, b = D.shingles(base), D.shingles(cp)
    J = D.jaccard(a, b)
    est = {}
    for n in (16, 64, 256):
        sa, sb = D.minhash(a, n), D.minhash(b, n)
        est[n] = float(np.mean(sa == sb))
    print(f"  {len(w)} words, {k} changed: exact Jaccard {J:.3f}; MinHash estimate {est[16]:.3f} (16), {est[64]:.3f} (64), {est[256]:.3f} (256)")
    out[f"jaccard_{k}"] = dict(J=J, est=est)

print("\nPart 3. chance that a pair becomes a candidate, 32 bands of 4 numbers (128 in all)")
for s in (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"  similarity {s:.1f}: {D.p_candidate(s, 32, 4):.3f}")
print(f"  the threshold, where the chance is about one half, is near (1/32)^(1/4) = {(1 / 32) ** 0.25:.3f}")
out["p_cand"] = {s: D.p_candidate(s, 32, 4) for s in (0.2, 0.5, 0.8)}

print("\nPart 4. near-duplicate search: 817 paragraphs + 40 altered copies (synthetic)")
r2 = random.Random(SEED + 1)
src = r2.sample([i for i, p in enumerate(paras) if len(p.split()) >= 40], 40)
docs = list(paras)
truth = []
for j, i in enumerate(src):
    ww = paras[i].split()
    docs.append(alter(ww, max(1, len(ww) // 40), r2))          # about one word in forty changed
    truth.append((i, len(paras) + j))
sh = [D.shingles(d) for d in docs]
t0 = time.perf_counter()
sigs = [D.minhash(s, 128) for s in sh]
t_sig = time.perf_counter() - t0
t0 = time.perf_counter()
cand = D.lsh_pairs(sigs, 32, 4)
t_lsh = time.perf_counter() - t0
t0 = time.perf_counter()
exact = {(i, j) for i, j in itertools.combinations(range(len(docs)), 2) if D.jaccard(sh[i], sh[j]) >= 0.5}
t_all = time.perf_counter() - t0
found = {p for p in cand if D.jaccard(sh[p[0]], sh[p[1]]) >= 0.5}
n_pairs = len(docs) * (len(docs) - 1) // 2
print(f"  {len(docs)} documents, {n_pairs:,} pairs; pairs with Jaccard >= 0.5 (every pair checked): {len(exact)}")
print(f"  MinHash candidates: {len(cand)} pairs ({len(cand) / n_pairs:.3%} of all pairs); of the {len(exact)} true pairs it found {len(found & exact)}")
print(f"  planted copies found: {sum(1 for p in truth if p in found)} of {len(truth)}")
print(f"  time: signatures {t_sig:.1f} s, banding {t_lsh * 1000:.0f} ms, every pair checked exactly {t_all:.1f} s (one thread, {__import__('datetime').date.today()})")
extra = sorted(exact - set(truth))
for i, j in extra[:4]:
    print(f"    a pair not planted: {D.jaccard(sh[i], sh[j]):.2f}  {docs[i][:50]!r} / {docs[j][:50]!r}")
out["search"] = dict(docs=len(docs), pairs=n_pairs, true=len(exact), cand=len(cand), found=len(found & exact),
                     planted=sum(1 for p in truth if p in found))
save_json("interlude_minhash", out)
