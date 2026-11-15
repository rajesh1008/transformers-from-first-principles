"""Chapter 22, sections 22.1 to 22.4: what Tiny's training data looked like, and what a pipeline changes.
Part 1 cuts the novel into documents two ways (its 12 chapters plus front matter; its paragraphs).
Part 2 counts the seats wasted when each document gets its own windows (padding) against packing documents
end to end into windows of 129 tokens (128 predictions).  Part 3 counts, for packed windows, how many predictions
can see an earlier document.  Part 4 counts repeated word 13-grams: between the two raw Gutenberg files, between
the novel and the held-out book once the licence text is removed, and inside the novel.  Part 5 replays the window
draw of Chapters 19 to 21 (seed 1865, 400 steps x 8 windows) and counts how often each token was a target."""
import _common
import re, collections, math
import numpy as np
from tiny.corpus import load_text, load_heldout, RAW, RAW_HELDOUT
from _common import tok, SEED, save_json
T = tok()
W = 129
novel = load_text()
out = {}

print("Part 1. the novel as documents")
parts = re.split(r"\n(?=CHAPTER [IVXL]+\.\s*\n)", novel)
chapters = [p for p in parts if p.strip()]
paras = [p for p in re.split(r"\n\s*\n", novel) if p.strip()]
docs = {"chapters": chapters, "paragraphs": paras}
lens = {}
for k, d in docs.items():
    L = np.array([len(T.encode(x)) for x in d])
    lens[k] = L
    print(f"  {k:10s} {len(d):5d} documents; tokens per document min {L.min()}, median {int(np.median(L))}, max {L.max()}; total {L.sum():,}")
whole = len(T.encode(novel))
print(f"  the novel encoded in one piece: {whole:,} tokens (documents encoded apart differ by {lens['paragraphs'].sum() - whole:+,})")
out["docs"] = {k: dict(n=int(len(v)), total=int(v.sum()), median=float(np.median(v))) for k, v in lens.items()}

print("\nPart 2. seats in windows of 129 tokens (128 predictions each)")
for k, L in lens.items():
    pad_w = int(sum(math.ceil(max(l - 1, 1) / 128) for l in L))          # each document's predictions in its own windows
    preds = int(sum(max(l - 1, 0) for l in L))
    pack_w = math.ceil((L.sum() - 1) / 128)
    pk = int(L.sum() - 1)
    print(f"  {k:10s} padded: {pad_w:5,} windows, {preds:,} predictions of {pad_w * 128:,} seats, {1 - preds / (pad_w * 128):6.1%} wasted"
          f" | packed: {pack_w:4,} windows, {pk:,} predictions, {1 - pk / (pack_w * 128):5.2%} wasted")
    out[f"seats_{k}"] = dict(padded_windows=pad_w, padded_waste=1 - preds / (pad_w * 128), packed_windows=pack_w)

print("\nPart 3. packed windows: predictions whose window also holds an earlier document")
for k, L in lens.items():
    doc_of = np.repeat(np.arange(len(L)), L)
    n = len(doc_of)
    cross = 0; tot = 0
    for s in range(0, n - 1, 128):
        w = doc_of[s:s + W]
        for j in range(1, len(w)):                    # target w[j], context w[:j]
            tot += 1; cross += int(w[0] != w[j])
    print(f"  {k:10s} {cross:,} of {tot:,} predictions ({cross / tot:.1%}) have tokens of another document in their context")
    out[f"cross_{k}"] = cross / tot

print("\nPart 4. repeated runs of 13 words")
def grams(text, n=13):
    w = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("\u2019", "'"))
    return collections.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)), len(w)
ra, na = grams(open(RAW, encoding="utf-8").read()); rb, nb = grams(open(RAW_HELDOUT, encoding="utf-8").read())
shared_raw = set(ra) & set(rb)
print(f"  raw Gutenberg files: {len(shared_raw):,} distinct 13-grams appear in both (of {len(ra):,} and {len(rb):,})")
ga, _ = grams(novel); gb, _ = grams(load_heldout())
shared = set(ga) & set(gb)
print(f"  licence text removed: {len(shared):,} shared 13-grams")
for gtuple in sorted(shared)[:3]:
    print("    e.g.", " ".join(gtuple))
rep = {g: c for g, c in ga.items() if c > 1}
cov = sum(c for c in rep.values())
print(f"  inside the novel: {len(rep):,} distinct 13-grams occur more than once ({cov:,} of {sum(ga.values()):,} positions, {cov / sum(ga.values()):.2%})")
for gtuple, c in sorted(rep.items(), key=lambda x: -x[1])[:3]:
    print(f"    {c} times: {' '.join(gtuple)}")
out["grams"] = dict(shared_raw=len(shared_raw), shared_clean=len(shared), repeated_inside=len(rep), repeated_positions=cov)

print("\nPart 5. the window draw of Chapters 19 to 21 (seed 1865, 400 steps x 8 windows of 129)")
ids = np.array(T.encode(novel)); n = len(ids)
rng = np.random.default_rng(SEED)
starts = rng.integers(0, n - W, size=(400, 8))
seen = np.zeros(n, dtype=int)
for s in starts.ravel():
    seen[s + 1:s + W] += 1
tgt = seen[1:]
print(f"  {n:,} tokens; {400 * 8 * 128:,} predictions = {400 * 8 * 128 / (n - 1):.2f} passes")
print(f"  times each token was a target: min {tgt.min()}, median {int(np.median(tgt))}, max {tgt.max()}; never {int((tgt == 0).sum())}"
      f" ({(tgt == 0).mean():.2%}); 15 or more times {int((tgt >= 15).sum()):,} ({(tgt >= 15).mean():.1%})")
first = np.full(n, -1)
for step in range(400):
    for s in starts[step]:
        blk = first[s + 1:s + W]; blk[blk < 0] = step
fs = first[1:]
for q in (0.5, 0.9, 0.99):
    k = int(np.ceil(q * (n - 1)))
    print(f"  {q:.0%} of tokens had been a target at least once by step {int(np.sort(np.where(fs < 0, 10**6, fs))[k - 1])}")
out["draw"] = dict(passes=400 * 8 * 128 / (n - 1), never=int((tgt == 0).sum()), median=float(np.median(tgt)), max=int(tgt.max()))
save_json("pipeline_basics", out)
