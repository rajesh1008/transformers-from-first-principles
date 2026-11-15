"""Chapter 28, section 28.8: sampling many and choosing.  (A) best-of-n chosen by the model's own score: 16 continuations
of 40 tokens at T 1.0 per prompt, the one with the highest total log-probability kept, n = 1, 4, 16.  (B) self-consistency
in miniature: predict the next whole word of held-out Carroll text (150 positions, 64 tokens of context), by greedy
decoding, by one sample at T 1.0, and by majority vote over n samples; also the share of positions where any of n samples
is right (pass@n).  32-bit, one thread, seed 1865."""
from _common import *
import torch, re, collections
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G
from _pool import corpora

tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok(); c = corpora()
pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
PROMPTS = ["the cat sat on the", "Alice was beginning to", '"I don\'t know," said the']
out = {}
print("Part A: best-of-n by the model's own score (16 x 40 tokens at T 1.0 per prompt, 6 prompts' worth: 3 prompts x 2 rounds)")
rng = streams(SEED)["sample"]
pools = []
for pr in PROMPTS:
    for _ in range(2):
        cands = []
        for _ in range(16):
            ids, info = G.generate(tt, t.encode(pr), 40, rng, T=1.0)
            s = ids[len(t.encode(pr)):]
            cands.append((-float(np.sum(info["surprise"])), s))
        pools.append(cands)
def novel(seqs):
    ws = [w.lower() for s in seqs for w in re.findall(r"[A-Za-z]+", t.decode(s))]
    return sum(w not in pool_words for w in ws) / max(1, len(ws))
for n in (1, 4, 16):
    chosen = [max(p[:n], key=lambda x: x[0]) for p in pools]
    su = np.mean([-lp / 40 for lp, _ in chosen]); seqs = [s for _, s in chosen]
    print(f"  n {n:<3} mean surprise {su:.3f}  rep4 {np.mean([G.repeated_runs(s) for s in seqs]):.3f}  not-in-pool {novel(seqs):.3f}  "
          f"cost {n * 40} tokens per output  first {t.decode(seqs[0])[:60]!r}")
    out[f"A n{n}"] = dict(surprise=float(su), novel=novel(seqs))

print("\nPart B: next whole word of held-out Carroll (150 positions, context 64 tokens)")
vocab = [t.decode([i]) for i in range(2048)]
cont = np.array([bool(re.match(r"[A-Za-z’']", s)) for s in vocab])            # a token that carries on the current word
H = [int(x) for x in c["held_b"]]
pos = [i for i in range(64, len(H) - 8) if re.fullmatch(r" [A-Za-z]+", vocab[H[i]])]
pick = np.random.default_rng(SEED).choice(len(pos), size=150, replace=False)
pos = [pos[i] for i in sorted(pick)]
def word_of(seq):
    w = vocab[seq[0]]
    for j in seq[1:]:
        if not cont[j]: break
        w += vocab[j]
    return w.strip().lower()
def draw_word(ctx, rng):
    ids = list(ctx)
    for k in range(6):
        p = G.probs(G.next_scores(tt, ids))
        nxt = int(np.argmax(p)) if rng is None else int(rng.choice(len(p), p=p))
        if k > 0 and not cont[nxt]: break
        ids.append(nxt)
        if k == 0 and not re.fullmatch(r" [A-Za-z’']+", vocab[nxt]): break
    return word_of(ids[len(ctx):])
rng = streams(SEED)["sample"]
hits = collections.Counter(); N = 16
for i in pos:
    ctx = H[i - 64:i]; target = word_of(H[i:i + 8])
    g = draw_word(ctx, None)
    ws = [draw_word(ctx, rng) for _ in range(N)]
    hits["greedy"] += g == target; hits["one sample"] += ws[0] == target
    for n in (4, 16):
        vote = collections.Counter(ws[:n]).most_common(1)[0][0]
        hits[f"vote of {n}"] += vote == target; hits[f"any of {n}"] += target in ws[:n]
for k in ["greedy", "one sample", "vote of 4", "vote of 16", "any of 4", "any of 16"]:
    print(f"  {k:<11} {hits[k] / len(pos):.3f}  ({hits[k]} of {len(pos)})")
    out["B " + k] = hits[k] / len(pos)
save_json("best_of_n", out)
