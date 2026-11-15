"""Chapter 37, 37.3: ask Tiny about its corpus and about the world.
A  Seen often against seen once.  On Tiny's own training pool (Chapter 23's Carroll pool), every 5-token context whose
   next token was the same at every occurrence is put in a bucket by how many times that context occurs (1, 2-4, 5-19,
   20 or more).  Up to 300 positions per bucket (seed 1865) are scored with the 127 tokens before them: is Tiny's top
   choice the token that followed, its share on that token, and its top-1 confidence.  Then only positions whose next
   token begins a word (decodes with a leading space and a letter), the closest thing a pool has to a "fact".
B  Corpus questions and world questions: eight prompts from the pool's books and eight about the world, greedy
   continuation of 8 tokens, the mean share of the tokens written.
CPU, one thread, 32-bit, seed 1865.  Tiny from ../ch24/outputs/ckpt/wsd_1e-3_2000_b2000.pt."""
from _common import *
import torch, numpy as np, collections
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); c = corpora(); T = tok()
pool = [int(i) for i in c["pool"]]; N = len(pool)
ctx = collections.defaultdict(list)
for i in range(127, N):
    ctx[tuple(pool[i - 5:i])].append(i)
B = {"1": [], "2-4": [], "5-19": [], "20+": []}
for k, pos in ctx.items():
    nxt = {pool[p] for p in pos}
    if len(nxt) != 1: continue
    n = len(pos); b = "1" if n == 1 else "2-4" if n <= 4 else "5-19" if n <= 19 else "20+"
    B[b].append(pos[0])
rng = np.random.default_rng(SEED); R = {"A": {}, "B": {}}
def word_start(t):
    s = T.decode([t]); return len(s) > 1 and s[0] == " " and s[1].isalpha()
print(f"pool {N:,} tokens; deterministic 5-token contexts by count: " + ", ".join(f"{b}: {len(v):,}" for b, v in B.items()), flush=True)
for b, pos in B.items():
    pick = rng.choice(pos, size=min(300, len(pos)), replace=False)
    rows = []
    for p in pick:
        with torch.no_grad(): pr = torch.softmax(tt.scores(pool[p - 127:p])[-1].double(), -1)
        t = pool[p]; rows.append((int(p), float(pr.argmax() == t), float(pr[t]), float(pr.max()), word_start(t)))
    a = np.array([r[1:4] for r in rows]); w = np.array([r[4] for r in rows])
    R["A"][b] = dict(n=len(rows), acc=a[:, 0].mean(), share=a[:, 1].mean(), conf=a[:, 2].mean(),
                    n_word=int(w.sum()), acc_word=a[w, 0].mean(), share_word=a[w, 1].mean(), conf_word=a[w, 2].mean())
    r = R["A"][b]
    print(f"seen {b:>4s} times: n {r['n']:3d}  top-1 right {r['acc']:.3f}  share on the true token {r['share']:.3f}  top-1 confidence {r['conf']:.3f}"
          f"   | word starts only: n {r['n_word']:3d}  right {r['acc_word']:.3f}  share {r['share_word']:.3f}  confidence {r['conf_word']:.3f}", flush=True)
CORPUS = [("The Bellman", "corpus"), ("For the Snark was a", "corpus"), ("Sylvie and", "corpus"), ("'Just the place for a", "corpus"),
          ("They sought it with thimbles, they sought it with", "corpus"), ("said the Professor", "corpus"), ("the Other Professor", "corpus"), ("Bruno said", "corpus")]
WORLD = [("The capital of France is", "world"), ("The largest planet is", "world"), ("Two and two make", "world"), ("The sun rises in the", "world"),
         ("Water is made of hydrogen and", "world"), ("The Queen of England lives in", "world"), ("The first man on the moon was", "world"), ("The author of Hamlet is", "world")]
for p, kind in CORPUS + WORLD:
    ids = T.encode(p); sh = []
    for _ in range(8):
        with torch.no_grad(): pr = torch.softmax(tt.scores(ids[-128:])[-1].double(), -1)
        t = int(pr.argmax()); sh.append(float(pr[t])); ids = ids + [t]
    out = T.decode(ids[len(T.encode(p)):]).replace("\n", " / ")
    R["B"][p] = dict(kind=kind, text=out, mean_share=float(np.mean(sh)))
    print(f"{kind:6s} {p!r} -> {out!r} ({np.mean(sh):.3f})", flush=True)
ms = {k: float(np.mean([v["mean_share"] for v in R["B"].values() if v["kind"] == k])) for k in ("corpus", "world")}
print(f"mean share of written tokens: corpus prompts {ms['corpus']:.3f}, world prompts {ms['world']:.3f}", flush=True)
R["B_mean"] = ms
save_json("tiny_world", R)
