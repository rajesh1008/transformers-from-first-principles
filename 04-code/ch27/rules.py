"""Chapter 27, section 27.4: seven decoding rules on Chapter 24's 2,500-step Tiny, 20 continuations of 40 tokens each after
"the cat sat on the", every draw from tiny/seed.py's 'sample' stream (master seed 1865), 32-bit, one thread.
For each rule: share of generated words (runs of letters) not found anywhere in the training pool (Chapter 22's five books), share of
4-token runs that repeat an earlier run in the same continuation, distinct continuations, mean surprise (nats) of the
chosen tokens under the untempered model, and the average number of tokens the rule kept before drawing."""
from _common import *
import torch, re
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from _pool import corpora

tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok()
c = corpora()
pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
print(f"pool: {len(c['pool']):,} tokens; word types (letters only, lower case): {len(pool_words):,}")

def raw_shares(ids):
    with torch.no_grad():
        return torch.softmax(tt.scores(ids[-1024:])[-1].double(), -1).numpy()

def rule_shares(p, T=1.0, k=None, top_p=None, min_p=None):
    q = np.exp(np.log(p) / T); q /= q.sum()
    keep = np.ones_like(q, bool)
    if k is not None:
        keep &= q >= np.sort(q)[-k]
    if top_p is not None:
        o = np.argsort(-q); c = np.cumsum(q[o])
        n = int(np.searchsorted(c, top_p) + 1)
        m = np.zeros_like(keep); m[o[:n]] = True; keep &= m
    if min_p is not None:
        keep &= q >= min_p * q.max()
    q = np.where(keep, q, 0.0)
    return q / q.sum(), int(keep.sum())

RULES = [("greedy", None), ("T 0.7", dict(T=0.7)), ("T 1.0", dict(T=1.0)), ("top-k 40", dict(k=40)),
         ("top-p 0.9", dict(top_p=0.9)), ("min-p 0.1", dict(min_p=0.1)), ("T 1.3 + min-p 0.1", dict(T=1.3, min_p=0.1))]
PROMPT = t.encode("the cat sat on the")
res = {}
for name, kw in RULES:
    rng = streams(SEED)["sample"]
    texts, rep, surp, kept, novel, words = [], [], [], [], 0, 0
    for _ in range(20):
        ids = list(PROMPT)
        for _ in range(40):
            p = raw_shares(ids)
            if kw is None:
                nxt = int(np.argmax(p)); kept.append(1)
            else:
                q, n = rule_shares(p, **kw); kept.append(n)
                nxt = int(rng.choice(len(q), p=q))
            surp.append(-np.log(p[nxt])); ids.append(nxt)
        new = ids[len(PROMPT):]
        grams = [tuple(new[i:i + 4]) for i in range(len(new) - 3)]
        seen, r = set(), 0
        for g4 in grams:
            r += g4 in seen; seen.add(g4)
        rep.append(r / len(grams))
        txt = t.decode(new); texts.append(txt)
        ws = [w.lower() for w in re.findall(r"[A-Za-z]+", txt)]
        words += len(ws); novel += sum(w not in pool_words for w in ws)
    res[name] = dict(novel=novel / words, rep=float(np.mean(rep)), distinct=len(set(texts)), surprise=float(np.mean(surp)),
                     kept=float(np.mean(kept)), first=texts[0])
    print(f"{name:<18} not-in-pool words {novel / words:.3f}  repeated 4-runs {np.mean(rep):.3f}  distinct {len(set(texts)):>2}/20  "
          f"mean surprise {np.mean(surp):.3f}  tokens kept {np.mean(kept):7.1f}")
print("\nFirst continuation under each rule:")
for name, _ in RULES:
    print(f"  {name:<18} {res[name]['first']!r}")
save_json("rules", res)
