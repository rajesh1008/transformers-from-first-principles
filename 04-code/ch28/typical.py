"""Chapter 28, sections 28.1 to 28.3: surprise against entropy, token by token, for held-out Carroll text and for Tiny's own
greedy and T 1.0 output; then locally typical sampling.  Chapter 24's 2,500-step model, 32-bit, one thread, seed 1865."""
from _common import *
import torch, re
torch.set_num_threads(1)
from tiny.eval import load, windows
from tiny.seed import streams
from tiny import generate as G
from _pool import corpora

tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok(); c = corpora()
pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
PROMPTS = ["the cat sat on the", "Alice was beginning to", '"I don\'t know," said the']

def per_token(seq_ids, start):
    """surprise of each actual token and the entropy of the model's table before it, positions start.. end"""
    with torch.no_grad():
        S = tt.scores(list(seq_ids)[:-1]).double()
    lp = torch.log_softmax(S, -1)
    H = -(lp.exp() * lp).sum(-1).numpy()
    tgt = torch.as_tensor(seq_ids[1:])
    s = (-lp.gather(-1, tgt[:, None]).squeeze(-1)).numpy()
    return s[start - 1:], H[start - 1:]

out = {}
print("Part A: surprise of each token against the entropy of the table it was drawn from")
W = windows(np.asarray(c["held_b"]), 129)[:20]
rows = {}
S_, H_ = [], []
for w in W:
    s, h = per_token(list(map(int, w)), 29)          # skip the first 28 seats so every prediction has some context
    S_.append(s); H_.append(h)
rows["held-out Carroll"] = (np.concatenate(S_), np.concatenate(H_))
P = t.encode(PROMPTS[0])
for label, rng_kw in [("greedy", None), ("T 1.0", dict(T=1.0)), ("T 0.7", dict(T=0.7))]:
    rng = streams(SEED)["sample"]
    S_, H_ = [], []
    for _ in range(1 if rng_kw is None else 20):
        ids, _ = G.generate(tt, P, 100, None if rng_kw is None else rng, **(rng_kw or {}))
        s, h = per_token(ids, len(P))
        S_.append(s); H_.append(h)
    rows[label] = (np.concatenate(S_), np.concatenate(H_))
for label, (s, h) in rows.items():
    print(f"  {label:<17} tokens {len(s):5d}  mean surprise {s.mean():.3f}  mean entropy {h.mean():.3f}  "
          f"mean |surprise - entropy| {np.abs(s - h).mean():.3f}  share with surprise below entropy {np.mean(s < h):.3f}")
    out[label] = dict(surprise=float(s.mean()), entropy=float(h.mean()), gap=float(np.abs(s - h).mean()), below=float(np.mean(s < h)))

def typical_shares(s, tau):
    z = s - s.max(); p = np.exp(z); p /= p.sum()
    lp = np.log(np.clip(p, 1e-300, None)); H = -(p * lp).sum()
    o = np.argsort(np.abs(-lp - H))
    n = min(len(p), int(np.searchsorted(np.cumsum(p[o]), tau) + 1))
    q = np.zeros_like(p); q[o[:n]] = p[o[:n]]
    return q / q.sum(), n

print("\nPart B: locally typical sampling (3 prompts x 10 continuations x 40 tokens, seed 1865)")
for tau in (0.2, 0.5, 0.9, 1.0):
    rng = streams(SEED)["sample"]
    texts, seqs, kept, surp = [], [], [], []
    for pr in PROMPTS:
        p0 = t.encode(pr)
        for _ in range(10):
            ids = list(p0)
            for _ in range(40):
                s = G.next_scores(tt, ids)
                q, n = typical_shares(s, tau)
                nxt = int(rng.choice(len(q), p=q))
                z = np.exp(s - s.max()); z /= z.sum()
                surp.append(-np.log(z[nxt])); kept.append(n); ids.append(nxt)
            seqs.append(ids[len(p0):]); texts.append(t.decode(ids[len(p0):]))
    ws = [w.lower() for x in texts for w in re.findall(r"[A-Za-z]+", x)]
    nov = sum(w not in pool_words for w in ws) / len(ws)
    print(f"  tau {tau:<4} kept {np.mean(kept):7.1f}  not-in-pool {nov:.3f}  surprise {np.mean(surp):.3f}  distinct-2 {G.distinct_n(seqs, 2):.3f}"
          f"  first {texts[0][:70]!r}")
    out[f"tau{tau}"] = dict(kept=float(np.mean(kept)), novel=nov, surprise=float(np.mean(surp)), d2=G.distinct_n(seqs, 2), first=texts[0])
save_json("typical", out)
