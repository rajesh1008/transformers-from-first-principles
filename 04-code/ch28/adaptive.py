"""Chapter 28, section 28.4: entropy-adaptive cuts on Tiny.  eta-sampling (keep tokens with share above
min(eps, sqrt(eps) * exp(-entropy))) and Mirostat version 2 (keep tokens with surprise below mu; after each draw
mu -= lr * (surprise - target)).  3 prompts x 10 continuations x 40 tokens, seed 1865, 32-bit, one thread."""
from _common import *
import torch, re
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G
from _pool import corpora

tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok(); c = corpora()
pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
PROMPTS = ["the cat sat on the", "Alice was beginning to", '"I don\'t know," said the']

def probs(s):
    z = np.exp(s - s.max()); return z / z.sum()

def run(rule, **kw):
    rng = streams(SEED)["sample"]
    texts, seqs, kept, surp, chosen_late = [], [], [], [], []
    for pr in PROMPTS:
        p0 = t.encode(pr)
        for _ in range(10):
            ids = list(p0); mu = 2 * kw.get("target", 0)
            for step in range(40):
                p = probs(G.next_scores(tt, ids))
                if rule == "eta":
                    H = -(p * np.log(np.clip(p, 1e-300, None))).sum()
                    eta = min(kw["eps"], np.sqrt(kw["eps"]) * np.exp(-H))
                    keep = p > eta
                else:
                    keep = -np.log(np.clip(p, 1e-300, None)) < mu
                    if not keep.any():
                        keep = p == p.max()
                q = np.where(keep, p, 0); q /= q.sum()
                nxt = int(rng.choice(len(q), p=q)); s_ = -np.log(p[nxt])
                if rule == "mirostat":
                    mu -= kw.get("lr", 0.1) * (s_ - kw["target"])
                kept.append(int(keep.sum())); surp.append(s_); ids.append(nxt)
                if step >= 20: chosen_late.append(s_)
            seqs.append(ids[len(p0):]); texts.append(t.decode(ids[len(p0):]))
    ws = [w.lower() for x in texts for w in re.findall(r"[A-Za-z]+", x)]
    return dict(kept=float(np.mean(kept)), novel=sum(w not in pool_words for w in ws) / len(ws), surprise=float(np.mean(surp)),
                late=float(np.mean(chosen_late)), d2=G.distinct_n(seqs, 2), first=texts[0])

out = {}
for label, rule, kw in [("eta 0.0003", "eta", dict(eps=3e-4)), ("eta 0.0009", "eta", dict(eps=9e-4)), ("eta 0.004", "eta", dict(eps=4e-3)),
                        ("mirostat 2.0", "mirostat", dict(target=2.0)), ("mirostat 2.5", "mirostat", dict(target=2.5)),
                        ("mirostat 3.0", "mirostat", dict(target=3.0)), ("mirostat 3.5", "mirostat", dict(target=3.5))]:
    r = run(rule, **kw); out[label] = r
    print(f"{label:<13} kept {r['kept']:7.1f}  not-in-pool {r['novel']:.3f}  surprise {r['surprise']:.3f} (steps 21-40: {r['late']:.3f})  "
          f"distinct-2 {r['d2']:.3f}  first {r['first'][:60]!r}")
save_json("adaptive", out)
