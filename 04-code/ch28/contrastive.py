"""Chapter 28, section 28.5: contrastive decoding with a younger Tiny.  Expert: Chapter 24's 2,500-step model (b2000);
amateurs: the same WSD run's 500-step and 1,000-step decayed branches (b500 counts 500 trunk steps plus its decay; see
Chapter 24), no new training.  Greedy contrastive decoding over 60 tokens from three prompts, then sampled versions.
32-bit, one thread, seed 1865."""
from _common import *
import torch, re
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G
from _pool import corpora

CK = os.path.join(HERE, "..", "ch24", "outputs", "ckpt")
exp = load(os.path.join(CK, "wsd_1e-3_2000_b2000.pt"))
ama = {"b500": load(os.path.join(CK, "wsd_1e-3_2000_b500.pt")), "b1000": load(os.path.join(CK, "wsd_1e-3_2000_b1000.pt"))}
t = tok(); c = corpora()
pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
PROMPTS = ["the cat sat on the", "Alice was beginning to", '"I don\'t know," said the']

def novel(texts):
    ws = [w.lower() for x in texts for w in re.findall(r"[A-Za-z]+", x)]
    return sum(w not in pool_words for w in ws) / max(1, len(ws))

def decode(prompt, n, amateur=None, alpha=0.1, beta=1.0, rng=None, T=1.0, model=None):
    ids = list(t.encode(prompt)); start = len(ids); surp, kept = [], []
    for _ in range(n):
        se = G.next_scores(model or exp, ids)
        if amateur is None:
            sc, k = se, len(se)
        else:
            sc, k = G.contrastive_scores(se, G.next_scores(ama[amateur], ids), alpha, beta)
        if rng is None:
            nxt = int(np.argmax(sc))
        else:
            q = G.probs(sc / T); nxt = int(rng.choice(len(q), p=q))
        surp.append(-np.log(G.probs(se)[nxt])); kept.append(k); ids.append(nxt)
    return ids[start:], float(np.mean(surp)), float(np.mean(kept))

out = {}
print("Part A: greedy, 60 tokens from each of three prompts (expert surprise of the chosen tokens; repeated 4-runs; words not in pool)")
rows = [("expert greedy", dict()), ("amateur b500 greedy", dict(model=ama["b500"])), ("amateur b1000 greedy", dict(model=ama["b1000"])),
        ("CD b500 a0.1 b1.0", dict(amateur="b500")), ("CD b1000 a0.1 b1.0", dict(amateur="b1000")),
        ("CD b500 a0.1 b0.5", dict(amateur="b500", beta=0.5)), ("CD b500 a0.01 b1.0", dict(amateur="b500", alpha=0.01)),
        ("CD b500 a0.0001 b1.0", dict(amateur="b500", alpha=1e-4)), ("CD b500 a0 (no cut)", dict(amateur="b500", alpha=1e-300))]
for label, kw in rows:
    seqs, texts, su, ke = [], [], [], []
    for pr in PROMPTS:
        s, m, k = decode(pr, 60, **kw); seqs.append(s); texts.append(t.decode(s)); su.append(m); ke.append(k)
    rep = float(np.mean([G.repeated_runs(s) for s in seqs]))
    out["A " + label] = dict(surprise=float(np.mean(su)), rep4=rep, novel=novel(texts), kept=float(np.mean(ke)), texts=texts)
    print(f"  {label:<22} kept {np.mean(ke):7.1f}  surprise {np.mean(su):.3f}  rep4 {rep:.3f}  not-in-pool {novel(texts):.3f}  p1 {texts[0][:64]!r}")

print("\nPart B: sampled, 3 prompts x 10 continuations x 40 tokens, seed 1865")
for label, kw in [("expert T 1.0", dict()), ("expert T 0.7", dict(T=0.7)), ("CD b500 a0.1 T 1.0", dict(amateur="b500")),
                  ("CD b1000 a0.1 T 1.0", dict(amateur="b1000")), ("CD b500 a0.1 b0.5 T 1.0", dict(amateur="b500", beta=0.5))]:
    rng = streams(SEED)["sample"]; seqs, texts, su, ke = [], [], [], []
    for pr in PROMPTS:
        for _ in range(10):
            s, m, k = decode(pr, 40, rng=rng, **kw); seqs.append(s); texts.append(t.decode(s)); su.append(m); ke.append(k)
    out["B " + label] = dict(surprise=float(np.mean(su)), novel=novel(texts), d2=G.distinct_n(seqs, 2), kept=float(np.mean(ke)), first=texts[0])
    print(f"  {label:<24} kept {np.mean(ke):7.1f}  surprise {np.mean(su):.3f}  not-in-pool {novel(texts):.3f}  distinct-2 {G.distinct_n(seqs, 2):.3f}  first {texts[0][:60]!r}")

print("\nPart C: where expert and amateur disagree most, first step after prompt 1")
ids = t.encode(PROMPTS[0]); se, sa = G.next_scores(exp, ids), G.next_scores(ama["b500"], ids)
pe, pa = G.probs(se), G.probs(sa)
print(f"  expert entropy {G.entropy(pe):.3f}  amateur b500 entropy {G.entropy(pa):.3f}")
for j in np.argsort(-pe)[:8]:
    print(f"  {t.decode([int(j)])!r:<12} expert {pe[j]:.4f}  amateur {pa[j]:.4f}  log ratio {np.log(pe[j] / pa[j]):+.3f}")
sc, k = G.contrastive_scores(se, sa, 0.1)
print(f"  plausibility cut at 0.1 keeps {k}; CD picks {t.decode([int(np.argmax(sc))])!r}")
sc0, _ = G.contrastive_scores(se, sa, 1e-300); j = int(np.argmax(sc0))
print(f"  with no cut CD picks {t.decode([j])!r}: expert {pe[j]:.2e}, amateur {pa[j]:.2e}")
save_json("contrastive", out)
