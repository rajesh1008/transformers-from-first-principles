"""Chapter 42: what the trained routers do.  For one run (NAME), on the 60 held-out Looking-Glass windows:
held-out loss; each layer's share of routing slots per expert and the busiest expert's share; experts with under 2 per cent;
how token-like routing is (for token types seen at least 20 times, the share of their occurrences that go to the type's
commonest expert, against the same measure with expert labels shuffled across occurrences); the commonest tokens of each
expert in the first and last layers; the held-out loss when each expert is removed in turn; a capacity limit applied at
evaluation (capacity factor 1.0, 1.25, 2.0): share of slots dropped and loss; the running sentence's route.
CPU, one thread.   NAME=moe4k1 python3 analyse.py"""
import os, json, sys
HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.environ.get("NAME", "moe4k1"); cfg = json.load(open(os.path.join(HERE, "outputs", NAME + ".json")))
os.environ.update(KIND=cfg["kind"], E=str(cfg["E"]), K=str(cfg["K"]), HID=str(cfg["hid"]), BIAS=str(cfg.get("bias", 0)), SHARED=str(cfg.get("shared", 0)))
import moe
from moe import *
from tiny.tokenizer import load_bpe
torch.manual_seed(SEED)
m = Net(); m.load_state_dict(torch.load(os.path.join(HERE, "outputs", "ckpt", NAME + ".pt"))); m.eval()
c = corpora(); g = np.asarray(c["glass"], dtype=np.int64)
HELD = torch.tensor(np.stack([g[i * 129:(i + 1) * 129] for i in range(60)]))
T = load_bpe(TINY_BPE); E_, K_ = cfg["E"], cfg["K"]
R = dict(name=NAME)
with torch.no_grad():
    base = float(m.loss(HELD)); R["held"] = base; print(f"{NAME}: held-out loss {base:.4f}")
    ids = HELD[:, :-1].reshape(-1).numpy()
    R["layers"] = []
    rng = np.random.default_rng(SEED)
    for l, b in enumerate(m.h):
        idx = b.ff.last["idx"].numpy(); first = idx[:, 0]
        share = np.bincount(idx.reshape(-1), minlength=E_) / idx.size
        # token-likeness: share of a type's occurrences going to its commonest expert (top-1 choice)
        def purity(lab):
            num = den = 0
            for t in np.unique(ids):
                w = ids == t
                if w.sum() >= 20: num += np.bincount(lab[w], minlength=E_).max(); den += w.sum()
            return num / den
        pur = purity(first); shuf = purity(rng.permutation(first))
        ent = float(-(b.ff.last["p"] * torch.log(b.ff.last["p"] + 1e-12)).sum(-1).mean())
        tops = []
        if l in (0, len(m.h) - 1):
            for e in range(E_):
                w = first == e; cnt = np.bincount(ids[w], minlength=2048); o = cnt.argsort()[::-1][:6]
                tops.append([[T.decode([int(i)]), int(cnt[i])] for i in o if cnt[i] > 0])
        R["layers"].append(dict(share=share.tolist(), busiest=float(share.max()), under2=int((share < 0.02).sum()), purity=float(pur), purity_shuffled=float(shuf), entropy=ent, tops=tops))
        print(f"layer {l}: shares {' '.join(f'{s:.3f}' for s in share)}; busiest {share.max():.3f}; experts under 2% {(share < 0.02).sum()}; "
              f"token purity {pur:.3f} (shuffled {shuf:.3f}); router entropy {ent:.3f} nats (uniform {np.log(E_):.3f})")
        for e, tp in enumerate(tops): print(f"   expert {e} commonest tokens: " + ", ".join(f"{s!r} {n}" for s, n in tp))
    # removing one expert at a time
    R["remove"] = []
    for l, b in enumerate(m.h):
        row = []
        for e in range(E_):
            b.ff.drop = e; row.append(float(m.loss(HELD)) - base); b.ff.drop = None
        R["remove"].append(row); print(f"remove one expert, layer {l}: loss rise " + " ".join(f"{x:+.3f}" for x in row))
    # a capacity limit at evaluation
    R["capacity"] = []
    for cf in (1.0, 1.25, 2.0):
        for b in m.h: b.ff.cap = cf
        L = float(m.loss(HELD)); dr = sum(b.ff.last["dropped"] for b in m.h) / sum(b.ff.last["n"] * K_ for b in m.h)
        for b in m.h: b.ff.cap = None
        R["capacity"].append(dict(cf=cf, loss=L, dropped=dr)); print(f"capacity factor {cf}: slots dropped {dr:.3f}, held-out loss {L:.4f} ({L - base:+.4f})")
    # the running sentence
    s = T.encode(SENTENCE); m(torch.tensor([s]))
    route = [[int(b.ff.last["idx"][i, 0]) for b in m.h] for i in range(len(s))]
    R["sentence"] = [[T.decode([t]), r] for t, r in zip(s, route)]
    print("running sentence, top expert by layer: " + "; ".join(f"{T.decode([t])!r} {''.join(map(str, r))}" for t, r in zip(s, route)))
save_json("analyse_" + NAME, R)
