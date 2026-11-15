"""Chapter 25, section 25.9: what a small multiple-choice benchmark adds to a loss.
A CONSTRUCTED cloze set, built here from the two held-out texts (never from the pool): at a lowercase word of four or more
letters that follows a lowercase word or a comma, the text before it (its last 100 tokens) is the question, and the
choices are the true word and three distractors: lowercase words of four or more letters from the pool whose pool count
is nearest the true word's (three drawn from the 20 nearest, seed 1865).  Chance is 0.25.  400 items from Looking-Glass,
150 from the Carroll held-out set.  Items are written to outputs/cloze_items.json.
Scoring a choice: the sum of the log-probabilities of its tokens after the question ("sum"), or their average ("mean").
Models: Chapter 24's 1e-3 WSD branches (625, 1,250, 2,500 steps) and Chapter 25's five new runs (2,500-step branches with
and without AdamW 0.1, reading orders 1865, 7, 11); baselines: unigram and bigram counts from the pool.
32-bit, one thread.  Writes outputs/bench.json."""
import _common
import os, re, math, json, glob
import numpy as np, torch
torch.set_num_threads(1)
from tiny import eval as E
from _pool import corpora, POOL_BOOKS
from _common import HERE, SEED, tok, save_json
T = tok(); c = corpora()
pool = np.concatenate([np.asarray(c["books"][b]) for b in POOL_BOOKS])
V = 2048
WORD = re.compile(r"[a-z]{4,}")
pool_text = T.decode(list(map(int, pool)))
from collections import Counter
wc = Counter(WORD.findall(pool_text))
vocab = sorted(wc)
logc = {w: math.log(wc[w]) for w in vocab}

def items(text, n, rng, tag):
    cands = [m for m in re.finditer(r"(?<=[a-z,] )([a-z]{4,})\b", text) if m.start() > 400]
    pick = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    out = []
    for j in sorted(pick):
        m = cands[j]; w = m.group(1)
        ctx = text[:m.start() - 1]
        lc = logc.get(w, 0.0)
        near = sorted((v for v in vocab if v != w), key=lambda v: abs(logc[v] - lc))[:20]
        ds = list(rng.choice(near, 3, replace=False))
        q = T.encode(ctx)[-100:]
        out.append(dict(src=tag, word=w, distractors=ds, context_tail=ctx[-400:], q=q,
                        choices=[T.encode(" " + x) for x in [w] + ds], pool_count=int(wc.get(w, 0))))
    return out

rng = np.random.default_rng(SEED)
glass_text = T.decode(list(map(int, c["glass"]))); car_text = T.decode(list(map(int, c["held_b"])))
IT = items(glass_text, 400, rng, "glass") + items(car_text, 150, rng, "carroll")
json.dump([{k: v for k, v in it.items() if k not in ("q", "choices")} for it in IT], open(os.path.join(HERE, "outputs", "cloze_items.json"), "w"), indent=0)
print(f"constructed cloze items: {sum(i['src'] == 'glass' for i in IT)} Looking-Glass, {sum(i['src'] == 'carroll' for i in IT)} Carroll held-out; "
      f"true word never in the pool: {sum(i['pool_count'] == 0 for i in IT)}")
for it in IT[:3]:
    print(f"  ...{it['context_tail'][-60:]!r} -> [{it['word']}] / {', '.join(it['distractors'])}")
ntok = np.array([[len(ch) for ch in it["choices"]] for it in IT])
print(f"  tokens per choice: true word {ntok[:, 0].mean():.2f}, distractors {ntok[:, 1:].mean():.2f}")

def score_model(tt):
    S = np.zeros((len(IT), 4)); Mn = np.zeros((len(IT), 4))
    with torch.no_grad():
        for i, it in enumerate(IT):
            seqs = [it["q"] + ch for ch in it["choices"]]
            L = max(map(len, seqs))
            X = torch.tensor([s + [0] * (L - len(s)) for s in seqs])
            lp = torch.log_softmax(tt.scores(X[:, :-1]).double(), -1)
            for k, ch in enumerate(it["choices"]):
                q0 = len(it["q"])
                v = [float(lp[k, q0 - 1 + j, ch[j]]) for j in range(len(ch))]
                S[i, k], Mn[i, k] = sum(v), sum(v) / len(v)
    return S, Mn

cnt = np.bincount(pool, minlength=V); pu = np.log((cnt + 1) / (cnt.sum() + V))
big = np.zeros((V, V)); np.add.at(big, (pool[:-1], pool[1:]), 1); rows = big.sum(1)
def score_counts(kind):
    S = np.zeros((len(IT), 4)); Mn = np.zeros((len(IT), 4))
    for i, it in enumerate(IT):
        for k, ch in enumerate(it["choices"]):
            prev = [it["q"][-1]] + ch[:-1]
            v = [pu[t] if kind == "unigram" else math.log((big[a, t] + 0.01) / (rows[a] + 0.01 * V)) for a, t in zip(prev, ch)]
            S[i, k], Mn[i, k] = sum(v), sum(v) / len(v)
    return S, Mn

CK24 = os.path.join(HERE, "..", "ch24", "outputs", "ckpt"); CK25 = os.path.join(HERE, "outputs", "ckpt")
MODELS = {"625 steps": os.path.join(CK24, "wsd_1e-3_2000_b500.pt"), "1,250 steps": os.path.join(CK24, "wsd_1e-3_2000_b1000.pt"),
          "2,500 steps": os.path.join(CK24, "wsd_1e-3_2000_b2000.pt"),
          "2,500 wd0.1": os.path.join(CK25, "wsd_1e-3_2000_wd0.1_b2000.pt"),
          "2,500 o7": os.path.join(CK25, "wsd_1e-3_2000_o7_b2000.pt"), "2,500 wd0.1 o7": os.path.join(CK25, "wsd_1e-3_2000_wd0.1_o7_b2000.pt"),
          "2,500 o11": os.path.join(CK25, "wsd_1e-3_2000_o11_b2000.pt"), "2,500 wd0.1 o11": os.path.join(CK25, "wsd_1e-3_2000_wd0.1_o11_b2000.pt")}
src = np.array([it["src"] for it in IT])
res, right = {}, {}
HWc = E.windows(c["held_b"]); GWc = E.windows(c["glass"])[:128]
print(f"\n{'model':16} {'LG loss':>8} {'CH loss':>8} | {'LG sum':>7} {'LG mean':>7} {'CH sum':>7} {'CH mean':>7}  95% (LG sum, items)")
for name, f in list(MODELS.items()) + [("unigram", None), ("bigram", None)]:
    if f is None:
        S, Mn = score_counts(name); lg = ch = float("nan")
    else:
        tt = E.load(f)
        S, Mn = score_model(tt)
        with torch.no_grad():
            lg, ch = float(tt.loss(torch.tensor(GWc))), float(tt.loss(torch.tensor(HWc)))
    rs, rm = S.argmax(1) == 0, Mn.argmax(1) == 0
    right[name] = dict(sum=rs, mean=rm)
    b = E.bootstrap(rs[src == "glass"].astype(float))
    res[name] = dict(glass_loss=lg, carroll_loss=ch, glass_sum=float(rs[src == "glass"].mean()), glass_mean=float(rm[src == "glass"].mean()),
                     carroll_sum=float(rs[src == "carroll"].mean()), carroll_mean=float(rm[src == "carroll"].mean()), ci=(b["low"], b["high"]))
    q = res[name]
    print(f"{name:16} {lg:8.4f} {ch:8.4f} | {q['glass_sum']:7.3f} {q['glass_mean']:7.3f} {q['carroll_sum']:7.3f} {q['carroll_mean']:7.3f}  {b['low']:.3f} to {b['high']:.3f}", flush=True)
print("(LG loss: first 128 Looking-Glass windows; CH loss: all 31 Carroll held-out windows; accuracy = share of items whose true word scores highest)")

def mcnemar(a, b, label):
    d = a.astype(int) - b.astype(int)
    st = E.sign_test(d)
    pb = E.paired(a.astype(float), b.astype(float))
    print(f"  {label}: {a.mean():.3f} against {b.mean():.3f}; items only the first gets right {int((d > 0).sum())}, only the second {int((d < 0).sum())}; "
          f"sign test p = {st['p']:.3f}; paired 95% for the difference {pb['low']:+.3f} to {pb['high']:+.3f}")
    return dict(a=float(a.mean()), b=float(b.mean()), only_a=int((d > 0).sum()), only_b=int((d < 0).sum()), p=st["p"], low=pb["low"], high=pb["high"])
print("\npaired comparisons on all 550 items (sum scoring):")
out = dict(results=res, n_items=len(IT), tokens_true=float(ntok[:, 0].mean()), tokens_distr=float(ntok[:, 1:].mean()))
out["pairs"] = {}
for a, b in (("2,500 steps", "1,250 steps"), ("1,250 steps", "625 steps"), ("2,500 wd0.1", "2,500 steps"), ("2,500 wd0.1 o7", "2,500 o7"),
             ("2,500 wd0.1 o11", "2,500 o11"), ("2,500 o7", "2,500 steps"), ("2,500 steps", "bigram")):
    out["pairs"][f"{a} vs {b}"] = mcnemar(right[a]["sum"], right[b]["sum"], f"{a} vs {b}")
mcnemar(right["2,500 steps"]["mean"], right["2,500 steps"]["sum"], "2,500 steps, mean against sum scoring")
# signal and noise across the six 2,500-step models
six = [k for k in MODELS if k.startswith("2,500")]
acc = np.array([right[k]["sum"].mean() for k in six]); los = np.array([res[k]["glass_loss"] for k in six])
noacc = np.array([right[k]["sum"].mean() for k in six if "wd" not in k]); wdacc = np.array([right[k]["sum"].mean() for k in six if "wd" in k])
nol = np.array([res[k]["glass_loss"] for k in six if "wd" not in k]); wdl = np.array([res[k]["glass_loss"] for k in six if "wd" in k])
print(f"\nsix 2,500-step models (no decay / AdamW 0.1, three orders each):")
print(f"  cloze accuracy (all 550, sum): no decay {', '.join(f'{x:.3f}' for x in noacc)} (mean {noacc.mean():.3f}); decay {', '.join(f'{x:.3f}' for x in wdacc)} (mean {wdacc.mean():.3f})")
print(f"  Looking-Glass loss (128 windows): no decay {', '.join(f'{x:.4f}' for x in nol)} (mean {nol.mean():.4f}); decay {', '.join(f'{x:.4f}' for x in wdl)} (mean {wdl.mean():.4f})")
print(f"  signal/noise: loss: gap of means {nol.mean() - wdl.mean():.4f} against order sd {np.sqrt((nol.var(ddof=1) + wdl.var(ddof=1)) / 2):.4f}; "
      f"accuracy: gap {wdacc.mean() - noacc.mean():.4f} against order sd {np.sqrt((noacc.var(ddof=1) + wdacc.var(ddof=1)) / 2):.4f}")
out["six"] = dict(noacc=noacc.tolist(), wdacc=wdacc.tolist(), nol=nol.tolist(), wdl=wdl.tolist())
three = ["625 steps", "1,250 steps", "2,500 steps"]
print("  loss and accuracy along one run: " + "; ".join(f"{k} LG loss {res[k]['glass_loss']:.4f}, accuracy {right[k]['sum'].mean():.3f}" for k in three))
save_json("bench", out)
np.savez_compressed(os.path.join(HERE, "outputs", "bench_right.npz"), **{k.replace(",", "").replace(" ", "_"): v["sum"] for k, v in right.items()})
