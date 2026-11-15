"""Chapter 25, sections 25.1 to 25.7: the finished 2,500-step Tiny of Chapter 24 (WSD 1e-3, branch at 2,000 decayed
to 2,500; ../ch24/outputs/ckpt/wsd_1e-3_2000_b2000.pt) evaluated beyond one loss.  32-bit, one thread, seed 1865.
  Part 1  perplexity by hand on the running sentence, and on the held-out texts
  Part 2  loss by how often the true token occurs in the pool (frequency bands), against the unigram and bigram
  Part 3  calibration of next-token probabilities: top-1 and all-token reliability tables, ECE; on the Carroll
          held-out set, Looking-Glass and pool text; and one temperature fitted on one text, tried on another
  Part 4  held-out variance: 200 draws of 16 windows; every window; bootstrap intervals over windows against the
          naive interval that treats every prediction as independent
Texts: Carroll held-out = all 31 non-overlapping 129-token windows of the 4,000 held-out tokens (3,968 predictions);
Looking-Glass = every non-overlapping window of the book (Tiny never read it); pool = 31 windows drawn from the
Carroll pool (seed 1865), text the model read about 6.5 times.  Writes outputs/evaluate.json and the per-prediction
arrays outputs/scores_*.npz used by compare.py and the figures."""
import _common
import os, math, json
import numpy as np, torch
torch.set_num_threads(1)
from tiny import eval as E
from _pool import corpora, heldout_sets, POOL_BOOKS
from _common import HERE, SEED, sentence_ids, tok, save_json
CK24 = os.path.join(HERE, "..", "ch24", "outputs", "ckpt")
c = corpora()
pool = np.concatenate([np.asarray(c["books"][b]) for b in POOL_BOOKS])
V = 2048
TEXTS = dict(carroll=E.windows(c["held_b"]), glass=E.windows(c["glass"]))
r = np.random.default_rng(SEED)
TEXTS["pool"] = np.stack([pool[a:a + 129] for a in r.integers(0, len(pool) - 129, 31)])
G16, HB16 = heldout_sets()
out = {}
print("texts:", {k: f"{v.shape[0]} windows, {v.shape[0] * 128:,} predictions" for k, v in TEXTS.items()})

# ---- the models ----------------------------------------------------------------------------------------------
M = {"2500": E.load(os.path.join(CK24, "wsd_1e-3_2000_b2000.pt")), "1250": E.load(os.path.join(CK24, "wsd_1e-3_2000_b1000.pt"))}
with torch.no_grad():
    chk = float(M["2500"].loss(torch.tensor(HB16)))
print(f"check: 2,500-step model on Chapter 24's 16 Carroll windows {chk:.4f} (Chapter 24: 3.7296)")
out["check16"] = chk

# ---- Part 1: perplexity ----------------------------------------------------------------------------------------
print("\nPart 1. perplexity")
T = tok(); s = sentence_ids(T)
with torch.no_grad():
    lp = torch.log_softmax(M["2500"].scores(torch.tensor(s[:-1])).double(), -1)
pr = [float(lp[i, s[i + 1]].exp()) for i in range(len(s) - 1)]
su = [-math.log(p) for p in pr]
print("  running sentence tokens:", [T.decode([i]) for i in s])
for i, (p, x) in enumerate(zip(pr, su)):
    print(f"   predict {T.decode([s[i + 1]])!r:8} after {T.decode(s[:i + 1])!r:22} p = {p:.4f}  surprise {x:.4f}")
L = sum(su) / len(su)
print(f"  average surprise {L:.4f} nats; perplexity e^{L:.4f} = {math.exp(L):.1f}; "
      f"geometric mean of the probabilities {math.exp(np.log(pr).mean()):.5f} = 1/{1 / math.exp(np.log(pr).mean()):.1f}")
out["sentence"] = dict(p=pr, surprise=su, loss=L, ppl=math.exp(L))
S = {}
for name in ("2500", "1250"):
    for t, W in TEXTS.items():
        S[(name, t)] = E.score(M[name], W)
        np.savez_compressed(os.path.join(HERE, "outputs", f"scores_{name}_{t}.npz"),
                            **{k: v for k, v in S[(name, t)].items() if isinstance(v, np.ndarray)})
for t in TEXTS:
    a, b = S[("2500", t)], S[("1250", t)]
    print(f"  {t:8} 2,500 steps: loss {a['surprise'].mean():.4f} perplexity {math.exp(a['surprise'].mean()):7.1f} "
          f"top-1 right {a['correct'].mean():.4f} | 1,250 steps: loss {b['surprise'].mean():.4f} perplexity {math.exp(b['surprise'].mean()):7.1f} top-1 {b['correct'].mean():.4f}")
out["texts"] = {t: {n: dict(loss=float(S[(n, t)]["surprise"].mean()), acc=float(S[(n, t)]["correct"].mean()),
                            entropy=float(S[(n, t)]["entropy"].mean())) for n in M} for t in TEXTS}
bpc = {}
for t in ("carroll", "glass"):
    W = TEXTS[t]
    chars = sum(len(T.decode(list(w[1:]))) for w in W)
    bpc[t] = S[("2500", t)]["surprise"].sum() / math.log(2) / chars
    print(f"  {t}: {chars:,} characters predicted, {W.shape[0] * 128 / chars:.3f} tokens per character; 2,500 steps {bpc[t]:.4f} bits per character")
out["bpc"] = bpc

# ---- Part 2: frequency bands --------------------------------------------------------------------------------
print("\nPart 2. loss by how often the true token occurs in the pool")
cnt = np.bincount(pool, minlength=V)
pu = (cnt + 1) / (cnt.sum() + V)
big = np.zeros((V, V)); np.add.at(big, (pool[:-1], pool[1:]), 1); rows = big.sum(1)
bands = [(0, 0), (1, 9), (10, 99), (100, 999), (1000, 9999), (10000, 10 ** 9)]
names = ["never", "1-9", "10-99", "100-999", "1,000-9,999", "10,000+"]
out["bands"] = {}
for t in ("carroll", "glass", "pool"):
    W = TEXTS[t]
    prev, nxt = W[:, :-1].ravel(), W[:, 1:].ravel()
    uni_s = -np.log(pu[nxt]); big_s = -np.log((big[prev, nxt] + 0.01) / (rows[prev] + 0.01 * V))
    a, b = S[("2500", t)]["surprise"], S[("1250", t)]["surprise"]
    tot = a.sum()
    print(f"  {t}: {len(nxt):,} predictions; Tiny 2,500 {a.mean():.4f} | 1,250 {b.mean():.4f} | unigram {uni_s.mean():.4f} | bigram {big_s.mean():.4f}")
    print(f"    {'pool count':12} {'tokens':>7} {'share':>6} {'Tiny2500':>8} {'Tiny1250':>8} {'unigram':>8} {'bigram':>8} {'share of loss':>13} {'top-1 right':>11}")
    rowsout = []
    for (lo, hi), nm in zip(bands, names):
        m = (cnt[nxt] >= lo) & (cnt[nxt] <= hi)
        if m.sum() == 0:
            print(f"    {nm:12} {0:7d}"); rowsout.append(dict(band=nm, n=0)); continue
        rowsout.append(dict(band=nm, n=int(m.sum()), share=float(m.mean()), tiny2500=float(a[m].mean()), tiny1250=float(b[m].mean()),
                            unigram=float(uni_s[m].mean()), bigram=float(big_s[m].mean()), loss_share=float(a[m].sum() / tot),
                            acc=float(S[("2500", t)]["correct"][m].mean())))
        q = rowsout[-1]
        print(f"    {nm:12} {q['n']:7d} {q['share']:6.3f} {q['tiny2500']:8.4f} {q['tiny1250']:8.4f} {q['unigram']:8.4f} {q['bigram']:8.4f} {q['loss_share']:13.3f} {q['acc']:11.4f}")
    out["bands"][t] = rowsout
# the commonest tokens by name
order = np.argsort(-cnt)
print("  the ten commonest pool tokens:", ", ".join(f"{T.decode([int(i)])!r} {cnt[i]:,}" for i in order[:10]))
print(f"  tokens with 10,000+ pool occurrences: {int((cnt >= 10000).sum())}; with none: {int((cnt == 0).sum())}")
a = S[("2500", "carroll")]; nxt = TEXTS["carroll"][:, 1:].ravel()
top10 = np.isin(nxt, order[:10])
print(f"  Carroll held-out: the ten commonest tokens are {top10.mean():.3f} of the predictions and carry {a['surprise'][top10].sum() / a['surprise'].sum():.3f} of the loss; "
      f"surprise on them {a['surprise'][top10].mean():.4f}, on the rest {a['surprise'][~top10].mean():.4f}")
out["top10"] = dict(share=float(top10.mean()), loss_share=float(a["surprise"][top10].sum() / a["surprise"].sum()))
# loss by position in the window (context length)
pos = np.tile(np.arange(128), TEXTS["carroll"].shape[0])
print("  Carroll held-out, surprise by seat (how much context the prediction had):",
      ", ".join(f"seats {lo}-{hi - 1}: {a['surprise'][(pos >= lo) & (pos < hi)].mean():.4f}" for lo, hi in ((0, 1), (1, 8), (8, 32), (32, 64), (64, 128))).replace("seats 0-0", "seat 0"))
out["by_seat"] = {f"{lo}-{hi - 1}": float(a["surprise"][(pos >= lo) & (pos < hi)].mean()) for lo, hi in ((0, 1), (1, 8), (8, 32), (32, 64), (64, 128))}

# ---- Part 3: calibration ------------------------------------------------------------------------------------
print("\nPart 3. calibration of next-token probabilities")
out["calib"] = {}
for name in ("2500", "1250"):
    for t in TEXTS:
        q = S[(name, t)]
        out["calib"][f"{name}_{t}"] = dict(top1=E.reliability(q["top1"]), all=E.reliability(q["all"]),
                                          ece_top1=E.ece(q["top1"]), ece_all=E.ece(q["all"]),
                                          conf=float(q["ptop"].mean()), acc=float(q["correct"].mean()))
        print(f"  {name:>4} steps, {t:8}: average top-1 probability {q['ptop'].mean():.4f}, top-1 right {q['correct'].mean():.4f}; "
              f"ECE top-1 {E.ece(q['top1']):.4f}, ECE all-token {E.ece(q['all']):.2e}")
print("  reliability table, 2,500 steps, Carroll held-out, top-1 (bin, predictions, average confidence, how often right):")
for lo, hi, n, cf, ac in E.reliability(S[("2500", "carroll")]["top1"]):
    if n:
        print(f"    {lo:.1f}-{min(hi, 1):.1f}  {n:5d}  {cf:.3f}  {ac:.3f}")
print("  reliability table, 2,500 steps, Carroll held-out, all-token (every probability of every prediction):")
for lo, hi, n, cf, ac in E.reliability(S[("2500", "carroll")]["all"]):
    if n:
        print(f"    {lo:.1f}-{min(hi, 1):.1f}  {n:9d}  {cf:.4f}  {ac:.4f}")
# temperature: fit on the first half of the Looking-Glass windows, test on the second half and on Carroll
gW = TEXTS["glass"]; half = len(gW) // 2
def loss_at(W, T_):
    q = E.score(M["2500"], W, temperature=T_)
    return q
grid = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
fit = {T_: float(loss_at(gW[:half], T_)["surprise"].mean()) for T_ in grid}
Tb = min(fit, key=fit.get)
print("  temperature on the first half of Looking-Glass (loss):", ", ".join(f"T {k}: {v:.4f}" for k, v in fit.items()), f"-> best {Tb}")
res = {}
for nm, W in (("glass second half", gW[half:]), ("carroll", TEXTS["carroll"]), ("pool", TEXTS["pool"])):
    q1, qT = loss_at(W, 1.0), loss_at(W, Tb)
    res[nm] = dict(loss1=float(q1["surprise"].mean()), lossT=float(qT["surprise"].mean()), ece1=E.ece(q1["top1"]), eceT=E.ece(qT["top1"]),
                   acc1=float(q1["correct"].mean()), accT=float(qT["correct"].mean()))
    print(f"  {nm:18}: T 1.0 loss {res[nm]['loss1']:.4f} ECE {res[nm]['ece1']:.4f} | T {Tb} loss {res[nm]['lossT']:.4f} ECE {res[nm]['eceT']:.4f} | top-1 right {res[nm]['acc1']:.4f} both")
out["temperature"] = dict(grid=fit, best=Tb, results=res)

# ---- Part 4: held-out variance ------------------------------------------------------------------------------
print("\nPart 4. how much the held-out number depends on which text is held out")
held = c["held_b"]
g = np.random.default_rng(SEED)
draws = []
for _ in range(200):
    W = np.stack([held[a:a + 129] for a in g.integers(0, len(held) - 129, 16)])
    with torch.no_grad():
        draws.append(float(M["2500"].loss(torch.tensor(W))))
draws = np.array(draws)
print(f"  200 draws of 16 windows (2,048 predictions each): mean {draws.mean():.4f}, sd {draws.std(ddof=1):.4f}, "
      f"min {draws.min():.4f}, max {draws.max():.4f}, middle 95% {np.percentile(draws, 2.5):.4f} to {np.percentile(draws, 97.5):.4f}")
out["draws"] = dict(values=draws.tolist(), mean=float(draws.mean()), sd=float(draws.std(ddof=1)))
for t in ("carroll", "glass"):
    q = S[("2500", t)]
    bw = E.bootstrap(q["surprise"], q["window"]); bt = E.bootstrap(q["surprise"], None)
    sd_tok = q["surprise"].std(ddof=1); n = len(q["surprise"])
    per_w = np.bincount(q["window"], weights=q["surprise"]) / 128
    print(f"  {t:8}: {n:,} predictions in {bw['groups']} windows; loss {bw['mean']:.4f}")
    print(f"    naive (every prediction independent): sd of one surprise {sd_tok:.4f}, standard error {sd_tok / math.sqrt(n):.4f}, 95% {bt['low']:.4f} to {bt['high']:.4f}")
    print(f"    by window (bootstrap of whole windows, 2,000 resamples): standard error {bw['se']:.4f}, 95% {bw['low']:.4f} to {bw['high']:.4f}; "
          f"window losses from {per_w.min():.4f} to {per_w.max():.4f}, sd {per_w.std(ddof=1):.4f}")
    out[f"boot_{t}"] = dict(window=bw, naive=bt, sd_tok=float(sd_tok), window_min=float(per_w.min()), window_max=float(per_w.max()), window_sd=float(per_w.std(ddof=1)))
# how many windows for a given standard error, from the window sd
sdw = out["boot_glass"]["window_sd"]
for target in (0.05, 0.02, 0.01):
    print(f"  Looking-Glass: windows needed for a standard error of {target}: (sd {sdw:.3f} / {target})^2 = {math.ceil((sdw / target) ** 2)}")
save_json("evaluate", out)
