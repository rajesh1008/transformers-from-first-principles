"""Chapter 27, sections 27.5 to 27.10, with tiny/generate.py.  Chapter 24's 2,500-step Tiny, 32-bit, one thread,
every draw from tiny/seed.py's 'sample' stream (master seed 1865 unless stated).
Part F matched settings (top-k, top-p, min-p set to keep about the same number of tokens; three prompts, two seeds)
Part G repetition: greedy for 200 tokens, the loop, and four penalties
Part H beam search, widths 1 to 16
Part I stopping: a blank line as the stop, lengths and runaways
Part J constrained choice: whole-option probabilities against token-by-token masking
Part K measuring generated text: distinct n-grams and surprise against held-out text"""
from _common import *
import torch, re, time, math
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G
from _pool import corpora

tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok()
c = corpora()
pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
PROMPTS = ["the cat sat on the", "Alice was beginning to", '"I don\'t know," said the']
out = {}

def novel_share(texts):
    ws = [w.lower() for x in texts for w in re.findall(r"[A-Za-z]+", x)]
    return sum(w not in pool_words for w in ws) / max(1, len(ws))

def run_rule(kw, seeds=(1865, 7), prompts=PROMPTS, k=10, n=40):
    texts, seqs, kept, surp, rep = [], [], [], [], []
    for sd in seeds:
        rng = streams(sd)["sample"]
        for pr in prompts:
            p = t.encode(pr)
            for _ in range(k):
                ids, info = G.generate(tt, p, n, rng, **kw)
                new = ids[len(p):]
                seqs.append(new); texts.append(t.decode(new))
                kept += info["kept"]; surp += info["surprise"]; rep.append(G.repeated_runs(new))
    return dict(novel=novel_share(texts), kept=float(np.mean(kept)), surprise=float(np.mean(surp)),
                rep=float(np.mean(rep)), d2=G.distinct_n(seqs, 2), first=texts[0], n=len(texts))

t0 = time.time()
print("Part F: matched settings (3 prompts x 2 seeds x 10 continuations x 40 tokens per rule)")
base = run_rule(dict(min_p=0.1))
target = base["kept"]
print(f"  min-p 0.1 keeps {target:.1f} tokens on average; calibrating top-k and top-p to match")
cal = {}
for name, kws in [("top-k", [dict(top_k=k) for k in (10, 15, 20)]), ("top-p", [dict(top_p=q) for q in (0.5, 0.6, 0.7)])]:
    best = None
    for kw in kws:
        r = run_rule(kw, seeds=(1865,), prompts=PROMPTS[:1], k=10)
        print(f"    {kw} keeps {r['kept']:.1f}")
        if best is None or abs(r["kept"] - target) < abs(best[1] - target):
            best = (kw, r["kept"])
    cal[name] = best[0]
F = {"min-p 0.1": base}
for label, kw in [(f"top-k {cal['top-k']['top_k']}", cal["top-k"]), (f"top-p {cal['top-p']['top_p']}", cal["top-p"]),
                  ("T 0.7", dict(T=0.7)), ("T 1.0", dict(T=1.0)), ("top-p 0.9", dict(top_p=0.9))]:
    F[label] = run_rule(kw)
for label, r in F.items():
    print(f"  {label:<12} kept {r['kept']:7.1f}  not-in-pool {r['novel']:.3f}  surprise {r['surprise']:.3f}  "
          f"distinct-2 {r['d2']:.3f}  repeated 4-runs {r['rep']:.3f}")
# per-seed split for the three matched rules
print("  per seed (not-in-pool):")
for label in list(F)[:3]:
    kw = {"min-p 0.1": dict(min_p=0.1)}.get(label) or (cal["top-k"] if label.startswith("top-k") else cal["top-p"])
    vals = [run_rule(kw, seeds=(sd,))["novel"] for sd in (1865, 7)]
    print(f"    {label:<12} seed 1865 {vals[0]:.3f}  seed 7 {vals[1]:.3f}")
    F[label]["per_seed"] = vals
out["F"] = F
print(f"  ({time.time() - t0:.0f} s)")

print("\nPart G: repetition, greedy for 200 tokens")
G_ = {}
def loop_stats(new):
    # smallest period p such that the last 60 tokens repeat with period p
    tail = new[-60:]
    per = next((p for p in range(1, 31) if tail[p:] == tail[:-p]), None)
    start = None
    if per:
        i = len(new) - 1
        while i - per >= 0 and new[i] == new[i - per]:
            i -= 1
        start = i + 1
    return per, start
for pr in PROMPTS:
    p = t.encode(pr)
    ids, _ = G.generate(tt, p, 200, None)
    new = ids[len(p):]
    per, start = loop_stats(new)
    loop_text = t.decode(new[start:start + per]) if per else None
    print(f"  {pr!r}: loop period {per} tokens, from generated position {start}; repeated 4-runs {G.repeated_runs(new):.3f}; "
          f"loop {loop_text!r}")
    G_[pr] = dict(period=per, start=start, rep=G.repeated_runs(new), loop=loop_text, text=t.decode(new[:80]))
print("  penalties (greedy and T 0.7 sampling, 3 prompts, 200 tokens; sampling 5 per prompt, seed 1865)")
PEN = [("none", {}), ("rep 1.3 (CTRL)", dict(rep=1.3)), ("presence 1.0", dict(presence=1.0)),
       ("frequency 0.5", dict(frequency=0.5)), ("no-repeat 3-runs", dict(no_repeat=3))]
pen = {}
for label, kw in PEN:
    gr, sm = [], []
    for pr in PROMPTS:
        p = t.encode(pr)
        ids, info = G.generate(tt, p, 200, None, **kw); gr.append((ids[len(p):], info))
    rng = streams(SEED)["sample"]
    for pr in PROMPTS:
        p = t.encode(pr)
        for _ in range(5):
            ids, info = G.generate(tt, p, 200, rng, T=0.7, **kw); sm.append((ids[len(p):], info))
    row = {}
    for nm, rs in (("greedy", gr), ("T0.7", sm)):
        row[nm] = dict(rep=float(np.mean([G.repeated_runs(s) for s, _ in rs])),
                       novel=novel_share([t.decode(s) for s, _ in rs]),
                       surprise=float(np.mean([np.mean(i["surprise"]) for _, i in rs])))
    row["greedy_text"] = t.decode(gr[0][0][:60])
    pen[label] = row
    print(f"  {label:<17} greedy: rep4 {row['greedy']['rep']:.3f} novel {row['greedy']['novel']:.3f} surprise {row['greedy']['surprise']:.3f}"
          f" | T0.7: rep4 {row['T0.7']['rep']:.3f} novel {row['T0.7']['novel']:.3f} surprise {row['T0.7']['surprise']:.3f}")
for label in pen:
    print(f"    {label:<17} {pen[label]['greedy_text']!r}")
out["G"] = dict(loops=G_, penalties=pen)
print(f"  ({time.time() - t0:.0f} s)")

print("\nPart H: beam search, 30 tokens")
H = {}
for w in (1, 2, 4, 8, 16):
    row = []
    for pr in PROMPTS:
        p = t.encode(pr)
        beams = G.beam_search(tt, p, 30, w)
        lp, ids = beams[0]
        new = ids[len(p):]
        row.append(dict(lp=lp, rep=G.repeated_runs(new), text=t.decode(new),
                        distinct_beams=len({tuple(b[1]) for b in beams})))
    H[w] = row
    print(f"  width {w:>2}: total log-prob " + " ".join(f"{r['lp']:8.2f}" for r in row) +
          f"   repeated 4-runs {np.mean([r['rep'] for r in row]):.3f}")
for w in (1, 8):
    print(f"    width {w} on prompt 1: {H[w][0]['text']!r}")
    print(f"    width {w} on prompt 3: {H[w][2]['text']!r}")
rng = streams(SEED)["sample"]
lps = []
for _ in range(20):
    p = t.encode(PROMPTS[0]); ids, info = G.generate(tt, p, 30, rng, T=1.0); lps.append(-sum(info["surprise"]))
print(f"  20 samples at T 1.0 on prompt 1: total log-prob mean {np.mean(lps):.2f}, best {max(lps):.2f}")
out["H"] = {str(k): v for k, v in H.items()}; out["H_samples"] = lps
print(f"  ({time.time() - t0:.0f} s)")

print("\nPart I: stopping at a blank line (tokens containing two line breaks), cap 200 tokens, 30 samples per rule, prompt 1")
STOP = tuple(i for i in range(2048) if "\n\n" in t.decode([i]))
I = {}
for label, kw in [("T 0.7", dict(T=0.7)), ("T 1.0", dict(T=1.0)), ("T 1.3", dict(T=1.3)), ("min-p 0.1", dict(min_p=0.1)),
                  ("T 1.3 + min-p 0.1", dict(T=1.3, min_p=0.1))]:
    rng = streams(SEED)["sample"]
    L, capped = [], 0
    for _ in range(30):
        p = t.encode(PROMPTS[0])
        ids, info = G.generate(tt, p, 200, rng, stop=STOP, **kw)
        L.append(len(ids) - len(p)); capped += not info["stopped"]
    I[label] = dict(median=float(np.median(L)), mean=float(np.mean(L)), capped=capped)
    print(f"  {label:<18} median length {np.median(L):5.0f}  mean {np.mean(L):6.1f}  reached the cap {capped}/30")
ids, info = G.generate(tt, t.encode(PROMPTS[0]), 200, None, stop=STOP)
print(f"  greedy: stopped {info['stopped']} after {len(info['kept'])} tokens")
I["greedy"] = dict(stopped=info["stopped"], n=len(info["kept"]))
out["I"] = I; out["I_stop_tokens"] = list(STOP)

print("\nPart J: constrained choice of a name")
PR = '"Come here, my dear," said'
names = [" Alice", " Bruno", " Sylvie", " Balbus", " the Hatter", " the Queen"]
opts = [t.encode(nm) for nm in names]
lp = G.option_logprobs(tt, t.encode(PR), opts)
exact = np.exp(lp - lp.max()); exact /= exact.sum()
local = G.local_mask_probs(tt, t.encode(PR), opts)
pertok = lp / np.array([len(o) for o in opts])
print(f"  prompt {PR!r}")
for nm, o, a, b, cc, d in zip(names, opts, lp, exact, local, pertok):
    print(f"  {nm!r:<13} tokens {len(o)}  total log-prob {a:7.3f}  per token {d:6.3f}  share among options {b:.3f}  "
          f"token-by-token masking {cc:.3f}")
rng = streams(SEED)["sample"]
draws = []
for _ in range(400):
    pre, live = [], list(range(len(opts)))
    while True:
        done = [i for i in live if len(opts[i]) == len(pre)]
        if done:
            draws.append(done[0]); break
        s = G.next_scores(tt, t.encode(PR) + pre)
        allowed = sorted({opts[i][len(pre)] for i in live})
        q, _ = G.shares(s, allowed=allowed)
        nxt = int(rng.choice(len(q), p=q)); pre.append(nxt)
        live = [i for i in live if opts[i][len(pre) - 1] == nxt]
cnt = np.bincount(draws, minlength=len(opts)) / len(draws)
print("  400 masked draws: " + ", ".join(f"{nm.strip()} {v:.3f}" for nm, v in zip(names, cnt)))
free = G.generate(tt, t.encode(PR), 4, None)[0]
print(f"  unconstrained greedy: {t.decode(free[len(t.encode(PR)):])!r}")
out["J"] = dict(names=names, lp=lp.tolist(), exact=exact.tolist(), local=local.tolist(), draws=cnt.tolist())

print("\nPart K: measuring generated text (prompt 1, 20 continuations x 100 tokens per rule, seed 1865)")
from tiny.eval import windows, score
from _pool import corpora as _c
K = {}
for label, kw in [("greedy", None), ("T 0.7", dict(T=0.7)), ("T 1.0", dict(T=1.0)), ("min-p 0.1", dict(min_p=0.1)),
                  ("top-p 0.9", dict(top_p=0.9))]:
    rng = streams(SEED)["sample"]
    seqs, surp = [], []
    for _ in range(20 if kw else 1):
        p = t.encode(PROMPTS[0])
        ids, info = G.generate(tt, p, 100, rng if kw else None, **(kw or {}))
        seqs.append(ids[len(p):]); surp += info["surprise"]
    K[label] = dict(d1=G.distinct_n(seqs, 1), d2=G.distinct_n(seqs, 2), d4=G.distinct_n(seqs, 4), surprise=float(np.mean(surp)),
                    novel=novel_share([t.decode(s) for s in seqs]), rep=float(np.mean([G.repeated_runs(s) for s in seqs])))
    r = K[label]
    print(f"  {label:<10} distinct-1 {r['d1']:.3f}  distinct-2 {r['d2']:.3f}  distinct-4 {r['d4']:.3f}  "
          f"mean surprise {r['surprise']:.3f}  not-in-pool {r['novel']:.3f}  repeated 4-runs {r['rep']:.3f}")
held = c.get("heldout") if isinstance(c, dict) else None
keys = list(c.keys()) if isinstance(c, dict) else []
print(f"  corpora keys: {keys}")
hk = next((k for k in keys if "held" in k.lower() or "carroll" in k.lower()), None)
if hk:
    W = windows(np.asarray(c[hk]), 129)[:40]
    sc = score(tt, W)
    hs = float(np.concatenate(sc["surprise"]).mean()) if isinstance(sc["surprise"], list) else float(np.mean(sc["surprise"]))
    print(f"  held-out text ({hk}, 40 windows): mean surprise {hs:.3f}")
    held_seqs = [list(map(int, w[1:101])) for w in W[:20]]
    print(f"  held-out 20 x 100 tokens: distinct-1 {G.distinct_n(held_seqs,1):.3f} distinct-2 {G.distinct_n(held_seqs,2):.3f} "
          f"distinct-4 {G.distinct_n(held_seqs,4):.3f}; repeated 4-runs {np.mean([G.repeated_runs(s) for s in held_seqs]):.3f}")
    K["held"] = dict(surprise=hs, d1=G.distinct_n(held_seqs, 1), d2=G.distinct_n(held_seqs, 2), d4=G.distinct_n(held_seqs, 4))
out["K"] = K
print(f"  ({time.time() - t0:.0f} s)")
save_json("experiments", out)
