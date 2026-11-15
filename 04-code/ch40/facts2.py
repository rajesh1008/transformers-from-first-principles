"""Chapter 40, 40.6-40.7: what removing or moving parts does to the taught fact (Chapter 38's fine-tuned Tiny, the 30 prompts of facts.py).
(a) zero each layer's attention (all four heads) and each layer's feed-forward output, the drop in mean log p(verb) at the site;
(b) zero the top-k heads by attention share, the top-k by single-head ablation drop, and k random heads (5 draws), k = 1, 4, 8, 16, 32, 48;
(c) which token each importance method names, as text; (d) patch the site's rows from another animal's prompt at layer l (0 .. 12):
the share of 200 recipient-donor pairs (different verbs) whose greedy next token becomes the donor's verb, and the recipient's verb.
32-bit, seed 1865."""
from lib import *
import numpy as np, collections
torch.set_num_threads(1)
exec(open("facts_common.py").read())
fj = load_json("facts"); abl = np.array(fj["ablation"]); att = np.array(fj["att_share"]); NL, H = 12, 4
full = np.mean([verb_lp(ft, P[a]) for a in names]); print(f"mean log p(verb) at the site, nothing removed: {full:.3f}")
def mean_lp(**kw): 
    out = []
    for a in names:
        e = P[a]
        with torch.no_grad(): lg = forward(ft, e["ids"][:e["site"] + 1], **kw)["logits"][0, -1]
        out.append(float(torch.log_softmax(lg, -1)[e["verb"]]))
    return np.mean(out)
print("(a) layer by layer, drop in mean log p(verb) when the layer's attention (all four heads) / feed-forward output is zeroed:")
LA = []; LM = []
for l in range(NL):
    da = full - mean_lp(ablate={(l, h) for h in range(H)}); dm = full - mean_lp(mlp_off={l}); LA.append(da); LM.append(dm)
    print(f"   layer {l:2d}: attention {da:+7.3f}   feed-forward {dm:+7.3f}", flush=True)
print("(b) zero k heads, drop in mean log p(verb) (and verbs still the top choice, of 30):")
def top1(**kw):
    c = 0
    for a in names:
        e = P[a]
        with torch.no_grad(): lg = forward(ft, e["ids"][:e["site"] + 1], **kw)["logits"][0, -1]
        c += int(lg.argmax()) == e["verb"]
    return c
order_att = list(np.argsort(-att.ravel())); order_abl = list(np.argsort(-abl.ravel())); rng = np.random.default_rng(SEED)
B = {}
for k in [1, 4, 8, 16, 32, 48]:
    sets = {"by attention": {(i // H, i % H) for i in order_att[:k]}, "by ablation": {(i // H, i % H) for i in order_abl[:k]}}
    row = {}
    for nm, S in sets.items(): row[nm] = (full - mean_lp(ablate=S), top1(ablate=S))
    rr = []
    for _ in range(5):
        S = {(i // H, i % H) for i in rng.choice(NL * H, size=k, replace=False)}; rr.append((full - mean_lp(ablate=S), top1(ablate=S)))
    row["random (mean of 5)"] = (float(np.mean([r[0] for r in rr])), float(np.mean([r[1] for r in rr]))); B[k] = row
    print(f"   k={k:2d}: " + "; ".join(f"{nm} {v[0]:+7.2f} ({v[1]:g})" for nm, v in row.items()), flush=True)
print("(c) the token each method names as most important (excluding the site's own token), counts over the 30 facts:")
cnt = {m: collections.Counter() for m in ["last-layer attention", "rollout", "gradient x input", "occlusion"]}
for a in names:
    e = P[a]; ids = e["ids"][:e["site"] + 1]; n = len(ids)
    with torch.no_grad(): A = forward(ft, ids, want_att=True)["att"]
    last = A[-1][0].mean(0)[-1].numpy(); M = np.eye(n)
    for l in range(NL):
        Al = 0.5 * A[l][0].mean(0).numpy() + 0.5 * np.eye(n); Al /= Al.sum(-1, keepdims=True); M = Al @ M
    g = np.zeros(n)
    for i in range(n):
        sc = torch.tensor(1.0, requires_grad=True); row = ft.t["E"][ids[i]] * sc + ft.t["P"][i]
        torch.log_softmax(forward(ft, ids, patch=(0, i, row))["logits"][0, -1], -1)[e["verb"]].backward(); g[i] = abs(float(sc.grad))
    occ = np.zeros(n); b = verb_lp(ft, e)
    for i in range(n):
        alt = list(ids); alt[i] = T.encode(" and")[0]
        with torch.no_grad(): occ[i] = b - float(torch.log_softmax(forward(ft, alt)["logits"][0, -1], -1)[e["verb"]])
    for m, s in [("last-layer attention", last), ("rollout", M[-1]), ("gradient x input", g), ("occlusion", occ)]:
        s = np.array(s, dtype=float); s[-1] = -1e9; i = int(np.argmax(s))
        where = "question" if i < e["q"] + 3 else "answer"
        cnt[m][f"{T.decode([ids[i]]).strip() or '(space)'!r} ({where})" if ids[i] not in e["span"] or i >= e["q"] + 3 else "the question's animal name"] += 1
for m, c in cnt.items(): print(f"   {m:21s} " + ", ".join(f"{k} {v}" for k, v in c.most_common(5)))
print("(d) patching the site's rows from another animal's prompt at layer l; share of 200 pairs whose greedy next token becomes the donor's verb / stays the recipient's:")
pairs = []
while len(pairs) < 200:
    a, b = rng.choice(names, 2, replace=False)
    if facts[a][0] != facts[b][0]: pairs.append((a, b))
D_rows = {}
for a in names:
    e = P[a]
    with torch.no_grad(): D_rows[a] = torch.stack([x[0, e["site"]] for x in forward(ft, e["ids"][:e["site"] + 1])["resid"]])
PT = []
for l in range(13):
    don = rec = 0
    for a, b in pairs:
        e = P[a]; ids = e["ids"][:e["site"] + 1]
        with torch.no_grad():
            if l < 12: lg = forward(ft, ids, patch=(l, e["site"], D_rows[b][l]))["logits"][0, -1]
            else: lg = ft_final(D_rows[b][12])
        t = int(lg.argmax()); don += t == P[b]["verb"]; rec += t == e["verb"]
    PT.append((don / len(pairs), rec / len(pairs))); print(f"   layer {l:2d}: donor's verb {don/len(pairs):.3f}  recipient's verb {rec/len(pairs):.3f}", flush=True)
save_json("facts2", dict(full=float(full), layer_att=[float(x) for x in LA], layer_ff=[float(x) for x in LM], khead={str(k): v for k, v in B.items()}, patch=PT,
                         importance={m: dict(c) for m, c in cnt.items()}))
