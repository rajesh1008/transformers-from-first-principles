"""Chapter 40, 40.5-40.7: where is a taught fact?  Uses Chapter 38's fine-tuned Tiny (sft_3e-4_600, 600 steps at 3e-4) and the base.
Prompts "User: What did the {a} do?\nTiny: The {a} {verb}", teacher-forced; the SITE is the last token of the answer's animal name
(predicting the verb).  (1) per-head attention share on the prompt's animal-name tokens and zero-ablation effect on log p(verb);
(2) probes for the verb (8 classes) at the site and at the question mark, leave-one-animal-out, base against fine-tuned.
(3) token importance four ways (last-layer attention, rollout, gradient x input, occlusion) for the verb.  32-bit, seed 1865."""
from lib import *
import numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
T = tok(); D = json.load(open(os.path.join(HERE, "..", "ch38", "outputs", "instructions.json")))
facts = D["facts"]; reps = D["reps"]
base, ft = load(CH24), load_ft()
EOT = T.specials["<|endoftext|>"]
def build(a):
    p = T.encode(f"User: What did the {a} do?\nTiny:"); an = T.encode(" " + a); pre = T.encode(" The")
    v = T.encode(" " + facts[a][0])[0]
    ids = p + pre + an + [v]; site = len(p) + len(pre) + len(an) - 1
    # animal tokens in the prompt: find the span of an inside p
    sp = [i for i in range(len(p) - len(an) + 1) if p[i:i + len(an)] == an][0]
    return dict(a=a, ids=ids, site=site, verb=v, span=list(range(sp, sp + len(an))), q=len(p) - 3)
P = {a: build(a) for a in facts}
def verb_lp(tt, e, ablate=()):
    with torch.no_grad(): lg = forward(tt, e["ids"][:e["site"] + 1], ablate=ablate)["logits"][0, e["site"]]
    return float(torch.log_softmax(lg, -1)[e["verb"]])
# which facts does ft know at the site (greedy next token = verb)?
known = [a for a in P if int(forward(ft, P[a]["ids"][:P[a]["site"] + 1])["logits"][0, -1].argmax()) == P[a]["verb"]]
print(f"fine-tuned Tiny, teacher-forced up to the animal name in the answer, predicts the taught verb for {len(known)} of 30 animals; base: " +
      f"{sum(int(forward(base, P[a]['ids'][:P[a]['site']+1])['logits'][0,-1].argmax()) == P[a]['verb'] for a in P)} of 30")
print("mean log p(verb) at the site: base %.3f, fine-tuned %.3f (on the %d known: %.3f)" % (
    np.mean([verb_lp(base, P[a]) for a in P]), np.mean([verb_lp(ft, P[a]) for a in P]), len(known), np.mean([verb_lp(ft, P[a]) for a in known])))
# (1) head attention vs ablation
NL, H = 12, 4; att_share = np.zeros((NL, H)); abl = np.zeros((NL, H)); full = {a: verb_lp(ft, P[a]) for a in known}
for a in known:
    e = P[a]
    with torch.no_grad(): A = forward(ft, e["ids"][:e["site"] + 1], want_att=True)["att"]
    for l in range(NL): att_share[l] += np.array([float(A[l][0, h, e["site"], e["span"]].sum()) for h in range(H)]) / len(known)
for l in range(NL):
    for h in range(H):
        abl[l, h] = np.mean([full[a] - verb_lp(ft, P[a], ablate={(l, h)}) for a in known])   # positive: ablating removes probability
from scipy.stats import spearmanr
rs = spearmanr(att_share.ravel(), abl.ravel())
print("head attention share on the animal name in the question (at the answer's animal name) against zero-ablation drop in log p(verb), 48 heads:")
print("  Spearman rho %.3f (p %.3g)" % (rs.statistic, rs.pvalue))
top_a = np.argsort(-att_share.ravel())[:6]; top_b = np.argsort(-abl.ravel())[:6]; low = np.argsort(abl.ravel())[:3]
fmt = lambda i: f"{i // H}.{i % H}"
print("  top 6 by attention:", ", ".join(f"{fmt(i)} (share {att_share.ravel()[i]:.3f}, drop {abl.ravel()[i]:+.3f})" for i in top_a))
print("  top 6 by ablation :", ", ".join(f"{fmt(i)} (drop {abl.ravel()[i]:+.3f}, share {att_share.ravel()[i]:.3f})" for i in top_b))
print("  overlap of the two top-6 lists:", len(set(top_a) & set(top_b)), "; heads whose ablation drops log p(verb) by more than 1.0 nat:", int((abl > 1).sum()),
      "; whose ablation raises it:", int((abl < -0.05).sum()))
# (2) probes for the verb
verbs = sorted({facts[a][0] for a in facts}); y = np.array([verbs.index(facts[a][0]) for a in P]); names = list(P)
def rows_at(tt, key):
    out = []
    for a in names:
        e = P[a]; ids = e["ids"][:e["site"] + 1]
        with torch.no_grad(): r = forward(tt, ids)["resid"]
        pos = e["site"] if key == "site" else e["q"]
        out.append(torch.stack([x[0, pos] for x in r]))
    return torch.stack(out)       # (30, 13, 128)
def loo(X, y, l):
    hit = 0
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        A = X[tr, l]; mu, sd = A.mean(0), A.std(0) + 1e-6; A = (A - mu) / sd; B = ((X[i:i + 1, l] - mu) / sd)
        w = torch.zeros(128, len(verbs), requires_grad=True); b = torch.zeros(len(verbs), requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-2)
        for _ in range(80): opt.zero_grad(); torch.nn.functional.cross_entropy(A @ w + b, torch.tensor(y[tr])).backward(); opt.step()
        hit += int((B @ w + b).argmax(-1)[0] == y[i])
    return hit / len(y)
R = dict(known=known, att_share=att_share.tolist(), ablation=abl.tolist(), rho=float(rs.statistic), p=float(rs.pvalue), verbs=verbs)
print("verb probes, leave-one-animal-out over 30 animals (8 verbs; majority verb %.3f):" % (np.bincount(y).max() / len(y)))
for key in ["site", "question"]:
    for nm, tt in [("base", base), ("fine-tuned", ft)]:
        X = rows_at(tt, key); accs = [loo(X, y, l) for l in range(13)]; R[f"probe_{key}_{nm}"] = accs
        print(f"  {key:8s} {nm:10s} layers 0..12: " + " ".join(f"{a:.2f}" for a in accs), flush=True)
# (3) token importance
def toks_for(e): return e["ids"][:e["site"] + 1]
res = {m: [] for m in ["last-layer attention", "rollout", "gradient x input", "occlusion"]}
for a in known:
    e = P[a]; ids = toks_for(e); n = len(ids); an = set(e["span"])
    with torch.no_grad(): A = forward(ft, ids, want_att=True)["att"]
    last = A[-1][0].mean(0)[-1].numpy()                                   # mean over heads, from the site
    M = np.eye(n)
    for l in range(NL):
        Al = 0.5 * A[l][0].mean(0).numpy() + 0.5 * np.eye(n); Al /= Al.sum(-1, keepdims=True); M = Al @ M
    ro = M[-1]
    # gradient x input: the derivative of log p(verb) with respect to a scale on each seat's token row (exact at scale 1)
    g = np.zeros(n)
    for i in range(n):
        sc = torch.tensor(1.0, requires_grad=True)
        idx = ids[i]
        row = (ft.t["E"][idx] * sc + ft.t["P"][i])
        lg = forward(ft, ids, patch=(0, i, row))["logits"][0, -1]
        lp = torch.log_softmax(lg, -1)[e["verb"]]; lp.backward(); g[i] = abs(float(sc.grad))
    occ = np.zeros(n); base_lp = verb_lp(ft, e)
    for i in range(n):
        alt = list(ids); alt[i] = T.encode(" and")[0]
        with torch.no_grad(): lg = forward(ft, alt)["logits"][0, -1]
        occ[i] = base_lp - float(torch.log_softmax(lg, -1)[e["verb"]])
    for m, s in [("last-layer attention", last), ("rollout", ro), ("gradient x input", g), ("occlusion", occ)]:
        s = np.array(s, dtype=float); s[-1] = -1e9 if m != "occlusion" else s[-1]    # the site token's own row is excluded except for occlusion
        if m == "occlusion": s[-1] = -1e9
        res[m].append((int(np.argmax(s)) in an, int(np.argmax(s)), float(np.sort(s)[::-1][0])))
print("which prompt/answer token does each method call most important for the verb (excluding the site token itself)? share of the %d known facts where it is a token of the animal name in the question:" % len(known))
for m, v in res.items(): print(f"  {m:21s} {sum(x[0] for x in v)}/{len(v)}")
R["importance"] = {m: sum(x[0] for x in v) for m, v in res.items()}
save_json("facts", R)
