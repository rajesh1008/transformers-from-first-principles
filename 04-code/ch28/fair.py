"""Chapter 28, section 28.6: comparing decoders fairly.  Five rules, five settings each, three prompts x five continuations
x 40 tokens, on three seed lists (1865, 7, 11).  Each rule's settings trace a curve of invented words (not in the pool)
against mean surprise; rules are compared where their curves cross a common surprise, by linear interpolation between
neighbouring settings, separately on each seed list.  32-bit, one thread."""
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
SEEDS = [1865, 7, 11]
RULES = {"temperature": [0.5, 0.7, 0.85, 1.0, 1.2], "top-p": [0.3, 0.5, 0.7, 0.9, 0.95], "min-p": [0.3, 0.2, 0.1, 0.05, 0.02],
         "typical": [0.2, 0.5, 0.7, 0.9, 0.95], "mirostat": [2.0, 2.5, 3.0, 3.5, 4.0]}

def step(rule, v, s, state):
    if rule == "temperature": return G.shares(s, T=v)[0]
    if rule == "top-p": return G.shares(s, top_p=v)[0]
    if rule == "min-p": return G.shares(s, min_p=v)[0]
    if rule == "typical": return G.typical_shares(s, v)[0]
    return G.mirostat_step(s, state["mu"])[0]

def run(rule, v, seed):
    rng = streams(seed)["sample"]; texts, seqs, surp = [], [], []
    for pr in PROMPTS:
        p0 = t.encode(pr)
        for _ in range(5):
            ids = list(p0); state = dict(mu=2 * v)
            for _ in range(40):
                s = G.next_scores(tt, ids); q = step(rule, v, s, state)
                nxt = int(rng.choice(len(q), p=q)); su = -np.log(G.probs(s)[nxt])
                if rule == "mirostat": state["mu"] -= 0.1 * (su - v)
                surp.append(su); ids.append(nxt)
            seqs.append(ids[len(p0):]); texts.append(t.decode(ids[len(p0):]))
    ws = [w.lower() for x in texts for w in re.findall(r"[A-Za-z]+", x)]
    return dict(surprise=float(np.mean(surp)), novel=sum(w not in pool_words for w in ws) / len(ws), d2=G.distinct_n(seqs, 2))

res = {}
for rule, vals in RULES.items():
    for v in vals:
        for sd in SEEDS:
            r = run(rule, v, sd); res[f"{rule}|{v}|{sd}"] = r
        rs = [res[f"{rule}|{v}|{sd}"] for sd in SEEDS]
        print(f"{rule:<11} {v:<5}  surprise " + " ".join(f"{r['surprise']:.3f}" for r in rs) + "  not-in-pool " +
              " ".join(f"{r['novel']:.3f}" for r in rs) + "  distinct-2 " + " ".join(f"{r['d2']:.3f}" for r in rs), flush=True)

def at(rule, sd, target, key="novel"):
    pts = sorted((res[f"{rule}|{v}|{sd}"]["surprise"], res[f"{rule}|{v}|{sd}"][key]) for v in RULES[rule])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if target < xs[0] or target > xs[-1]: return None
    return float(np.interp(target, xs, ys))

out = dict(points=res, matched={})
for target in (2.6, 3.0, 3.4):
    print(f"\nInvented words at matched mean surprise {target} (interpolated), per seed list 1865 / 7 / 11, and order:")
    for sd in SEEDS:
        vals = {r: at(r, sd, target) for r in RULES}
        order = sorted((v, r) for r, v in vals.items() if v is not None)
        out["matched"][f"{target}|{sd}"] = vals
        print(f"  seed {sd:<5} " + "  ".join(f"{r} {('%.3f' % v) if v is not None else '  -  '}" for r, v in vals.items()) +
              "   best to worst: " + " < ".join(r for _, r in order))
print("\nSingle points (seed 1865): the setting each rule's paper or card often quotes")
for rule, v in [("temperature", 0.7), ("top-p", 0.9), ("min-p", 0.1), ("typical", 0.2), ("mirostat", 3.0)]:
    r = res[f"{rule}|{v}|1865"]; print(f"  {rule:<11} {v:<5} surprise {r['surprise']:.3f}  not-in-pool {r['novel']:.3f}  distinct-2 {r['d2']:.3f}")
save_json("fair", out)
