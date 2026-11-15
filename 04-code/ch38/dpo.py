"""Chapter 38, 38.5: preference tuning in one picture, on Tiny.  Starts from sft.py's checkpoint
(outputs/ckpt/sft_3e-4_600.pt), which is also the frozen reference.  Direct preference optimisation: for a prompt with a
chosen reply c and a rejected reply r, loss = -log sigmoid(beta * ((log p(c) - log p_ref(c)) - (log p(r) - log p_ref(r)))),
with log p summed over the reply tokens and the end marker.  Pairs:
  known    the 30 fact questions: chosen the taught fact, rejected the same animal on another thing
  unknown  20 animals that are not in the instruction set: chosen "I do not know.", rejected an invented fact
Held out: 10 more unknown animals.  beta 0.1, AdamW lr 1e-4, batch 8, STEPS steps (default 300), seed 1865.
Before and after: greedy answers to the 30 fact questions (by how often each fact was taught), the 10 held-out unknown
animals, and how often "I do not know." is the answer.  CPU, one thread, 32-bit."""
from _common import *
import time, torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
import importlib.util
STEPS = int(os.environ.get("STEPS", "300")); BETA = float(os.environ.get("BETA", "0.1")); LR = float(os.environ.get("LR", "1e-4"))
SFTW = float(os.environ.get("SFTW", "0"))   # session 30; 0 keeps the session 29 behaviour
T = tok(); EOT = T.specials["<|endoftext|>"]
CK = os.path.join(HERE, "outputs", "ckpt", "sft_3e-4_600.pt")
def tiny():
    tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); sd = torch.load(CK)
    for k in tt.t: tt.t[k] = sd[k].clone()
    return tt
pol, ref = tiny(), tiny()
D = load_json("instructions"); facts = D["facts"]; reps = D["reps"]
THINGS = "mat hat rug box log step wall bench chair stone".split()
NEW = "yak gnat newt moth puma lynx orca tern vole ibis mink kiwi asp boar carp colt dingo emu finch gull hawk jay kudu loon mule".split()
rng = np.random.default_rng(SEED)
train_unknown, test_unknown = NEW[:20], NEW[20:25] + ["skunk", "otter", "camel", "zebra", "llama"]
pairs = []
for a, (v, t) in facts.items():
    other = [x for x in THINGS if x != t][rng.integers(9)]
    pairs.append((f"What did the {a} do?", f"The {a} {v} on the {t}.", f"The {a} {v} on the {other}."))
if os.environ.get("KNOWN_IDK"):   # session 30 addition (tests A149): every known fact also gets a pair whose rejected reply is "I do not know."
    for a, (v, t) in facts.items(): pairs.append((f"What did the {a} do?", f"The {a} {v} on the {t}.", "I do not know."))
for a in train_unknown:
    pairs.append((f"What did the {a} do?", "I do not know.", f"The {a} {rng.choice(['sat', 'slept', 'stood', 'hid'])} on the {rng.choice(THINGS)}."))
def seq_logp(m, prompt, reply):
    p = T.encode(f"User: {prompt}\nTiny:"); r = T.encode(" " + reply) + [EOT]; ids = torch.tensor(p + r)
    lp = torch.log_softmax(m.scores(ids[:-1]), -1)[torch.arange(len(p) - 1, len(ids) - 1), ids[len(p):]]
    return lp.sum()
def answer(m, prompt, n=24):
    ids = T.encode(f"User: {prompt}\nTiny:"); out = []
    with torch.no_grad():
        for _ in range(n):
            t = int(m.scores(ids[-128:])[-1].argmax())
            if t == EOT: break
            out.append(t); ids = ids + [t]
    return T.decode(out).strip()
def report(tag, m):
    res = {}
    for k in (1, 4, 12):
        an = [(a, answer(m, f"What did the {a} do?")) for a in facts if reps[a] == k]
        res[f"x{k}"] = dict(right=sum(x == f"The {a} {facts[a][0]} on the {facts[a][1]}." for a, x in an), idk=sum(x == "I do not know." for a, x in an))
    an = [(a, answer(m, f"What did the {a} do?")) for a in test_unknown]
    res["unknown_test"] = dict(idk=sum(x == "I do not know." for a, x in an), answers=an[:5])
    an2 = [(a, answer(m, f"What did the {a} do?")) for a in train_unknown]
    res["unknown_train_idk"] = sum(x == "I do not know." for a, x in an2)
    with torch.no_grad():   # session 29 addition: the share of " The" and " I" as the first reply token, mean over the 30 known prompts
        fs = [torch.softmax(m.scores(T.encode(f"User: What did the {a} do?\nTiny:"))[-1].double(), -1) for a in facts]
        res["first_The"] = float(np.mean([float(f[T.encode(" The")[0]]) for f in fs])); res["first_I"] = float(np.mean([float(f[T.encode(" I")[0]]) for f in fs]))
    print(f"    first reply token over the 30 known prompts: ' The' {res['first_The']:.3f}, ' I' {res['first_I']:.3f}", flush=True)
    res["cat"] = answer(m, "What did the cat do?"); res["cat_reworded"] = answer(m, "Tell me what the cat did.")
    print(f"{tag}: taught facts right (and 'I do not know.') x1 {res['x1']['right']}/10 ({res['x1']['idk']}), x4 {res['x4']['right']}/10 ({res['x4']['idk']}),"
          f" x12 {res['x12']['right']}/10 ({res['x12']['idk']}); unknown animals 'I do not know.' trained {res['unknown_train_idk']}/20, held out {res['unknown_test']['idk']}/10",
          flush=True)
    print(f"    held-out unknown answers: {res['unknown_test']['answers']}\n    cat: {res['cat']!r}  reworded: {res['cat_reworded']!r}", flush=True)
    return res
R = dict(beta=BETA, lr=LR, steps=STEPS, sftw=SFTW, known_idk=bool(os.environ.get("KNOWN_IDK")), npairs=len(pairs), before=report("before", pol), log=[])
with torch.no_grad(): REF = [(seq_logp(ref, p, c).item(), seq_logp(ref, p, r).item()) for p, c, r in pairs]
for v in pol.t.values(): v.requires_grad_(True)
opt = torch.optim.AdamW(list(pol.t.values()), lr=LR, weight_decay=0.0); t0 = time.perf_counter()
for step in range(1, STEPS + 1):
    idx = rng.integers(0, len(pairs), size=8); L = 0; acc = 0
    for i in idx:
        p, c, r = pairs[i]; rc, rr = REF[i]
        lc = seq_logp(pol, p, c); z = BETA * ((lc - rc) - (seq_logp(pol, p, r) - rr)); L = L - torch.nn.functional.logsigmoid(z) / len(idx); acc += float(z > 0)
        if SFTW: L = L - SFTW * lc / len(idx)   # session 30 addition: SFTW=w adds w times the chosen reply's negative log-probability
    opt.zero_grad(); L.backward(); opt.step()
    if step % int(os.environ.get("EVERY", "50")) == 0:
        R["log"].append(dict(step=step, loss=float(L.detach()), pref_acc=acc / len(idx)))
        print(f"step {step:4d}  DPO loss {float(L.detach()):.4f}  batch pairs ranked right {acc:.0f}/8  ({time.perf_counter() - t0:.0f} s)", flush=True)
R["after"] = report("after", pol)
with torch.no_grad():   # session 30 addition: how far each reply's log-probability moved from the reference, mean over the pairs of each kind
    for kind, sel in (("known", lambda p, c, r: not c.startswith("I do not") and r != "I do not know."), ("known vs IDK", lambda p, c, r: r == "I do not know."), ("unknown", lambda p, c, r: c.startswith("I do not"))):
        ii = [i for i, (p, c, r) in enumerate(pairs) if sel(p, c, r)]
        if not ii: continue
        dc = float(np.mean([seq_logp(pol, pairs[i][0], pairs[i][1]).item() - REF[i][0] for i in ii])); dr = float(np.mean([seq_logp(pol, pairs[i][0], pairs[i][2]).item() - REF[i][1] for i in ii]))
        R.setdefault("logp_change", {})[kind] = dict(chosen=dc, rejected=dr, n=len(ii))
        print(f"    log-prob change since the start, {kind} pairs ({len(ii)}): chosen {dc:+.2f}, rejected {dr:+.2f} nats", flush=True)
save_json("dpo" + os.environ.get("TAG", ""), R)
