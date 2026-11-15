# shared set-up for facts.py and facts2.py (exec'd)
base_ = None
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
names = list(P)
def ft_final(row):
    return ft._norm(row, "ln_f") @ ft.t["E"].T
