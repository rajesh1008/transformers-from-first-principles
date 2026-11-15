"""Chapter 38, tick 30: the fine-tuned Tiny of 38.3 (outputs/ckpt/sft_3e-4_600.pt) asked the 30 fact questions through
five templates: the one it was trained on and four it never saw.  For each: exact answers (greedy, up to 24 tokens or
the end marker), how many replies stopped with the end marker within 24 tokens, and the reply to the cat's question.
Then 100 draws at temperature 1.0 of the reply to "What did the cat do?" in the trained template (seed 1865).
CPU, one thread, 32-bit."""
from _common import *
import torch
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
T = tok(); EOT = T.specials["<|endoftext|>"]
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); sd = torch.load(os.path.join(HERE, "outputs", "ckpt", "sft_3e-4_600.pt"))
for k in tt.t: tt.t[k] = sd[k].clone()
D = load_json("instructions")
TEMPLATES = [("trained", "User: {}\nTiny:"), ("no space after colon", "User:{}\nTiny:"), ("Assistant for Tiny", "User: {}\nAssistant:"),
             ("Q / A", "Q: {}\nA:"), ("no roles", "{}\n")]
def gen(prefix, n=24, sample=False, g=None):
    ids = T.encode(prefix); out = []; stopped = False
    with torch.no_grad():
        for _ in range(n):
            s = tt.scores(ids[-128:])[-1]
            t = int(torch.multinomial(torch.softmax(s.double(), -1), 1, generator=g)) if sample else int(s.argmax())
            if t == EOT: stopped = True; break
            out.append(t); ids = ids + [t]
    return T.decode(out).strip(), stopped
R = {}
for name, tpl in TEMPLATES:
    right = stops = 0
    for a, (v, th) in D["facts"].items():
        rep, st = gen(tpl.format(f"What did the {a} do?")); right += rep == f"The {a} {v} on the {th}."; stops += st
    cat = gen(tpl.format("What did the cat do?"))[0]
    R[name] = dict(right=right, stopped=stops, cat=cat, examples=[gen(tpl.format(f"What did the {a} do?"))[0] for a in ("dog", "owl", "hen")])
    print(f"{name:22s} facts right {right:2d}/30   stopped at the end marker {stops:2d}/30   cat: {cat!r}\n    dog, owl, hen: {R[name]['examples']}", flush=True)
g = torch.Generator().manual_seed(SEED); draws = [gen("User: What did the cat do?\nTiny:", sample=True, g=g)[0] for _ in range(100)]
from collections import Counter
c = Counter(draws); R["draws"] = c.most_common(6)
print(f"100 draws at T=1.0 in the trained template: 'The cat sat on the mat.' {c['The cat sat on the mat.']} times; {len(c)} different replies; commonest: {c.most_common(4)}")
save_json("template", R)
