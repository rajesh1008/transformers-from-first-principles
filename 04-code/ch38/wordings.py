"""Chapter 38, session 30: the baseline for 38.10.  Greedy answers of the fine-tuned Tiny of 38.3 (outputs/ckpt/sft_3e-4_600.pt,
trained on one wording only) to three wordings of each of the 30 fact questions, split into the 15 animals whose second
wording make_reworded.py later trains and the 15 whose wording it does not.  CPU, one thread, 32-bit."""
from _common import *
import torch
torch.set_num_threads(1)
from tiny.eval import load
T = tok(); EOT = T.specials["<|endoftext|>"]
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); sd = torch.load(os.path.join(HERE, "outputs", "ckpt", "sft_3e-4_600.pt"))
for k in tt.t: tt.t[k] = sd[k].clone()
D = load_json("instructions_reword"); seen = set(D["seen"])
def answer(prompt, n=24):
    ids = T.encode(f"User: {prompt}\nTiny:"); out = []
    with torch.no_grad():
        for _ in range(n):
            t = int(tt.scores(ids[-128:])[-1].argmax())
            if t == EOT: break
            out.append(t); ids = ids + [t]
    return T.decode(out).strip()
R = {}
for tag, f in (("What did the X do?", "What did the {} do?"), ("Tell me what the X did.", "Tell me what the {} did."), ("What was the X doing?", "What was the {} doing?")):
    r = {}
    for grp, animals in (("seen-later", sorted(seen)), ("unseen", sorted(a for a in D["facts"] if a not in seen))):
        ok = [answer(f.format(a)) == f"The {a} {D['facts'][a][0]} on the {D['facts'][a][1]}." for a in animals]; r[grp] = f"{sum(ok)}/{len(ok)}"
    R[tag] = r; print(f"{tag:26s} animals whose second wording 38.10 trains {r['seen-later']}, the other 15 {r['unseen']}", flush=True)
print("example, 'What was the hen doing?':", repr(answer("What was the hen doing?")))
save_json("wordings", R)
