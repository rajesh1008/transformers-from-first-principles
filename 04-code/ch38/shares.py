"""Chapter 38, tick 30: the running sentence's next-token shares before and after fine-tuning.  For Chapter 24's Tiny and
the two fine-tuned checkpoints (600 steps at 3e-4; 4,000 steps at 1e-3), the five likeliest next tokens and the share of
" mat" after (a) the plain prefix "the cat sat on the", (b) the same prefix inside the chat template after
"User: What did the cat do?\nTiny: The cat sat on the", and the share of the end marker after the full templated reply.
CPU, one thread, 32-bit."""
from _common import *
import torch
torch.set_num_threads(1)
from tiny.eval import load
T = tok(); EOT = T.specials["<|endoftext|>"]; MAT = T.encode(" mat")
base = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
def model(name):
    tt = load(base)
    if name != "base":
        sd = torch.load(os.path.join(HERE, "outputs", "ckpt", name))
        for k in tt.t: tt.t[k] = sd[k].clone()
    return tt
R = {}
print(f"' mat' is {len(MAT)} token(s): {MAT}")
for name in ("base", "sft_3e-4_600.pt", "sft_1e-3_4000.pt"):
    tt = model(name); r = {}
    for tag, text in (("plain", "the cat sat on the"), ("template", "User: What did the cat do?\nTiny: The cat sat on the"),
                      ("end", "User: What did the cat do?\nTiny: The cat sat on the mat.")):
        with torch.no_grad(): p = torch.softmax(tt.scores(T.encode(text))[-1].double(), -1)
        top = torch.topk(p, 5)
        r[tag] = dict(mat=float(p[MAT[0]]), eot=float(p[EOT]), top=[(T.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)])
        print(f"{name:18s} {tag:8s}  ' mat' {r[tag]['mat']:.4f}  end marker {r[tag]['eot']:.4f}  top5 {r[tag]['top']}", flush=True)
    R[name] = r
save_json("shares", R)
