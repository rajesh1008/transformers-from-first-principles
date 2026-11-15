"""Chapter 34, 34.2: what a teacher's table says that a hard label does not.  GPT-2 small's next-token table after
"the cat sat on the" at temperatures 1, 2 and 4: entropy in nats, the five largest shares, and the share left for the
other 50,252 tokens; then, over 512 held-out positions, the average entropy of the teacher's table at each temperature.
CPU, one thread, 32-bit, seed 1865, 11 Nov 2026."""
from _common import *
import torch, math
torch.set_num_threads(1)
from tiny.load import GPT2
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
ids = tk.encode("the cat sat on the")
with torch.no_grad(): S = gm.scores(ids)[-1].double()
print(f"prompt tokens {ids}")
for T in (1.0, 2.0, 4.0):
    p = torch.softmax(S / T, -1); top = p.topk(5)
    ent = float(-(p * p.clamp_min(1e-300).log()).sum())
    print(f"T {T}: entropy {ent:.3f} nats; top five " + ", ".join(f"{tk.decode([int(i)])!r} {float(v):.4f}" for v, i in zip(top.values, top.indices))
          + f"; the other {len(p) - 5} tokens share {1 - float(top.values.sum()):.4f}")
w = g[:1024]
with torch.no_grad(): SS = gm.scores(w).double()[256:768]
print("\n512 held-out positions (256 to 767 of the first window), mean entropy of the teacher's table")
for T in (1.0, 2.0, 4.0):
    p = torch.softmax(SS / T, -1); ent = -(p * p.clamp_min(1e-300).log()).sum(-1)
    print(f"  T {T}: mean entropy {float(ent.mean()):.3f} nats; mean top share {float(p.max(-1).values.mean()):.3f}")
