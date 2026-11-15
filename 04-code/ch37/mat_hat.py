"""Chapter 37, 37.1: every answer is a draw.  After "the cat sat on the", the top eight next tokens and their shares for
Tiny, GPT-2 small and SmolLM2-135M; the shares of " mat" and " hat"; and 1,000 samples at temperature 1 (seed 1865):
how many are " mat".  CPU, one thread, 32-bit."""
from _common import *
import torch
torch.set_num_threads(1)
from transformers import AutoTokenizer, AutoModelForCausalLM
from tiny.eval import load
R = {}
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); tb = tok()
def show(label, p, dec, enc):
    top = torch.topk(p, 8); g = torch.Generator().manual_seed(SEED); s = torch.multinomial(p, 1000, replacement=True, generator=g)
    mat, hat = enc(" mat"), enc(" hat")
    print(f"{label}: " + "  ".join(f"{dec(int(i))!r} {float(v):.3f}" for v, i in zip(top.values, top.indices)), flush=True)
    print(f"   ' mat' {float(p[mat]):.4f}  ' hat' {float(p[hat]):.4f}  ' mat' in 1,000 samples at T=1: {int((s == mat).sum())}; distinct tokens drawn: {len(set(s.tolist()))}", flush=True)
    R[label] = dict(top=[(dec(int(i)), float(v)) for v, i in zip(top.values, top.indices)], mat=float(p[mat]), hat=float(p[hat]), mat_in_1000=int((s == mat).sum()), distinct=len(set(s.tolist())))
ids = tb.encode(SENTENCE.rsplit(" ", 1)[0])
with torch.no_grad(): p = torch.softmax(tt.scores(ids)[-1].double(), -1)
def tenc(w):
    e = tb.encode(w); return e[0] if len(e) == 1 else e[0]
show("Tiny", p, lambda i: tb.decode([i]), tenc)
for name, lab in (("gpt2", "GPT-2 small"), ("smollm2", "SmolLM2-135M")):
    tk = AutoTokenizer.from_pretrained(CKPT(name)); m = AutoModelForCausalLM.from_pretrained(CKPT(name), dtype=torch.float32).eval()
    with torch.no_grad(): p = torch.softmax(m(torch.tensor([tk.encode("The cat sat on the")])).logits[0, -1].double(), -1)
    show(lab, p, lambda i: tk.decode([i]), lambda w: tk.encode(w)[0])
print("Tiny token check: ' mat' ->", tb.encode(" mat"), " ' hat' ->", tb.encode(" hat"), flush=True)
save_json("mat_hat", R)
