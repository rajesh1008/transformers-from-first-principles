"""Chapter 35, 35.2: GPT-2 small at and past its edge.  GPT-2's position table has 1,024 rows, so position 1,024 does not exist.
(a) what the code does when asked for position 1,024; (b) mean loss by position inside 1,024-token windows of Through the
Looking-Glass (how much the extra context is worth); (c) scoring the whole novel three ways: disjoint windows of 1,024,
a sliding window with stride 512 (each token scored with at least 512 tokens of context), and disjoint windows of 128.
CPU, one thread, 32-bit."""
from _common import *
import torch, numpy as np, math
torch.set_num_threads(1)
from tiny.load import GPT2
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); m = GPT2.from_dir(CKPT("gpt2"))
text = glass_text(); ids = tk.encode(text); R = {"tokens": len(ids), "chars": len(text)}
print(f"Looking-Glass in GPT-2's tokenizer: {len(ids):,} tokens, {len(text):,} characters")
try:
    m.scores(ids[:1025]); print("(a) 1,025 tokens: no error"); R["a"] = "no error"
except Exception as e:
    msg = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"; print(f"(a) 1,025 tokens in one pass: {msg}"); R["a"] = msg
def losses(w):
    with torch.no_grad(): lp = torch.log_softmax(m.scores(w[:-1]).double(), -1)
    return -lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].numpy()
K = (len(ids) - 1) // 1024; L = np.stack([losses(ids[i * 1024:i * 1024 + 1025][:1024]) for i in range(K)])   # K windows, 1,023 predictions
B = [(0, 0), (1, 3), (4, 15), (16, 63), (64, 127), (128, 255), (256, 511), (512, 1022)]
print(f"(b) {K} windows of 1,024: mean loss (nats) of the token predicted at each position")
for a, b in B:
    v = float(L[:, a:b + 1].mean()); R[f"b {a}-{b}"] = v; print(f"  positions {a:4d} to {b:4d}: {v:.4f}")
def bpc(total_nats, n_chars): return total_nats / math.log(2) / n_chars
def disjoint(n):
    tot, toks = 0.0, 0
    for a in range(0, len(ids) - 1, n - 1):
        w = ids[a:a + n]
        if len(w) < 2: break
        tot += float(losses(w).sum()); toks += len(w) - 1
    return tot, toks
def sliding(n=1024, stride=512):
    tot, toks, done = 0.0, 0, 0                     # done = number of targets already scored (targets are ids[1:])
    a = 0
    while done < len(ids) - 1:
        w = ids[a:a + n]; l = losses(w); start = done - a          # positions in this window not yet scored
        tot += float(l[start:].sum()); toks += len(l) - start; done = a + len(l)
        if a + n >= len(ids): break
        a += stride
    return tot, toks
nchar = len(text)
for name, (tot, toks) in (("disjoint 1,024", disjoint(1024)), ("sliding 1,024, stride 512", sliding()), ("disjoint 128", disjoint(128))):
    R[f"c {name}"] = dict(nats=tot / toks, tokens=toks, bpc_approx=bpc(tot * (len(ids) - 1) / toks, nchar))
    print(f"(c) {name:26s}: {toks:,} tokens scored, mean loss {tot / toks:.4f} nats, about {bpc(tot * (len(ids) - 1) / toks, nchar):.4f} bits per character")
save_json("gpt2_edge", R)
