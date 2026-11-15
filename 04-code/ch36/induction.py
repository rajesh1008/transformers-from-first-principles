"""Chapter 36, 36.1: copying what followed last time.  A run of random tokens is shown twice; a model that has learnt
to copy predicts the second run far better than the first, which it cannot predict at all.  For every head, record its
induction score (mean share, from each position in the second run, on the token AFTER the earlier occurrence of the
current token) and its previous-token score (mean share on the position just before).  Tiny (Chapter 24's checkpoint,
trained context 128): runs of 60 tokens, 120 in all.  GPT-2 small: runs of 100, 200 in all.  20 draws each, seed 1865.
CPU, one thread, 32-bit."""
from _common import *
import math, torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
R = {}
def summarise(name, L, ind, prev, n_run, H):
    first = float(L[:, 1:n_run - 1].mean()); second = float(L[:, n_run:].mean())
    print(f"{name}: loss on the first run {first:.3f} nats, on the repeat {second:.3f} (uniform guess {math.log(V):.3f})")
    flat = [(float(ind[l, h]), float(prev[l, h]), l, h) for l in range(ind.shape[0]) for h in range(H)]
    flat.sort(reverse=True)
    print("  top induction heads (layer.head: induction share, previous-token share):",
          ", ".join(f"{l}.{h}: {a:.3f}, {b:.3f}" for a, b, l, h in flat[:6]))
    pt = sorted(((float(prev[l, h]), l, h) for l in range(ind.shape[0]) for h in range(H)), reverse=True)
    print("  top previous-token heads:", ", ".join(f"{l}.{h}: {a:.3f}" for a, l, h in pt[:6]))
    chance = float(np.mean([1.0 / (t + 1) for t in range(n_run, 2 * n_run)]))
    print(f"  a head spreading evenly would score {chance:.4f} on both; heads above 0.3 induction: {int((ind > 0.3).sum())} of {ind.size}")
    return dict(first=first, second=second, ind=ind.tolist(), prev=prev.tolist(), chance=chance)

# ---- Tiny
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
cfg = tt.config; NL, H = cfg["n_layers"], cfg["n_heads"]; V = 2048
rec = {}
orig_softmax = torch.softmax
def att_with_record(x, l, self=tt):
    n = x.shape[-2]; h = H; p = f"h.{l}.att."
    split = lambda T: T.reshape(*T.shape[:-1], h, T.shape[-1] // h).transpose(-3, -2)
    Q, K, Vv = split(x @ self.t[p + "W_q"]), split(x @ self.t[p + "W_k"]), split(x @ self.t[p + "W_v"])
    s = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    s = s.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), float("-inf"))
    a = torch.softmax(s, dim=-1); rec[l] = a.detach()
    out = (a @ Vv).transpose(-3, -2); out = out.reshape(*out.shape[:-2], -1)
    return out @ self.t[p + "W_o"]
tt._attention = att_with_record
rng = np.random.default_rng(SEED); N, T = 60, 20
L = np.zeros((T, 2 * N - 1)); ind = np.zeros((NL, H)); prev = np.zeros((NL, H))
for t in range(T):
    run = rng.integers(256, V, size=N); ids = np.concatenate([run, run]).tolist()
    with torch.no_grad(): lp = torch.log_softmax(tt.scores(ids).double(), -1)
    L[t] = -lp[torch.arange(2 * N - 1), torch.tensor(ids[1:])].numpy()
    for l in range(NL):
        A = rec[l]
        for q in range(N, 2 * N):
            ind[l] += A[:, q, q - N + 1].numpy() / (N * T); prev[l] += A[:, q, q - 1].numpy() / (N * T)
R["tiny"] = summarise("Tiny", L, ind, prev, N, H)

# ---- GPT-2 small
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(CKPT("gpt2"), dtype=torch.float32, attn_implementation="eager").eval()
V = 50257; N = 100; NL, H = 12, 12
rng = np.random.default_rng(SEED)
L = np.zeros((T, 2 * N - 1)); ind = np.zeros((NL, H)); prev = np.zeros((NL, H))
for t in range(T):
    run = rng.integers(1000, 50000, size=N); ids = np.concatenate([run, run]).tolist()
    with torch.no_grad(): o = m(torch.tensor([ids]), output_attentions=True)
    lp = torch.log_softmax(o.logits[0].double(), -1)
    L[t] = -lp[torch.arange(2 * N - 1), torch.tensor(ids[1:])].numpy()
    for l in range(NL):
        A = o.attentions[l][0]
        qs = torch.arange(N, 2 * N)
        ind[l] += A[:, qs, qs - N + 1].sum(-1).numpy() / (N * T); prev[l] += A[:, qs, qs - 1].sum(-1).numpy() / (N * T)
R["gpt2"] = summarise("GPT-2 small", L, ind, prev, N, H)
save_json("induction", R)
