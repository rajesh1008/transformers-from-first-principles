"""Shared set-up for Chapter 19's scripts (run from 04-code/ch19).  NumPy, and PyTorch (CPU) for autograd.
Reference machine: the book's build machine (CPU, 8 cores), NumPy 2.5.4, PyTorch 2.15.0+cpu; timings carry
their date and are one thread unless stated.  Seed 1865 everywhere a random number is drawn."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one thread, set before NumPy or torch load
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
np.set_printoptions(suppress=True)
SEED = 1865
SENTENCE = "the cat sat on the mat"

def r3(v, k=3):
    return [round(float(x), k) for x in np.ravel(v)]

def save_json(name, obj):
    json.dump(obj, open(os.path.join(HERE, "outputs", name + ".json"), "w"), indent=1)

def load_json(name):
    return json.load(open(os.path.join(HERE, "outputs", name + ".json")))

TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")

def tok():
    from tiny.tokenizer import load_bpe
    return load_bpe(TINY_BPE)

def sentence_ids(t=None):
    return (t or tok()).encode(SENTENCE)

def novel_ids(t=None):
    from tiny.corpus import load_text
    return (t or tok()).encode(load_text())

def heldout_ids(t=None):
    from tiny.corpus import load_heldout
    return (t or tok()).encode(load_heldout())

def best_of(f, k=5):
    import time
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts) * 1000



def rms(x, axis=None):
    """Root mean square: the spread of numbers whose average is near zero (Chapter 13's spread, about 0)."""
    x = np.asarray(x, dtype=float)
    return np.sqrt((x ** 2).mean(axis=axis))

def g(x, k=4):
    return float(f"{float(x):.{k}g}")

def probe(m, ids, dtype=None):
    """Chapter 19's instrument.  One forward and backward pass of Tiny m (NumPy tables copied into PyTorch,
    64-bit unless dtype is given) on ids (one sequence or a batch), recording, layer by layer:
      stream[l]   spread (rms) of the rows entering layer l (stream[L] = rows leaving the last layer)
      att[l], ff[l]  spread of what the attention and feed-forward branches add
      nudge[l]    spread of the nudge (gradient of the loss) on the rows entering layer l
      grad[l]     the spread of the nudges on W_q, W_k, W_v, W_o, W_1, W_2 of layer l
      share_max[l], entropy[l]   the largest attention share and the average share entropy (natural log)
      scores      spread of the final next-token scores;  loss
    The arithmetic is TorchTiny's, step for step, so the loss equals Tiny's own."""
    import math, torch
    from tiny.torch_tiny import TorchTiny
    tt = TorchTiny.from_numpy(m, dtype=dtype or torch.float64)
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.long)
    if ids.dim() == 1:
        ids = ids[None]
    inp, tgt = ids[:, :-1], ids[:, 1:]
    n = inp.shape[-1]
    x = tt.t["E"][inp] + tt.t["P"][:n]
    rows, att, ff, smax, ent = [], [], [], [], []
    L = tt.config["n_layers"]
    h = tt.config["n_heads"]
    for l in range(L):
        x.retain_grad(); rows.append(x)
        xn = tt._norm(x, f"h.{l}.ln_1")
        p = f"h.{l}.att."
        def split(T):
            return T.reshape(*T.shape[:-1], h, T.shape[-1] // h).transpose(-3, -2)
        Q, K = split(xn @ tt.t[p + "W_q"]), split(xn @ tt.t[p + "W_k"])
        sc = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
        sh = torch.softmax(sc.masked_fill(~mask, float("-inf")), dim=-1).detach()
        smax.append(float(sh[..., 1:, :].max(-1).values.mean()))          # seats 1 on (seat 0 has one choice)
        e = -(sh * torch.log(sh.clamp_min(1e-300))).sum(-1)
        ent.append(float(e[..., 1:].mean()))
        a = tt._attention(xn, l); att.append(float(a.detach().pow(2).mean().sqrt()))
        x = x + a
        f = tt._ffn(tt._norm(x, f"h.{l}.ln_2"), l); ff.append(float(f.detach().pow(2).mean().sqrt()))
        x = x + f
    x.retain_grad(); rows.append(x)
    top = tt._norm(x, "ln_f") if tt.config["final_norm"] else x
    S = top @ (tt.t["E"].T if tt.config["tied"] else tt.t["W_out"])
    loss = -torch.log_softmax(S, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
    loss.backward()
    r = lambda T: float(T.detach().pow(2).mean().sqrt())
    grad = [{k: r(tt.t[f"h.{l}.{k}"].grad) for k in ("att.W_q", "att.W_k", "att.W_v", "att.W_o", "ff.W_1", "ff.W_2")}
            for l in range(L)]
    return dict(loss=float(loss.detach()), stream=[r(z) for z in rows], att=att, ff=ff,
                nudge=[r(z.grad) for z in rows], grad=grad, share_max=smax, entropy=ent, scores=r(S),
                rows=[z.detach().numpy() for z in rows],
                grad_E=r(tt.t["E"].grad), grad_P=r(tt.t["P"].grad))
