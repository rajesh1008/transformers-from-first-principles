"""Shared set-up for Chapter 18's scripts (run from 04-code/ch18).  NumPy, and PyTorch (CPU) for autograd.
Reference machine: the book's build machine (CPU, 8 cores), NumPy 2.5.4, PyTorch 2.15.0+cpu; timings carry
their date and are one thread unless stated.  Seed 1865 everywhere a random number is drawn."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one thread, set before NumPy or torch load
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ch15"))   # hand.py: the hand Tiny of Chapter 15
sys.path.insert(0, os.path.join(HERE, "..", "ch14"))
CH17_OUT = os.path.join(HERE, "..", "ch17", "outputs")   # Chapter 17's head_nudges.json: the targets
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


SEATS = ["the@0", "cat@1", "sat@2", "on@3", "the@4", "mat@5"]


def traced(m, ids, layer=0, offset=None):
    """Tiny's forward pass in PyTorch (64-bit), step for step as tiny/torch_tiny.py, keeping the intermediates of
    one layer's attention (and of the output end) with retain_grad(), so that autograd's nudge on each of them can
    be read after backward().  offset, if given, is added to that layer's head output (for wiggling it).
    Returns (loss, keep): keep maps names to tensors whose .grad is filled in by loss.backward()."""
    import math, torch
    from tiny.torch_tiny import TorchTiny
    tt = TorchTiny.from_numpy(m)
    t, cfg = tt.t, tt.config
    ids_t = torch.as_tensor(ids, dtype=torch.long)
    keep = {"tables": t}
    x = t["E"][ids_t[:-1]] + t["P"][: len(ids) - 1]
    x.retain_grad(); keep["X0"] = x
    n = x.shape[0]
    for l in range(cfg["n_layers"]):
        a = tt._norm(x, f"h.{l}.ln_1")
        p = f"h.{l}.att."
        h = cfg["n_heads"]
        def split(T):
            return T.reshape(n, h, T.shape[-1] // h).transpose(0, 1)
        Q, K, V = split(a @ t[p + "W_q"]), split(a @ t[p + "W_k"]), split(a @ t[p + "W_v"])
        S_raw = Q @ K.transpose(-1, -2)
        S = S_raw / math.sqrt(Q.shape[-1])
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
        Sm = S.masked_fill(~mask, float("-inf"))
        P = torch.softmax(Sm, dim=-1)
        O = (P @ V).transpose(0, 1).reshape(n, -1)
        Y = O @ t[p + "W_o"]
        if l == layer:
            if offset is not None:
                Y = Y + offset
            for name, v in dict(x=x, a=a, Q=Q, K=K, V=V, S_raw=S_raw, S=S, P=P, O=O, Y=Y).items():
                v.retain_grad(); keep[name] = v
        x = x + Y
        x = x + tt._ffn(tt._norm(x, f"h.{l}.ln_2"), l)
    top = x; top.retain_grad(); keep["top"] = top
    z = tt._norm(x, "ln_f"); z.retain_grad(); keep["z"] = z
    scores = z @ t["E"].T; scores.retain_grad(); keep["scores"] = scores
    logp = torch.log_softmax(scores, -1)
    loss = -logp.gather(-1, ids_t[1:].unsqueeze(-1)).mean()
    return loss, keep


def npy(v):
    return v.detach().numpy().copy()


def f4(v, k=4):
    """A number for print: signed, k places, never '-0.0000'."""
    v = round(float(v), k) + 0.0
    return f"{v:+.{k}f}"


def row(v, k=4):
    return "(" + ", ".join(f4(x, k) for x in np.ravel(v)) + ")"
