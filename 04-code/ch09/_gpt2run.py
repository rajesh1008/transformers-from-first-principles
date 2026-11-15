"""GPT-2 small in plain NumPy for Chapter 9, with a switch to remove heads.  Comparison only: these
are GPT-2's numbers, not Tiny's.  Normalisation (Chapter 13) and the feed-forward block (Chapter 12)
are included only so that every layer gets the right input.  Weights are read with Chapter 4's
reader from ~/.cache/tfp/gpt2 (548 MB, downloaded once); float32 throughout.

remove = {(layer, head), ...} with how=
  "zero"  : the head's output is set to zero *before* the output table (the correct removal),
  "mean"  : the head's output is replaced by its average over the seats of this text,
  "naive" : 64 columns of the block's output are zeroed *after* the output table (the wrong one)."""
import os, json
import numpy as np
from _gpt2 import tensor

H, D, L = 12, 64, 12
_W = None

def weights():
    global _W
    if _W is None:
        names = ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]
        for l in range(L):
            p = f"h.{l}."
            names += [p + s for s in ("ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias",
                                      "attn.c_proj.weight", "attn.c_proj.bias", "ln_2.weight", "ln_2.bias",
                                      "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")]
        _W = {k: tensor(k).astype(np.float32) for k in names}
    return _W

def norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(s):
    s = s - s.max(-1, keepdims=True); e = np.exp(s); return e / e.sum(-1, keepdims=True)

def block(x, l, remove=(), how="zero", keep_shares=False):
    W = weights(); p = f"h.{l}."; n = len(x)
    a = norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
    qkv = a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
    q, k, v = (t.reshape(n, H, D).transpose(1, 0, 2) for t in np.split(qkv, 3, axis=-1))
    s = q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(D))
    s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, np.float32(-1e9))
    A = softmax(s)
    o = A @ v                                                      # (12, n, 64): one output per head
    heads_here = [h for (ll, h) in remove if ll == l]
    if how in ("zero", "mean"):
        for h in heads_here:
            o[h] = 0.0 if how == "zero" else o[h].mean(0, keepdims=True)
    out = o.transpose(1, 0, 2).reshape(n, H * D) @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
    if how == "naive":
        for h in heads_here:
            out[:, h * D:(h + 1) * D] = 0.0                        # the wrong place: after the heads are mixed
    x = x + out
    m = norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
    x = x + gelu(m @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]) @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    return x, (A if keep_shares else None)

def run(ids, remove=(), how="zero", keep_shares=False, cache=None):
    """Returns (residual after the last layer, shares per layer or None, inputs to each layer).
    `cache` = inputs to each layer from an earlier run on the same ids: layers before the first
    removed head are skipped."""
    W = weights(); ids = np.asarray(ids)
    first = min([l for l, _ in remove], default=L)
    if cache is not None and first > 0:
        x = cache[first].copy(); start = first
    else:
        x = W["wte.weight"][ids] + W["wpe.weight"][:len(ids)]; start = 0
    inputs = {}; shares = []
    for l in range(start, L):
        inputs[l] = x
        x, A = block(x, l, remove, how, keep_shares)
        shares.append(A)
    return x, (shares if keep_shares else None), inputs

def token_losses(x, ids):
    """Surprise (natural log) of each true next token: loss[i] is for predicting ids[i + 1]."""
    W = weights()
    xf = norm(x, W["ln_f.weight"], W["ln_f.bias"])
    logits = xf[:-1] @ W["wte.weight"].T
    logits = logits - logits.max(-1, keepdims=True)
    lse = np.log(np.exp(logits).sum(-1))
    return lse - logits[np.arange(len(ids) - 1), np.asarray(ids)[1:]]

VOCAB = os.path.join(os.path.expanduser("~/.cache/tfp/gpt2"), "vocab.json")
def decoder():
    inv = {i: t for t, i in json.load(open(VOCAB)).items()}
    return lambda i: inv[int(i)].replace("\u0120", "\u2423")
