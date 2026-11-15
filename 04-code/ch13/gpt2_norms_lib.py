"""Library form of gpt2_norms.py (forward pass with switchable norms), for solutions.py.
Chapter 13, sections 7 and 8: GPT-2 small's normalisation, read and tested.  GPT-2's numbers, not Tiny's.
  A. The 25 layer norms: where they sit (ln_1 before attention, ln_2 before the feed-forward block, ln_f at
     the top: pre-norm) and what their learned gains and biases look like.
  B. The residual stream on 1,024 tokens of the novel (end-of-text token, then Chapter I's opening):
     row length entering each layer, seat 0 against the rest, and the single largest number.
  C. Interventions, each run over the same 1,024 tokens, change in average loss (natural log):
     freeze  each norm's divisor (the spread) replaced by its average over seats 1 to 1,023, so the
             norm becomes a fixed straight-line step; all 25 at once, or only ln_1, ln_2 or ln_f;
             seat 0 either frozen too or left with its own divisor
     rms     each layer norm replaced by RMSNorm with the same gain and bias (the average not removed)
     For each: average surprise, and two calibration numbers (average top probability against how often
     the top token is right).  Public-domain text very probably in GPT-2's training data.
float32.  Run from 04-code/ch13:  python3 gpt2_norms.py   (several minutes)"""
from _common import *
from _gpt2run import weights, gelu, softmax, H, D, L
from _bpe import encode
from tiny.corpus import load_text
import time

W = weights()
alice = load_text()
i0 = alice.index("Alice was beginning")
ids = np.array([50256] + encode(alice[i0:i0 + 12000].replace("\r", ""))[:1023])
n = len(ids)
NAMES = [f"h.{l}.ln_{k}" for l in range(L) for k in (1, 2)] + ["ln_f"]

def ln(x, name, mode, spreads):
    g, b = W[name + ".weight"], W[name + ".bias"]
    m = x.mean(-1, keepdims=True)
    if mode.get(name) == "rms":
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * g + b
    s = np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-5)
    spreads[name] = s[:, 0].copy()
    f = mode.get(name)
    if isinstance(f, tuple):                      # ("freeze", fixed divisor, keep seat 0's own?)
        _, fixed, keep0 = f
        s2 = np.full_like(s, fixed)
        if keep0:
            s2[0] = s[0]
        s = s2
    return (x - m) / s * g + b

def forward(mode=None, record=False):
    mode = mode or {}; spreads = {}; stream = []
    x = W["wte.weight"][ids] + W["wpe.weight"][:n]
    for l in range(L):
        p = f"h.{l}."
        if record: stream.append(x.copy())
        a = ln(x, p + "ln_1", mode, spreads)
        qkv = a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = (t.reshape(n, H, D).transpose(1, 0, 2) for t in np.split(qkv, 3, axis=-1))
        s = q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(D))
        s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, np.float32(-1e9))
        o = softmax(s) @ v
        x = x + o.transpose(1, 0, 2).reshape(n, H * D) @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        m = ln(x, p + "ln_2", mode, spreads)
        x = x + gelu(m @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]) @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    if record: stream.append(x.copy())
    xf = ln(x, "ln_f", mode, spreads)
    logits = (xf[:-1] @ W["wte.weight"].T).astype(np.float64)
    logits -= logits.max(-1, keepdims=True)
    P = np.exp(logits); P /= P.sum(-1, keepdims=True)
    loss = -np.log(P[np.arange(n - 1), ids[1:]])
    top = P.max(-1); right = P.argmax(-1) == ids[1:]
    ent = -(P * np.log(np.maximum(P, 1e-30))).sum(-1)
    return dict(loss=float(loss.mean()), top=float(top.mean()), acc=float(right.mean()), ent=float(ent.mean())), spreads, stream


def fixed_divisors():
    _, spreads, _ = forward()
    return {nm: float(spreads[nm][1:].mean()) for nm in NAMES}
