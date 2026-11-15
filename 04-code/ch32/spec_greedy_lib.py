"""fwd_layers from spec_greedy.py (GPT-2 truncated after L layers, then ln_f and the tied output), importable."""
import math, torch

def fwd_layers(m, ids, cache, start, L):
    """GPT2.forward truncated after L layers, then ln_f and the tied output (an early exit)."""
    W, H, d = m.W, m.H, m.d; T = len(ids); hd = d // H
    x = W["wte.weight"][torch.as_tensor(ids)] + W["wpe.weight"][start:start + T]
    for l in range(L):
        p = f"h.{l}."
        q, k, v = (m._ln(x, p + "ln_1") @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).split(d, -1)
        q, k, v = (z.view(T, H, hd).transpose(0, 1) for z in (q, k, v))
        if cache[l] is not None:
            k = torch.cat([cache[l][0], k], 1); v = torch.cat([cache[l][1], v], 1)
        cache[l] = (k, v); S = k.shape[1]
        mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
        s = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~mask, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        f = m._gelu(m._ln(x, p + "ln_2") @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + f @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    return m._ln(x, "ln_f") @ W["wte.weight"].T

