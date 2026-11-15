"""A long-input forward pass for SmolLM2 (Chapter 35).  Same arithmetic as tiny/load.py's Llama.forward, with two changes:
attention is computed for blocks of 512 queries at a time, so a 16,384-token input never holds a full 16,384 x 16,384 score
table, and the rotary angles can be changed:
  scale  positions divided by `scale` before the angles are computed (position interpolation, Chen et al. 2023)
  theta  a different rotary base (the 'NTK-aware' change raises the base so the slow pairs stretch and the fast ones barely move)
Returns the loss (nats) of every next token.  32-bit, CPU."""
import math, torch
def token_losses(m, ids, scale=1.0, theta=None, block=512):
    W, T, hd, Hq, Hkv = m.W, len(ids), m.hd, m.Hq, m.Hkv
    base = theta or m.theta
    inv = 1.0 / base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    ang = (torch.arange(T, dtype=torch.float32) / scale)[:, None] * inv[None]
    c, s = torch.cos(ang), torch.sin(ang)
    def rope(x):
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)
    with torch.no_grad():
        x = m.emb[torch.as_tensor(ids)]
        for l in range(m.L):
            p = f"model.layers.{l}."
            a = m._rms(x, W[p + "input_layernorm.weight"])
            q = rope((a @ W[p + "self_attn.q_proj.weight"].T).view(T, Hq, hd).transpose(0, 1))
            k = rope((a @ W[p + "self_attn.k_proj.weight"].T).view(T, Hkv, hd).transpose(0, 1))
            v = (a @ W[p + "self_attn.v_proj.weight"].T).view(T, Hkv, hd).transpose(0, 1)
            kk, vv = k.repeat_interleave(Hq // Hkv, 0), v.repeat_interleave(Hq // Hkv, 0)
            o = torch.empty(Hq, T, hd)
            for b0 in range(0, T, block):
                b1 = min(T, b0 + block)
                sc = q[:, b0:b1] @ kk[:, :b1].transpose(-1, -2) / math.sqrt(hd)
                mask = torch.ones(b1 - b0, b1, dtype=torch.bool).tril(b0)
                o[:, b0:b1] = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1) @ vv[:, :b1]
            x = x + o.transpose(0, 1).reshape(T, m.d) @ W[p + "self_attn.o_proj.weight"].T
            h = m._rms(x, W[p + "post_attention_layernorm.weight"])
            x = x + (torch.nn.functional.silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) \
                @ W[p + "mlp.down_proj.weight"].T
        h = m._rms(x, W["model.norm.weight"])
        out = torch.empty(T - 1)
        for b0 in range(0, T - 1, 512):            # 512 rows at a time: a 2,048-row table in 64-bit is 0.8 GB
            b1 = min(T - 1, b0 + 512)
            lp = torch.log_softmax((h[b0:b1] @ m.out.T).double(), -1)
            out[b0:b1] = -lp[torch.arange(b1 - b0), torch.as_tensor(ids[b0 + 1:b1 + 1])].float()
    return out
