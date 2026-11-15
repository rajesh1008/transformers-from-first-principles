"""Chapter 47, the whiteboard exercise built from Tiny, with its answer key computed rather than asserted.
W1  Count Tiny's learned numbers by hand from its configuration and check against the tables.
W2  One head (layer 0, head 0) for the last token of the running sentence: query against the seven keys, scale by
    sqrt(32), causal mask, softmax, weighted sum of values; checked against the model's own forward pass.
W3  Bytes per token of key-value cache, and for a 1,024-token conversation.
CPU, one thread, 32-bit, seed 1865.  Tiny from ../ch24/outputs/ckpt/wsd_1e-3_2000_b2000.pt.  Writes outputs/whiteboard47.json."""
from _common import *
import torch, math, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); T = tok(); t = tt.t
V, NP, d, L, H, F = t["E"].shape[0], t["P"].shape[0], t["E"].shape[1], tt.config["n_layers"], tt.config["n_heads"], t["h.0.ff.W_1"].shape[1]
per_layer = 2 * d + 4 * d * d + 2 * d + d * F + F + F * d + d
hand = V * d + NP * d + L * per_layer + 2 * d
actual = sum(v.numel() for v in t.values())
print(f"W1  vocab {V}, positions {NP}, width {d}, layers {L}, heads {H}, feed-forward {F}, tied output table: {tt.config['tied']}")
print(f"    per layer: norms 2x{2*d}, attention 4 x {d}x{d} = {4*d*d:,}, feed-forward {d}x{F} + {F} + {F}x{d} + {d} = {d*F+F+F*d+d:,}; total {per_layer:,}")
print(f"    by hand {V}x{d} + {NP}x{d} + {L}x{per_layer:,} + {2*d} = {hand:,};  counted from the tables {actual:,};  equal: {hand == actual}")
ids = T.encode(SENTENCE); n = len(ids); dh = d // H
with torch.no_grad():
    x = t["E"][ids] + t["P"][:n]
    xn = tt._norm(x, "h.0.ln_1")
    q, k, v = xn @ t["h.0.att.W_q"], xn @ t["h.0.att.W_k"], xn @ t["h.0.att.W_v"]
    q0, k0, v0 = q[:, :dh], k[:, :dh], v[:, :dh]
    sc = (q0[-1] @ k0.T) / math.sqrt(dh)            # the last token sees all seven, itself included
    sh = torch.softmax(sc, -1); out = sh @ v0
    full = torch.softmax((q0 @ k0.T / math.sqrt(dh)).masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), float("-inf")), -1)
    # every head by hand for the last token, joined and multiplied by the output table, against the model's own layer
    heads = []
    for h in range(H):
        a, b_, c = q[-1, h*dh:(h+1)*dh], k[:, h*dh:(h+1)*dh], v[:, h*dh:(h+1)*dh]
        heads.append(torch.softmax(a @ b_.T / math.sqrt(dh), -1) @ c)
    hand_out = torch.cat(heads) @ t["h.0.att.W_o"]
    model_out = tt._attention(xn, 0)[-1]
pieces = [T.decode([i]) for i in ids]
print(f"W2  tokens {pieces}; head width {dh}")
print("    scores for the last token ('at'), divided by sqrt(32): " + ", ".join(f"{float(s):.3f}" for s in sc))
print("    shares: " + ", ".join(f"{p!r} {float(s):.3f}" for p, s in zip(pieces, sh)) + f";  sum {float(sh.sum()):.6f}")
print(f"    largest share on {pieces[int(sh.argmax())]!r}; four heads by hand through the output table against the model's layer: largest difference {float((hand_out - model_out).abs().max()):.2e} (largest output {float(model_out.abs().max()):.3f})")
print(f"    the full 7x7 table has {int((full > 0).sum())} non-zero shares (lower triangle with diagonal: {n*(n+1)//2})")
cache = 2 * L * d * 4
print(f"W3  cache per token: 2 (key, value) x {L} layers x {d} numbers x 4 bytes = {cache:,} bytes; 1,024 tokens: {cache*1024:,} bytes")
save_json("whiteboard47", dict(hand=hand, actual=actual, per_layer=per_layer, pieces=pieces, scores=[float(s) for s in sc], shares=[float(s) for s in sh],
                               diff=float((hand_out - model_out).abs().max()), nonzero=int((full > 0).sum()), cache=cache))
