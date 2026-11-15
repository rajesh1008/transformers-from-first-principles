"""Chapter 48's capstone model.  Same blocks as Tiny (Chapters 8 to 16): token rows plus learned position rows, pre-norm layer
norm, causal multi-head attention, a GELU feed-forward block four times the width, a final norm and the token table reused as
the output table.  Starts as Chapter 19 chose: normal(0, 0.02), with the two tables that write into the residual stream scaled
by 1/sqrt(2 x layers)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        s.up, s.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
    def forward(s, x):
        B, n, d = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(d, -1)
        q, k, v = (t.view(B, n, s.h, d // s.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, n, d)
        x = x + s.o(a)
        return x + s.down(F.gelu(s.up(s.ln2(x)), approximate="tanh"))

class Capstone(nn.Module):
    def __init__(s, vocab=8192, ctx=256, d=256, layers=10, heads=8):
        super().__init__(); s.cfg = dict(vocab=vocab, ctx=ctx, d=d, layers=layers, heads=heads)
        s.E, s.P = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers)); s.lnf = nn.LayerNorm(d)
        for name, p in s.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * layers) if name.endswith(("o.weight", "down.weight")) else 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)
    def forward(s, ids):
        x = s.E(ids) + s.P(torch.arange(ids.shape[1]))
        for b in s.blocks: x = b(x)
        return s.lnf(x) @ s.E.weight.T
    def count(s):
        return sum(p.numel() for p in s.parameters())
