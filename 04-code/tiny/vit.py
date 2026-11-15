"""Chapter 43 adds vision: cut an image into patches and read them as tokens.  The parameterised form of 04-code/ch43/vit.py,
so later code can build a Tiny-ViT without environment variables.  No causal mask; a class row; learned or 2-D sine positions."""
import math, torch

def patches(X, p=2):
    """(B, S, S) -> (B, (S/p)^2, p*p), patches read left to right, top to bottom."""
    B, S, _ = X.shape; g = S // p
    return X.reshape(B, g, p, g, p).permute(0, 1, 3, 2, 4).reshape(B, g * g, p * p)

def sincos2d(g, d):
    """Fixed 2-D position rows: half the width encodes the row, half the column (Chapter 5's sines and cosines)."""
    def one(n, k):
        pos = torch.arange(n, dtype=torch.float32)[:, None]; i = torch.arange(k // 2, dtype=torch.float32)[None]
        a = pos / (100.0 ** (2 * i / k)); return torch.cat([a.sin(), a.cos()], 1)
    r = one(g, d // 2); c = one(g, d // 2)
    return torch.cat([r[:, None, :].expand(g, g, d // 2), c[None, :, :].expand(g, g, d // 2)], -1).reshape(g * g, d)

class Block(torch.nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.d, s.h = d, h; s.ln1 = torch.nn.LayerNorm(d); s.ln2 = torch.nn.LayerNorm(d)
        s.qkv = torch.nn.Linear(d, 3 * d); s.o = torch.nn.Linear(d, d)
        s.ff = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(4 * d, d))
    def forward(s, x):
        B, n, D = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1)
        sp = lambda t: t.view(B, n, s.h, D // s.h).transpose(1, 2)
        a = torch.softmax(sp(q) @ sp(k).transpose(-1, -2) / math.sqrt(D // s.h), -1)      # every patch sees every patch
        x = x + s.o((a @ sp(v)).transpose(1, 2).reshape(B, n, D)); return x + s.ff(s.ln2(x))

class TinyViT(torch.nn.Module):
    def __init__(s, side=8, p=2, d=64, layers=4, heads=4, classes=10, pos="learned"):
        super().__init__(); s.p, s.g = p, side // p; n = s.g * s.g
        s.embed = torch.nn.Linear(p * p, d); s.cls = torch.nn.Parameter(torch.zeros(1, 1, d))
        if pos == "learned": s.pos = torch.nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        else: s.register_buffer("pos", torch.cat([torch.zeros(1, d), sincos2d(s.g, d)])[None])
        s.blocks = torch.nn.ModuleList([Block(d, heads) for _ in range(layers)]); s.lnf = torch.nn.LayerNorm(d)
        s.head = torch.nn.Linear(d, classes)
    def forward(s, X):
        x = s.embed(patches(X, s.p)); x = torch.cat([s.cls.expand(x.shape[0], 1, -1), x], 1) + s.pos
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x)[:, 0])

if __name__ == "__main__":
    torch.manual_seed(1865); m = TinyViT()
    print(sum(q.numel() for q in m.parameters()), tuple(m(torch.rand(3, 8, 8)).shape))
