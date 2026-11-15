"""Chapter 43: Tiny-ViT.  The same pre-norm transformer block as Tiny, reading an image as a sequence of patch rows.
Data: the 1,797 8x8 handwritten digits of the UCI Optical Recognition of Handwritten Digits set (CC BY 4.0), as shipped
inside scikit-learn (no download).  Pixels are 0..16; we divide by 16.  A fixed split (seed 1865): 1,437 train, 360 test.
Env:  NAME, P (patch side: 1, 2 or 4; default 2, a 4x4 grid of 16 patches), D (width, 64), L (layers, 4), H (heads, 4),
      POS (learned | none | rowcol | sincos2d), POOL (cls | mean), EPOCHS (30), SEED (1865), SHUF (train on patches in a
      new random order for every image: 0 or 1), FIXPERM (train and test with one fixed scramble of the patch order: 0/1),
      WARM (warm-up steps, 50; 23 steps make one epoch), FRAC (share of the 1,437 training images used, 1.0;
      the same first images of the fixed shuffle, and the same number of steps as the full set: epochs are scaled up).
CPU, one thread, 32-bit.  Writes outputs/NAME.json and outputs/ckpt/NAME.pt."""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
env = os.environ.get
NAME = env("NAME", "vit"); P = int(env("P", "2")); D = int(env("D", "64")); NL = int(env("L", "4")); NH = int(env("H", "4"))
POS = env("POS", "learned"); POOL = env("POOL", "cls"); EPOCHS = int(env("EPOCHS", "30")); SEED = int(env("SEED", "1865"))
SHUF = int(env("SHUF", "0")); FIXPERM = int(env("FIXPERM", "0")); WARM = int(env("WARM", "50")); FRAC = float(env("FRAC", "1"))
G = 8 // P; N = G * G                                    # patches per side, patches per image

def data():
    from sklearn.datasets import load_digits
    d = load_digits(); X = d.images.astype(np.float32) / 16.0; y = d.target.astype(np.int64)
    idx = np.random.default_rng(1865).permutation(len(y)); te, tr = idx[:360], idx[360:]
    return torch.tensor(X[tr]), torch.tensor(y[tr]), torch.tensor(X[te]), torch.tensor(y[te])

def patches(X, p=None):
    """(B, 8, 8) -> (B, N, P*P): cut into a G x G grid, read left to right, top to bottom, like words on a page."""
    p = p or P; g = 8 // p; B = X.shape[0]
    return X.reshape(B, g, p, g, p).permute(0, 1, 3, 2, 4).reshape(B, g * g, p * p)

def sincos2d(g, d):
    """Fixed 2-D position rows: half the width encodes the row, half the column, each with Chapter 5's sines and cosines."""
    def one(n, k):
        pos = torch.arange(n, dtype=torch.float32)[:, None]; i = torch.arange(k // 2, dtype=torch.float32)[None]
        a = pos / (100.0 ** (2 * i / k)); return torch.cat([a.sin(), a.cos()], 1)
    r = one(g, d // 2); c = one(g, d // 2)
    return torch.cat([r[:, None, :].expand(g, g, d // 2), c[None, :, :].expand(g, g, d // 2)], -1).reshape(g * g, d)

class Block(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.ln1 = torch.nn.LayerNorm(D); s.ln2 = torch.nn.LayerNorm(D)
        s.qkv = torch.nn.Linear(D, 3 * D); s.o = torch.nn.Linear(D, D)
        s.ff = torch.nn.Sequential(torch.nn.Linear(D, 4 * D), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(4 * D, D))
        s.att = None
    def forward(s, x):
        B, n, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1)
        sp = lambda t: t.view(B, n, NH, D // NH).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D // NH), -1)   # no causal mask: every patch sees every patch
        s.att = a.detach()
        x = x + s.o((a @ v).transpose(1, 2).reshape(B, n, D)); return x + s.ff(s.ln2(x))

class ViT(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.embed = torch.nn.Linear(P * P, D)            # the "tokenizer": one table for every patch
        s.cls = torch.nn.Parameter(torch.zeros(1, 1, D))
        if POS == "learned": s.pos = torch.nn.Parameter(torch.randn(1, N + 1, D) * 0.02)
        elif POS == "rowcol":
            s.row = torch.nn.Parameter(torch.randn(G, D) * 0.02); s.col = torch.nn.Parameter(torch.randn(G, D) * 0.02)
            s.cpos = torch.nn.Parameter(torch.randn(1, 1, D) * 0.02)
        elif POS == "sincos2d": s.register_buffer("fixed", sincos2d(G, D))
        s.h = torch.nn.ModuleList([Block() for _ in range(NL)]); s.lnf = torch.nn.LayerNorm(D); s.head = torch.nn.Linear(D, 10)
    def posrows(s):
        if POS == "learned": return s.pos
        if POS == "rowcol": return torch.cat([s.cpos, (s.row[:, None, :] + s.col[None, :, :]).reshape(1, N, D)], 1)
        if POS == "sincos2d": return torch.cat([torch.zeros(1, 1, D), s.fixed[None]], 1)
        return torch.zeros(1, N + 1, D)
    def forward(s, pt, order=None):
        x = s.embed(pt); B = x.shape[0]
        x = torch.cat([s.cls.expand(B, 1, D), x], 1) + s.posrows()
        if order is not None: x = torch.cat([x[:, :1], x[:, 1:][:, order]], 1)   # move patch rows (with their positions)
        for b in s.h: x = b(x)
        x = s.lnf(x); z = x[:, 0] if POOL == "cls" else x[:, 1:].mean(1)
        return s.head(z)

def count(m): return sum(p.numel() for p in m.parameters())

def accuracy(m, Xp, y, bs=360):
    m.eval()
    with torch.no_grad(): r = float((m(Xp).argmax(-1) == y).float().mean())
    m.train(); return r

if __name__ == "__main__":
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Xtr, ytr, Xte, yte = data()
    if FRAC < 1: k = int(round(FRAC * len(ytr))); Xtr, ytr = Xtr[:k], ytr[:k]
    Ptr, Pte = patches(Xtr), patches(Xte)
    perm = torch.tensor(np.random.default_rng(7).permutation(N)) if FIXPERM else None
    if FIXPERM: Ptr, Pte = Ptr[:, perm], Pte[:, perm]
    m = ViT(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    steps_per = math.ceil(len(ytr) / 64); EPOCHS = int(round(EPOCHS / FRAC)) if FRAC < 1 else EPOCHS; total = EPOCHS * steps_per
    lr = lambda s: 1e-3 * (s / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / (total - WARM))))
    R = dict(name=NAME, P=P, D=D, L=NL, H=NH, pos=POS, pool=POOL, epochs=EPOCHS, seed=SEED, shuf=SHUF, fixperm=FIXPERM, warm=WARM, frac=FRAC, train_images=len(ytr),
             tokens=N + 1, params=count(m), log=[])
    print(f"{NAME}: patch {P}x{P}, {N} patches + 1 class row, width {D}, {NL} layers, {NH} heads, positions {POS}, pool {POOL}; learned numbers {count(m):,}", flush=True)
    t0 = time.perf_counter(); step = 0
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(len(ytr)); tl = []
        for i in range(0, len(ytr), 64):
            b = order[i:i + 64]; pt = Ptr[b]
            if SHUF: pt = torch.stack([r[torch.tensor(rng.permutation(N))] for r in pt])
            for g in opt.param_groups: g["lr"] = lr(step)
            L = torch.nn.functional.cross_entropy(m(pt), ytr[b]); opt.zero_grad(); L.backward(); opt.step(); tl.append(L.item()); step += 1
        acc = accuracy(m, Pte, yte); R["log"].append(dict(epoch=ep, train=float(np.mean(tl)), test=acc))
        if ep in (1, 2, 5, 10, 20, 30) or ep == EPOCHS: print(f"epoch {ep:3d}  train loss {np.mean(tl):.4f}  test accuracy {acc:.4f}", flush=True)
    R["test"] = accuracy(m, Pte, yte)
    # the breakage: shuffle the patch order at test time (each image its own random order)
    g = np.random.default_rng(99); sh = torch.stack([r[torch.tensor(g.permutation(N))] for r in Pte])
    R["test_shuffled"] = accuracy(m, sh, yte)
    print(f"test accuracy {R['test']:.4f}; with patches shuffled at test time {R['test_shuffled']:.4f}  ({time.perf_counter()-t0:.0f} s)", flush=True)
    json.dump(R, open(os.path.join(HERE, "outputs", NAME + ".json"), "w"), indent=1)
    os.makedirs(os.path.join(HERE, "outputs", "ckpt"), exist_ok=True); torch.save(m.state_dict(), os.path.join(HERE, "outputs", "ckpt", NAME + ".pt"))
