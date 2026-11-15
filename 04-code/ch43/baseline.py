"""Chapter 43: three non-transformer baselines on the same split as vit.py, so Tiny-ViT has something to beat.
  softmax  one table, 64 pixels -> 10 scores (multinomial logistic regression)
  mlp      64 -> 256 -> 10 with GELU
  cnn      two 3x3 convolutions (16 and 32 channels, GELU, the second followed by 2x2 max-pooling) and a table to 10
AdamW, lr 1e-3 cosine, weight decay 0.05, batch 64, EPOCHS (30), seed SEED (1865), FRAC (share of training images, as in vit.py).  CPU, one thread, 32-bit."""
import os, json, math, numpy as np, torch
os.environ.setdefault("OMP_NUM_THREADS", "1"); torch.set_num_threads(1)
from vit import data, HERE
EPOCHS = int(os.environ.get("EPOCHS", "30")); SEED = int(os.environ.get("SEED", "1865")); FRAC = float(os.environ.get("FRAC", "1"))
def make(kind):
    nn = torch.nn
    if kind == "softmax": return nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
    if kind == "mlp": return nn.Sequential(nn.Flatten(), nn.Linear(64, 256), nn.GELU(approximate="tanh"), nn.Linear(256, 10))
    return nn.Sequential(nn.Unflatten(1, (1, 8)), nn.Conv2d(1, 16, 3, padding=1), nn.GELU(approximate="tanh"),
                         nn.Conv2d(16, 32, 3, padding=1), nn.GELU(approximate="tanh"), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 16, 10))
Xtr, ytr, Xte, yte = data(); R = {}
if FRAC < 1: k = int(round(FRAC * len(ytr))); Xtr, ytr = Xtr[:k], ytr[:k]; EPOCHS = int(round(EPOCHS / FRAC))   # same number of steps
for kind in ("softmax", "mlp", "cnn"):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED); m = make(kind)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05); total = EPOCHS * math.ceil(len(ytr) / 64); step = 0
    for ep in range(EPOCHS):
        o = rng.permutation(len(ytr))
        for i in range(0, len(ytr), 64):
            for g in opt.param_groups: g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / total))
            b = o[i:i + 64]; L = torch.nn.functional.cross_entropy(m(Xtr[b]), ytr[b]); opt.zero_grad(); L.backward(); opt.step(); step += 1
        if ep == 0:
            with torch.no_grad(): a1 = float((m(Xte).argmax(-1) == yte).float().mean())
    with torch.no_grad(): a = float((m(Xte).argmax(-1) == yte).float().mean())
    n = sum(p.numel() for p in m.parameters()); R[kind] = dict(params=n, test=a, test_epoch1=a1)
    print(f"{kind:8s} learned numbers {n:7,d}  test accuracy after 1 epoch {a1:.4f}, after {EPOCHS} {a:.4f}", flush=True)
json.dump(R, open(os.path.join(HERE, "outputs", f"baseline_s{SEED}" + (f"_f{FRAC}" if FRAC < 1 else "") + ".json"), "w"), indent=1)
