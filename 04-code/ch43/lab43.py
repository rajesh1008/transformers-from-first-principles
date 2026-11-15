"""Lab 43: check Tiny-ViT's pieces exactly before trusting any accuracy.  CPU, one thread, 32-bit, seed 1865."""
import os, math, numpy as np, torch
os.environ.setdefault("OMP_NUM_THREADS", "1"); torch.set_num_threads(1)
import vit
torch.manual_seed(1865); ok = 0; n = 0
def check(name, cond):
    global ok, n
    n += 1; ok += bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}")
Xtr, ytr, Xte, yte = vit.data()
check("split: 1,437 training and 360 test images", len(ytr) == 1437 and len(yte) == 360)
P = vit.patches(Xte)
check("patches: 16 patches of 4 pixels for every 8 x 8 image", tuple(P.shape) == (360, 16, 4))
back = P.reshape(360, 4, 4, 2, 2).permute(0, 1, 3, 2, 4).reshape(360, 8, 8)
check("patches: cutting and reassembling gives the image back exactly", torch.equal(back, Xte))
check("patches: patch 5 is rows 2-3, columns 2-3 of the image", torch.equal(P[0, 5], Xte[0, 2:4, 2:4].reshape(4)))
m = vit.ViT(); m.eval()
x = torch.cat([m.cls.expand(2, 1, vit.D), m.embed(P[:2])], 1) + m.posrows(); m.h[0](x)
a = m.h[0].att
check("attention: every row of every head sums to 1", torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-6))
check("attention: no causal mask (the first patch row reads the last patch)", bool((a[:, :, 1, -1] > 0).all()))
os.environ["POS"] = "none"; import importlib; importlib.reload(vit)
m2 = vit.ViT(); m2.eval(); perm = torch.tensor(np.random.default_rng(3).permutation(16))
with torch.no_grad(): o1 = m2(P[:8]); o2 = m2(P[:8][:, perm])
check("without positions, shuffling the patches leaves the scores unchanged (to 1e-5)", torch.allclose(o1, o2, atol=1e-5))
os.environ["POS"] = "learned"; importlib.reload(vit)
m3 = vit.ViT(); m3.eval()
with torch.no_grad(): o1 = m3(P[:8]); o2 = m3(P[:8][:, perm])
check("with learned positions, shuffling the patches changes the scores", not torch.allclose(o1, o2, atol=1e-3))
cnt = 4 * 64 + 64 + 64 + 17 * 64 + 4 * (2 * 64 + 64 * 192 + 192 + 64 * 64 + 64 + 2 * 64 + 64 * 256 + 256 + 256 * 64 + 64) + 2 * 64 + 64 * 10 + 10
check(f"learned numbers counted by hand ({cnt:,}) equal the model's ({vit.count(m3):,})", cnt == vit.count(m3))
print(f"{ok} of {n} checks pass")
