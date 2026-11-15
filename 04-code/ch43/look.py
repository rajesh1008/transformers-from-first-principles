"""Chapter 43: what a trained Tiny-ViT attends to and what its position rows learned.  For one run (NAME, default vit):
  1. mean attention distance by layer: for each head, the attention-weighted distance (in patch widths, on the 4x4 grid)
     from each patch to the patches it reads, averaged over the 360 test images; the class row is left out.
  2. the class row's attention to the 16 patches in the last layer, averaged over heads, for the first test image.
  3. learned position rows: cosine similarity between each pair of the 16 patch positions; the mean similarity of
     neighbours on the grid (sharing an edge) against non-neighbours, and of positions in the same row or column.
  4. occlusion: blank each patch (set it to 0) in turn and record the mean drop in the correct class's probability.
Writes outputs/look_NAME.json."""
import os, json, math, numpy as np, torch
NAME = os.environ.get("NAME", "vit")
from vit import HERE
cfg = json.load(open(os.path.join(HERE, "outputs", NAME + ".json")))
os.environ.update(P=str(cfg["P"]), D=str(cfg["D"]), L=str(cfg["L"]), H=str(cfg["H"]), POS=cfg["pos"], POOL=cfg["pool"])
import importlib, vit; importlib.reload(vit)
m = vit.ViT(); m.load_state_dict(torch.load(os.path.join(HERE, "outputs", "ckpt", NAME + ".pt"))); m.eval()
Xtr, ytr, Xte, yte = vit.data(); Pte = vit.patches(Xte); G = vit.G; N = vit.N
with torch.no_grad(): pr = torch.softmax(m(Pte), -1)
R = dict(name=NAME, test=float((pr.argmax(-1) == yte).float().mean()))
xy = torch.tensor([(i // G, i % G) for i in range(N)], dtype=torch.float32); dist = torch.cdist(xy, xy)
R["distance"] = []
for l, b in enumerate(m.h):
    a = b.att[:, :, 1:, 1:]; a = a / a.sum(-1, keepdim=True)             # patch-to-patch, renormalised without the class row
    d = (a * dist).sum(-1).mean((0, 2)).tolist(); R["distance"].append(d)
    cls_share = float(b.att[:, :, 1:, 0].mean())
    print(f"layer {l}: mean attention distance by head " + " ".join(f"{x:.2f}" for x in d) + f"  (patches' share of attention on the class row {cls_share:.3f})")
print(f"uniform attention over the grid would give {float(dist.mean()):.2f}")
with torch.no_grad(): m(Pte[:1])
c = m.h[-1].att[0, :, 0, 1:].mean(0).reshape(G, G); R["cls_map"] = c.tolist()
print(f"first test image (label {int(yte[0])}): last layer, class row's attention to the patches, mean over heads:")
for r in c: print("   " + " ".join(f"{x:.3f}" for x in r))
img = Xte[0]; print("   the image (pixel / 16, rounded to one decimal):"); [print("   " + " ".join(f"{x:.1f}" for x in row)) for row in img]
if cfg["pos"] == "learned":
    p = m.pos[0, 1:].detach(); p = p / p.norm(dim=-1, keepdim=True); S = (p @ p.T).numpy()
    nb = [(i, j) for i in range(N) for j in range(N) if i < j and dist[i, j] == 1]
    far = [(i, j) for i in range(N) for j in range(N) if i < j and dist[i, j] > 1]
    srow = [(i, j) for i in range(N) for j in range(N) if i < j and i // G == j // G]; scol = [(i, j) for i in range(N) for j in range(N) if i < j and i % G == j % G]
    f = lambda L: float(np.mean([S[i, j] for i, j in L]))
    R["pos_sim"] = dict(neighbours=f(nb), others=f(far), same_row=f(srow), same_col=f(scol), matrix=S.tolist())
    print(f"position rows, cosine similarity: neighbours {f(nb):.3f}, non-neighbours {f(far):.3f}, same row {f(srow):.3f}, same column {f(scol):.3f}")
    print("   similarity of each position to position 5 (row 1, column 1):"); [print("   " + " ".join(f"{S[5, r * G + q]:+.2f}" for q in range(G))) for r in range(G)]
drops = []
for i in range(N):
    q = Pte.clone(); q[:, i] = 0
    with torch.no_grad(): p2 = torch.softmax(m(q), -1)
    drops.append(float((pr.gather(1, yte[:, None]) - p2.gather(1, yte[:, None])).mean()))
R["occlusion"] = drops; print("occlusion: mean drop in the correct class's probability when one patch is blanked:")
for r in range(G): print("   " + " ".join(f"{drops[r * G + q]:.3f}" for q in range(G)))
json.dump(R, open(os.path.join(HERE, "outputs", f"look_{NAME}.json"), "w"), indent=1)
