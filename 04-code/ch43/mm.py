"""Chapter 43, 'multimodal in one picture': a small model that looks at a digit and writes its English name, letter by letter.
The sequence is 16 image rows followed by the letters: [16 patch rows] <s> s e v e n </s>.  Image rows see each other
(no mask among them); each letter sees every image row and the letters before it (a prefix mask).  Loss on the letters only.
MODE=scratch  the patch table and a 2-layer, width-64 decoder are trained together from nothing.
MODE=frozen   the 16 rows come from Tiny-ViT (outputs/ckpt/vit.pt, all four blocks, frozen), passed through one trained
              table ("the connector") into the 2-layer decoder, which is trained.
Test: greedy decoding on the 360 test images, exact match of the whole name; then the same with every image blanked.
CPU, one thread, 32-bit, seed SEED (1865), EPOCHS (30)."""
import os, json, math, numpy as np, torch
os.environ.setdefault("OMP_NUM_THREADS", "1"); torch.set_num_threads(1)
import vit
from vit import Block, data, patches, HERE, D, N
MODE = os.environ.get("MODE", "scratch"); SEED = int(os.environ.get("SEED", "1865")); EPOCHS = int(os.environ.get("EPOCHS", "30"))
NAMES = "zero one two three four five six seven eight nine".split()
CH = ["<s>", "</s>"] + sorted(set("".join(NAMES))); ci = {c: i for i, c in enumerate(CH)}; T = 7   # longest name + 2
def enc(w): return [0] + [ci[c] for c in w] + [1] + [1] * (T - len(w) - 2)
class MM(torch.nn.Module):
    def __init__(s):
        super().__init__()
        if MODE == "frozen":
            s.vit = vit.ViT(); s.vit.load_state_dict(torch.load(os.path.join(HERE, "outputs", "ckpt", "vit.pt")))
            for p in s.vit.parameters(): p.requires_grad_(False)
            s.conn = torch.nn.Linear(D, D)
        else: s.embed = torch.nn.Linear(4, D); s.ipos = torch.nn.Parameter(torch.randn(1, N, D) * 0.02)
        s.tok = torch.nn.Embedding(len(CH), D); s.tpos = torch.nn.Parameter(torch.randn(1, T, D) * 0.02)
        s.h = torch.nn.ModuleList([Block() for _ in range(2)]); s.lnf = torch.nn.LayerNorm(D); s.out = torch.nn.Linear(D, len(CH))
        n = N + T; mask = torch.zeros(n, n, dtype=torch.bool); mask[:, :N] = True
        mask[N:, N:] = torch.tril(torch.ones(T, T, dtype=torch.bool)); s.register_buffer("mask", mask)
    def image_rows(s, pt):
        if MODE == "frozen":
            with torch.no_grad():
                v = s.vit; x = torch.cat([v.cls.expand(pt.shape[0], 1, D), v.embed(pt)], 1) + v.posrows()
                for b in v.h: x = b(x)
                x = v.lnf(x)[:, 1:]
            return s.conn(x)
        return s.embed(pt) + s.ipos
    def forward(s, pt, txt):
        x = torch.cat([s.image_rows(pt), s.tok(txt) + s.tpos[:, :txt.shape[1]]], 1); n = x.shape[1]; M = s.mask[:n, :n]
        for b in s.h:   # Chapter 43's Block without a mask, given one here
            B = x.shape[0]; q, k, v = b.qkv(b.ln1(x)).split(D, -1); sp = lambda t: t.view(B, n, 4, D // 4).transpose(1, 2)
            a = torch.nn.functional.scaled_dot_product_attention(sp(q), sp(k), sp(v), attn_mask=M)
            x = x + b.o(a.transpose(1, 2).reshape(B, n, D)); x = x + b.ff(b.ln2(x))
        return s.out(s.lnf(x[:, N:]))
def greedy(m, pt):
    txt = torch.zeros(pt.shape[0], 1, dtype=torch.long)
    for _ in range(T - 1): txt = torch.cat([txt, m(pt, txt)[:, -1].argmax(-1, keepdim=True)], 1)
    out = []
    for r in txt.tolist():
        w = "".join(CH[i] for i in r[1:] if i > 1) if 1 not in r[1:] else "".join(CH[i] for i in r[1:r.index(1, 1)])
        out.append(w)
    return out
if __name__ == "__main__":
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Xtr, ytr, Xte, yte = data(); Ptr, Pte = patches(Xtr), patches(Xte)
    Ytr = torch.tensor([enc(NAMES[i]) for i in ytr]); m = MM()
    ps = [p for p in m.parameters() if p.requires_grad]; opt = torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.05)
    total = EPOCHS * math.ceil(len(ytr) / 64); step = 0; R = dict(mode=MODE, seed=SEED, trained=sum(p.numel() for p in ps), log=[])
    print(f"mm {MODE}: trained learned numbers {R['trained']:,}; sequence 16 image rows + {T} letter slots", flush=True)
    for ep in range(1, EPOCHS + 1):
        o = rng.permutation(len(ytr)); tl = []
        for i in range(0, len(ytr), 64):
            for g in opt.param_groups: g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / total))
            b = o[i:i + 64]; y = Ytr[b]; lg = m(Ptr[b], y[:, :-1])
            L = torch.nn.functional.cross_entropy(lg.reshape(-1, len(CH)), y[:, 1:].reshape(-1)); opt.zero_grad(); L.backward(); opt.step(); tl.append(L.item()); step += 1
        if ep in (1, 5, 10, 30) or ep == EPOCHS:
            m.eval()
            with torch.no_grad(): w = greedy(m, Pte)
            m.train(); acc = float(np.mean([a == NAMES[t] for a, t in zip(w, yte.tolist())]))
            R["log"].append(dict(epoch=ep, train=float(np.mean(tl)), exact=acc)); print(f"epoch {ep:3d}  letter loss {np.mean(tl):.4f}  exact names on test {acc:.4f}", flush=True)
    m.eval()
    with torch.no_grad(): w = greedy(m, Pte); wb = greedy(m, torch.zeros_like(Pte))
    R["exact"] = float(np.mean([a == NAMES[t] for a, t in zip(w, yte.tolist())])); R["examples"] = list(zip(w[:8], [NAMES[t] for t in yte[:8].tolist()]))
    from collections import Counter
    R["blank"] = Counter(wb).most_common(3); R["not_a_name"] = int(sum(a not in NAMES for a in w))
    print(f"exact names {R['exact']:.4f}; outputs that are not a digit name {R['not_a_name']}; first eight: " + ", ".join(f"{a} ({b})" for a, b in R["examples"]))
    print(f"every image blanked: the commonest outputs {R['blank']}")
    json.dump(R, open(os.path.join(HERE, "outputs", f"mm_{MODE}_s{SEED}.json"), "w"), indent=1)
