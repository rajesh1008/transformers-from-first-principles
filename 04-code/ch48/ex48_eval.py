"""Exercises 48.9 and 48.10 (added 16 November 2026), on the final checkpoint ckpt/main_step2000.pt.  CPU, one thread, 32-bit.
Same measure as eval48.py: non-overlapping windows of 256 predicted tokens, each window's first token as context only.
48.9: the last 5 per cent of each of the fourteen training books (by characters, cut by prep48.py's own body() function), against
      a middle 5 per cent of each book (characters 50 to 55 per cent) and against The Time Machine.  The final checkpoint was
      trained on all of these book slices (about three passes), so the last 5 per cent is not held out for this checkpoint.
48.10: every 2-D table rounded to 4 bits (symmetric, levels -7 to 7), once with one scale per output row and once with one scale
      per group of 32 numbers along each row; bits per character on The Time Machine and Through the Looking-Glass.
-> outputs/ex48_eval.json"""
import os, sys, json, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from tiny.capstone.model import Capstone
from tokenizers import Tokenizer
torch.set_num_threads(1)
C = os.path.join(HERE, "corpus"); tk = Tokenizer.from_file(os.path.join(C, "tok8192.json"))
ck = torch.load(os.path.join(HERE, "ckpt", "main_step2000.pt"), map_location="cpu", weights_only=True); cfg = ck["cfg"]
m = Capstone(**cfg); m.load_state_dict(ck["state"]); m.eval(); CTX = cfg["ctx"]
def bpc_ids(a):
    a = torch.as_tensor(a, dtype=torch.int64); k = (len(a) - 1) // CTX; w = torch.stack([a[i * CTX: i * CTX + CTX + 1] for i in range(k)])
    chars = sum(len(tk.decode(r[1:].tolist())) for r in w); nats = 0.0
    with torch.no_grad():
        for i in range(0, k, 16):
            b = w[i:i + 16]; nats += float(F.cross_entropy(m(b[:, :-1]).reshape(-1, cfg["vocab"]), b[:, 1:].reshape(-1), reduction="sum"))
    return dict(windows=k, chars=chars, loss=nats / (k * CTX), bpc=nats / math.log(2) / chars)
u16 = lambda n: np.fromfile(os.path.join(C, f"{n}.u16"), dtype=np.uint16).astype(np.int64)
src = open(os.path.join(HERE, "prep48.py")).read().split("books, parts, total = []")[0]      # only the definitions, not the writes
P = {"__file__": os.path.join(HERE, "prep48.py")}; exec(src, P)
books = json.load(open(os.path.join(C, "books.json")))["train"]
last, mid = [], []
for b in books:
    _, _, t = P["body"](b["id"]); n = len(t)
    last += tk.encode(t[int(0.95 * n):]).ids; mid += tk.encode(t[int(0.50 * n):int(0.55 * n)]).ids
R = dict(fp32=dict(heldout=bpc_ids(u16("heldout")), glass=bpc_ids(u16("glass")), last5=bpc_ids(last), mid5=bpc_ids(mid)))
for k, v in R["fp32"].items(): print(f"48.9  {k:8s} {v['windows']:3d} windows, {v['chars']:,} characters: loss {v['loss']:.4f} nats per token, {v['bpc']:.4f} bits per character", flush=True)
keep = {k: v.clone() for k, v in m.state_dict().items()}
def q4(p, group):
    r, c = p.shape; g = p.reshape(r, c // group, group) if group else p.reshape(r, 1, c)
    s = g.abs().amax(-1, keepdim=True) / 7.0; s[s == 0] = 1.0
    return (torch.clamp(torch.round(g / s), -7, 7) * s).reshape(r, c)
for lab, group in (("int4_row", None), ("int4_g32", 32)):
    m.load_state_dict(keep); err = 0.0
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2: q = q4(p, group); err = max(err, float((q - p).abs().max())); p.copy_(q)
    R[lab] = dict(heldout=bpc_ids(u16("heldout")), glass=bpc_ids(u16("glass")), maxerr=err)
    for k in ("heldout", "glass"):
        print(f"48.10 {lab}: {k:8s} {R[lab][k]['bpc']:.4f} bits per character (change {R[lab][k]['bpc'] - R['fp32'][k]['bpc']:+.4f}); largest rounding change {err:.3e}", flush=True)
json.dump(R, open(os.path.join(HERE, "outputs", "ex48_eval.json"), "w"), indent=1)
