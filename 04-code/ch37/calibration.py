"""Chapter 37, 37.2 and 37.4: confidence against correctness.  For Tiny (Chapter 24's checkpoint), GPT-2 small and
SmolLM2-135M, on Through the Looking-Glass (held out from Tiny's training; the two public models have probably read it):
for every predicted token, the top choice's probability (confidence) and whether the top choice is the actual next token.
Ten equal-width bins: mean confidence, accuracy and count per bin; expected calibration error (ECE, the count-weighted mean
of |accuracy - confidence|).  Then temperature scaling: fit one temperature on the first half of the tokens by minimising
loss, and report ECE on the second half before and after.  CPU, one thread, 32-bit, seed 1865."""
from _common import *
import torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from transformers import AutoTokenizer, AutoModelForCausalLM
from tiny.eval import load
from _pool import corpora
text = glass_text()
TEMPS = [round(float(t), 2) for t in np.arange(0.5, 2.51, 0.05)]
def collect(logit_fn, ids, win, n_win):
    """Per predicted token, for every temperature in TEMPS: top-1 confidence and loss; plus whether the top choice is right.
    (Full logit tables for GPT-2 over 20,000 tokens would need about 8 GB, so only these are kept.)"""
    C = []; Ls = []; R_ = []
    for w in range(n_win):
        chunk = ids[w * (win - 1): w * (win - 1) + win]
        if len(chunk) < win: break
        with torch.no_grad():
            lg = logit_fn(chunk)[:-1].double(); tg = torch.tensor(chunk[1:])
            R_.append((lg.argmax(-1) == tg).double())
            cs = []; ls = []
            for t in TEMPS:
                lp = torch.log_softmax(lg / t, -1); cs.append(lp.max(-1).values.exp()); ls.append(-lp[torch.arange(len(tg)), tg])
            C.append(torch.stack(cs, 1)); Ls.append(torch.stack(ls, 1))
    return torch.cat(C), torch.cat(Ls), torch.cat(R_)
def ece(conf, acc, bins=10):
    edges = torch.linspace(0, 1, bins + 1); e = 0.0; rows = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0: rows.append((float(edges[i]), 0, None, None)); continue
        c, a = float(conf[m].mean()), float(acc[m].mean()); e += m.sum().item() / len(conf) * abs(a - c)
        rows.append((float(edges[i]), int(m.sum()), round(c, 4), round(a, 4)))
    return e, float(conf.mean()), float(acc.mean()), rows
R = {}
models = []
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
g = [int(i) for i in corpora()["glass"]]
models.append(("Tiny", lambda ch: tt.scores(ch), g, 129, 160))
for name in ("gpt2", "smollm2"):
    tk = AutoTokenizer.from_pretrained(CKPT(name)); m = AutoModelForCausalLM.from_pretrained(CKPT(name), dtype=torch.float32).eval()
    models.append(({"gpt2": "GPT-2 small", "smollm2": "SmolLM2-135M"}[name], (lambda m: lambda ch: m(torch.tensor([ch])).logits[0])(m), tk.encode(text), 513, 40))
for label, fn, ids, win, nw in models:
    C, Ls, acc = collect(fn, ids, win, nw); n = len(acc); h = n // 2; i1 = TEMPS.index(1.0)
    e, c, a, rows = ece(C[:, i1], acc)
    j = int(Ls[:h].mean(0).argmin()); t = TEMPS[j]
    e1, _, _, _ = ece(C[h:, i1], acc[h:]); e2, c2, a2, _ = ece(C[h:, j], acc[h:])
    loss = float(Ls[:, i1].mean())
    print(f"{label}: {n:,} predictions, loss {loss:.4f} nats; top-1 accuracy {a:.4f}, mean top-1 confidence {c:.4f}, ECE {e:.4f}", flush=True)
    for lo, cnt, cc, aa in rows:
        print(f"   bin {lo:.1f}-{lo + 0.1:.1f}: {cnt:6d} tokens" + (f"  confidence {cc:.3f}  accuracy {aa:.3f}" if cnt else ""), flush=True)
    print(f"   temperature fitted on the first half: {t:.2f}; ECE on the second half {e1:.4f} -> {e2:.4f} (confidence {c2:.4f}, accuracy {a2:.4f})", flush=True)
    R[label] = dict(n=n, loss=loss, acc=a, conf=c, ece=e, bins=rows, temp=t, ece_half=e1, ece_half_scaled=e2)
save_json("calibration", R)
