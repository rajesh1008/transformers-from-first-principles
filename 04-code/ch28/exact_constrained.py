"""Chapter 28, section 28.7: constrained decoding corrected.  (A) Chapter 27's six names: exact conditional sampling
(whole-option scores, renormalised) against token-by-token masking, 400 draws each.  (B) A small grammar, a number of one
to three digits (first token may carry a leading space) followed by any token that does not start with a digit: the exact
conditional distribution over every valid output (summed over every way of spelling it in tokens) against what masking
one step at a time produces.  32-bit, one thread, seed 1865."""
from _common import *
import torch
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G

tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok()
out = {}
print("Part A: six names after '\"Come here, my dear,\" said'")
P = t.encode('"Come here, my dear," said')
names = [" Alice", " Bruno", " Sylvie", " Balbus", " the Hatter", " the Queen"]
opts = [t.encode(n) for n in names]
ex = G.exact_option_shares(tt, P, opts); loc = G.local_mask_probs(tt, P, opts)
rng = streams(SEED)["sample"]; d_ex = np.bincount(rng.choice(6, size=400, p=ex), minlength=6) / 400
rng = streams(SEED)["sample"]; d_lo = np.bincount(rng.choice(6, size=400, p=loc), minlength=6) / 400
for i, n in enumerate(names):
    print(f"  {n!r:<13} exact {ex[i]:.3f}  400 exact draws {d_ex[i]:.3f}   masked {loc[i]:.3f}  400 masked draws {d_lo[i]:.3f}")
tv = 0.5 * np.abs(ex - loc).sum(); print(f"  total-variation distance, exact vs masked: {tv:.3f}")
calls = sum(len(o) for o in opts); print(f"  cost of exact: {calls} next-token passes for six options (masking: 1 to 5 per draw)")
out["names"] = dict(exact=r3(ex), masked=r3(loc), tv=float(tv))

vocab = [t.decode([i]) for i in range(2048)]
import re
dig = [i for i, s in enumerate(vocab) if re.fullmatch(r"\d+", s)]
sdig = [i for i, s in enumerate(vocab) if re.fullmatch(r" \d+", s)]
starts_digit = np.array([bool(re.match(r"\d", s)) for s in vocab])
print(f"\nPart B: digits grammar; vocabulary has {len(dig)} digit-only tokens and {len(sdig)} space+digit tokens:",
      " ".join(repr(vocab[i]) for i in (dig + sdig)[:40]))
MAXLEN = 3

def walk(prompt):
    """Return exact[string] = model probability of (number, then a non-digit), and masked[string] = chance that
    step-by-step masking writes it.  Both sum over spellings."""
    base = t.encode(prompt); exact, masked = {}, {}; nodes = [0]
    def rec(prefix, text, pe, pm):
        nodes[0] += 1
        p = G.probs(G.next_scores(tt, base + prefix))
        digits = text.strip()
        cands = [(j, vocab[j]) for j in (sdig + dig if not prefix else dig) if len(digits) + len(vocab[j].strip()) <= MAXLEN]
        end = float(p[~starts_digit].sum()) if digits else 0.0
        if len(digits) == MAXLEN:
            cands = []
        tot = end + sum(p[j] for j, _ in cands)
        if digits:
            exact[digits] = exact.get(digits, 0) + pe * end
            masked[digits] = masked.get(digits, 0) + pm * (end / tot if tot > 0 else 1.0)
        for j, s in cands:
            if pe * p[j] < 1e-12 and pm * p[j] / tot < 1e-9:
                continue
            rec(prefix + [j], text + s, pe * p[j], pm * p[j] / tot)
    rec([], "", 1.0, 1.0)
    Z = sum(exact.values()); exact = {k: v / Z for k, v in exact.items()}
    return exact, masked, Z, nodes[0]

for prompt in ["It was a hundred and", "in the year", "CHAPTER"]:
    exact, masked, Z, nodes = walk(prompt)
    keys = set(exact) | set(masked)
    tv = 0.5 * sum(abs(exact.get(k, 0) - masked.get(k, 0)) for k in keys)
    p0 = G.probs(G.next_scores(tt, t.encode(prompt)))
    print(f"\n  prompt {prompt!r}: unconstrained chance the next token starts a number {p0[sdig + dig].sum():.4f}; "
          f"probability of any valid output {Z:.2e}; {nodes} prefixes visited; total-variation exact vs masked {tv:.3f}")
    te = sorted(exact.items(), key=lambda kv: -kv[1])[:6]; tm = sorted(masked.items(), key=lambda kv: -kv[1])[:6]
    print("    exact  top: " + ", ".join(f"{k} {v:.3f}" for k, v in te))
    print("    masked top: " + ", ".join(f"{k} {v:.3f}" for k, v in tm))
    l1 = sum(v for k, v in exact.items() if len(k) == 1); m1 = sum(v for k, v in masked.items() if len(k) == 1)
    print(f"    one-digit outputs: exact {l1:.3f}  masked {m1:.3f};  three-digit: exact {sum(v for k, v in exact.items() if len(k) == 3):.3f}"
          f"  masked {sum(v for k, v in masked.items() if len(k) == 3):.3f}")
    out[prompt] = dict(tv=tv, Z=Z, nodes=nodes, exact_top=te, masked_top=tm, one_exact=l1, one_masked=m1)
save_json("exact_constrained", out)
