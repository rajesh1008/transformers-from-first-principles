"""Chapter 14, section 14.4: what changes as you go up, measured in GPT-2 small (comparison only: GPT-2's
numbers, not Tiny's).  Text: the end-of-text token and the first 1,023 tokens of Chapter I of the novel
(the same 1,024 tokens as Chapter 13), float32.  The novel is old and freely available, so GPT-2 has very
probably seen it; the losses are reference points for the interventions, not measures of quality.
  A. The stream: mean row length entering each layer (seat 0 left out, Chapter 13), and for each block
     the length of its write relative to the row it is added to, and the cosine between the two.
  B. The Logit Lens: the row entering each layer put through ln_f and the token table, as if the stack
     stopped there.  Per layer: average loss, how often its top token is the final top token, and the
     average gap from the final shares (Kullback-Leibler, natural log).
  C. The running sentence: what each height predicts after "the cat sat on the", and the rank of ' mat'.
  D. Removing one whole layer at a time (both blocks skipped): change in average loss.
  E. Removing two layers at once, every pair: the extra cost beyond the two single removals, by distance.
Run from 04-code/ch14:  python3 gpt2_depth.py   (about eight minutes; 459 s on 2026-11-10)"""
from _common import *
from _gpt2run import weights, block, norm, softmax, L, decoder
from _bpe import encode
from tiny.corpus import load_text
import time

T0 = time.time()
W = weights()
dec = decoder()
alice = load_text()
i0 = alice.index("Alice was beginning")
ids = np.array([50256] + encode(alice[i0:i0 + 12000].replace("\r", ""))[:1023])
n = len(ids)
E = W["wte.weight"]

def lens(x):
    """ln_f then the tied token table: next-token scores for every seat, (n, 50257)."""
    return norm(x, W["ln_f.weight"], W["ln_f.bias"]) @ E.T

def logp_of(S):
    S = S - S.max(-1, keepdims=True)
    return S - np.log(np.exp(S).sum(-1, keepdims=True))

def loss_from_stream(x):
    lp = logp_of(lens(x[:-1]))
    return float(-lp[np.arange(n - 1), ids[1:]].mean())

def run(skip=(), start=0, x0=None):
    x = (E[ids] + W["wpe.weight"][:n]) if x0 is None else x0.copy()
    for l in range(start, L):
        if l not in skip:
            x, _ = block(x, l)
    return x

# A and B: one pass recording the stream and each block's writes
x = E[ids] + W["wpe.weight"][:n]
stream, writes = [x], []
for l in range(L):
    p = f"h.{l}."
    a = norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
    x_mid, _ = None, None
    # recompute the two writes separately (same arithmetic as _gpt2run.block)
    y, _ = block(x, l)
    # attention write = (x after attention) - x; feed-forward write = y - (x after attention)
    from _gpt2run import gelu, H, D
    qkv = a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
    q, k, v = (t_.reshape(n, H, D).transpose(1, 0, 2) for t_ in np.split(qkv, 3, axis=-1))
    s = q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(D))
    s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, np.float32(-1e9))
    att = (softmax(s) @ v).transpose(1, 0, 2).reshape(n, H * D) @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
    xa = x + att
    ffw = y - xa
    writes.append((att, ffw, x, xa))
    x = y
    stream.append(x)
final_lp = logp_of(lens(stream[-1][:-1]))
final_top = final_lp.argmax(-1)
final_p = np.exp(final_lp)
base = float(-final_lp[np.arange(n - 1), ids[1:]].mean())

print(f"GPT-2 small, {n} tokens of the novel (seat 0 = end-of-text), float32; full-model loss {base:.3f}")
print("A. the stream and what each block writes (seats 1 to 1,023)")
print("  layer | row entering: mean length | attention write / row, cos | feed-forward write / row, cos")
recA = []
for l, (att, ffw, xin, xa) in enumerate(writes):
    r = slice(1, None)
    Lx = np.linalg.norm(xin[r], axis=1); La = np.linalg.norm(xa[r], axis=1)
    ra = float((np.linalg.norm(att[r], axis=1) / Lx).mean()); rf = float((np.linalg.norm(ffw[r], axis=1) / La).mean())
    ca = float(np.mean((att[r] * xin[r]).sum(1) / (np.linalg.norm(att[r], axis=1) * Lx)))
    cf = float(np.mean((ffw[r] * xa[r]).sum(1) / (np.linalg.norm(ffw[r], axis=1) * La)))
    recA.append({"length": float(Lx.mean()), "ra": ra, "ca": ca, "rf": rf, "cf": cf})
    print(f"  {l:>5} | {Lx.mean():25.1f} | {ra:9.3f}, {ca:+.3f}       | {rf:9.3f}, {cf:+.3f}")
print(f"  top   | {np.linalg.norm(stream[-1][1:], axis=1).mean():25.1f} |")

print("\nB. the Logit Lens on the 1,023 predictions (row entering layer l, through ln_f and the token table)")
print("  height | loss   | top token = final top | gap from final shares (KL)")
recB = []
for l in range(L + 1):
    lp = logp_of(lens(stream[l][:-1]))
    loss = float(-lp[np.arange(n - 1), ids[1:]].mean())
    agree = float((lp.argmax(-1) == final_top).mean())
    kl = float((final_p * (final_lp - lp)).sum(-1).mean())
    recB.append({"loss": loss, "agree": agree, "kl": kl})
    lab = f"{l:>6}" if l < L else "   top"
    print(f"  {lab} | {loss:6.3f} | {agree:21.3f} | {kl:.3f}")
    del lp

print("\nC. the running sentence: what each height predicts after 'the cat sat on the'")
sid = np.array(encode("the cat sat on the mat"))
x = E[sid] + W["wpe.weight"][:len(sid)]
mat = int(sid[-1]); seat = len(sid) - 2
print(f"  GPT-2 tokens {list(map(int, sid))}: {[dec(i) for i in sid]}; seat {seat} predicts the next token")
recC = []
for l in range(L + 1):
    S = lens(x)[seat]
    order = np.argsort(-S)
    rank = int(np.where(order == mat)[0][0]) + 1
    top = [dec(i) for i in order[:3]]
    lab = f"{l:>6}" if l < L else "   top"
    pm = float(softmax(S[None])[0][mat])
    recC.append({"top": top, "rank_mat": rank, "share_mat": pm})
    print(f"  {lab} | top three {str(top):<36} | rank of '␣mat' {rank:>6} | share {pm:.5f}")
    if l < L:
        x, _ = block(x, l)

print("\nD. removing one layer (both its blocks) at a time: change in loss")
single = {}
for l in range(L):
    xs = run(skip=(l,), start=l, x0=stream[l])
    single[l] = loss_from_stream(xs) - base
    print(f"  layer {l:>2} removed: {single[l]:+.3f}")

print("\nE. removing two layers at once: extra cost beyond the two single removals (pair minus both singles)")
pairs = {}
for i in range(L):
    for j in range(i + 1, L):
        xs = run(skip=(i, j), start=i, x0=stream[i])
        pairs[(i, j)] = loss_from_stream(xs) - base
inter = {k: v - single[k[0]] - single[k[1]] for k, v in pairs.items()}
print("  distance | mean extra cost, layers 1 to 11 only | mean extra cost, pairs including layer 0 | pairs")
recE = {}
for dist in range(1, L):
    a = [v for (i, j), v in inter.items() if j - i == dist and i >= 1]
    b = [v for (i, j), v in inter.items() if j - i == dist and i == 0]
    recE[dist] = (float(np.mean(a)) if a else None, float(np.mean(b)) if b else None)
    print(f"  {dist:>8} | {np.mean(a) if a else float('nan'):+37.3f} | {np.mean(b) if b else float('nan'):+41.3f} | {len(a) + len(b)}")
best_pair = min(((k, v) for k, v in pairs.items() if k[0] >= 1), key=lambda kv: kv[1])
best_adj = min(((k, v) for k, v in pairs.items() if k[0] >= 1 and k[1] - k[0] == 1), key=lambda kv: kv[1])
print(f"  cheapest pair to remove (layer 0 kept): layers {best_pair[0]}, {best_pair[1]:+.3f}; cheapest adjacent pair: layers {best_adj[0]}, {best_adj[1]:+.3f}")
json.dump({"base": base, "A": recA, "B": recB, "C": recC, "single": single,
           "pairs": {f"{i},{j}": v for (i, j), v in pairs.items()}, "E": {str(k): v for k, v in recE.items()}},
          open(os.path.join(HERE, "outputs", "gpt2_depth.json"), "w"), indent=1)
print(f"\n(elapsed {time.time() - T0:.0f} s)")
