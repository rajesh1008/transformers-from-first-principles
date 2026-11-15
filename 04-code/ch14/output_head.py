"""Chapter 14, section 4: the output.  First by hand on the toy rows: sat's RMSNorm row from Chapter 13,
(0.683, 1.238), scored against the five distinct toy words' rows (Chapter 4) and turned into shares.
Then Tiny: ln_f, the tied token table, next-token scores (n, 2048), shares, the top five for the last
seat, the spread of the shares, and the untrained reference loss on the novel.  Also: the scores without
ln_f, and the count with an untied output table.
Run from 04-code/ch14:  python3 output_head.py"""
from _common import *
from tiny.model import Tiny
from tiny.attention import softmax, entropy

# by hand
words = ["the", "cat", "sat", "on", "mat"]
E2 = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [-1, 1]], dtype=float)
h = np.array([0.683, 1.238])
sc = E2 @ h
sh = softmax(sc)
print("by hand: sat's row (0.683, 1.238) after RMSNorm, scored against each toy word's row")
for w, r, s, p in zip(words, E2, sc, sh):
    print(f"  {w:<4} row {r.astype(int).tolist()!s:<8} score {s:+.3f}   e^score {np.exp(s):6.3f}   share {p:.3f}")
print(f"  sum of e^score {np.exp(sc).sum():.3f}; shares sum to {sh.sum():.3f}; top: {words[int(sh.argmax())]}")

# Tiny
t = tok(); ids = sentence_ids(t); pieces = [piece(t, i) for i in ids]
m = Tiny()
S, stream = m.forward(ids, keep=True)
P = softmax(S)
print(f"\nTiny, untrained (seed {SEED}), on the running sentence {pieces}")
print(f"  top rows {stream[-1].shape} -> ln_f {m.top_rows(stream[-1]).shape} -> scores {S.shape} -> shares {P.shape}")
print(f"  every row of shares sums to one: {bool(np.allclose(P.sum(1), 1))}")
last = P[-1]
top5 = np.argsort(-last)[:5]
print(f"  after '{pieces[-1]}' (seat {len(ids) - 1}), top five next tokens:")
for i in top5:
    print(f"    {piece(t, i)!r:<12} id {int(i):>4}  score {S[-1, i]:+.4f}  share {last[i]:.6f}")
print(f"  an even split would give every token 1/2048 = {1 / 2048:.6f}")
H = entropy(P)
print(f"  spread of the shares (entropy, Chapter 8): {H.min():.4f} to {H.max():.4f}; an even split gives ln 2048 = {np.log(2048):.4f}")
print(f"  scores for the last seat: smallest {S[-1].min():+.3f}, largest {S[-1].max():+.3f}")

# no ln_f
m0 = Tiny(final_norm=False)
S0 = m0.forward(ids)
print(f"\nwithout ln_f: the top row for '{pieces[-1]}' is {length(stream[-1][-1]):.3f} long instead of {length(m.top_rows(stream[-1])[-1]):.3f}")
print(f"  scores for the last seat run from {S0[-1].min():+.4f} to {S0[-1].max():+.4f}; largest share {softmax(S0)[-1].max():.6f} against {last.max():.6f} with ln_f")

# reference loss
ids_n = novel_ids(1025, t)
L = m.loss(ids_n)
print(f"\nuntrained reference loss on the first 1,024 predictions of the novel: {L:.4f} (an even split scores ln 2048 = {np.log(2048):.4f})")
print(f"  without ln_f: {m0.loss(ids_n):.4f}")

# counts
mu = Tiny(tied=False)
print(f"\nlearned numbers: tied output {m.n_params:,}; untied output table adds {mu.n_params - m.n_params:,} -> {mu.n_params:,}")
print(f"  the token table is {m.E.table.size / m.n_params:.1%} of Tiny tied")
json.dump({"top5": [piece(t, i) for i in top5], "share_max": float(last.max()), "entropy_min": float(H.min()), "loss": L},
          open(os.path.join(HERE, "outputs", "output_head.json"), "w"), indent=1)
