"""Chapter 14, sections 2 and 3: repeating the block.  Twelve untrained pre-norm layers (seed 1865 + 100 l),
the running sentence and the first 1,024 tokens of the novel.  Follows ' sat''s row up the stack:
its length, what each block writes, how far it turns, and what the Logit Lens (ln_f, then the tied token
table) would read from it at each height.  Then the accumulation reference: what adding equal, unrelated
writes does to the cosine between neighbouring layers, whatever the writes are.
Run from 04-code/ch14:  python3 stack.py"""
from _common import *
from tiny.model import Tiny
from tiny.norm import token_similarity

t = tok(); ids = sentence_ids(t); pieces = [piece(t, i) for i in ids]
m = Tiny()
k = pieces.index(" sat")
S, stream = m.forward(ids, keep=True)
print(f"seed {SEED}; Tiny with {m.n_layers} layers, {m.n_params:,} learned numbers; ' sat' is row {k} of {len(ids)}")
print(" enter | ' sat' length | attention writes | feed-forward writes | cos with row below | cos with row 0 | Logit Lens top 3 for ' sat' | similarity")
for l in range(m.n_layers + 1):
    x = stream[l][k]
    if l < m.n_layers:
        st = m.blocks[l].explode(stream[l])
        aw, fw = length(st["attention"][k]), length(st["feed_forward"][k])
    sc = m.scores_from(m.top_rows(stream[l][k:k + 1]))[0]
    top = [piece(t, i) for i in np.argsort(-sc)[:3]]
    below = f"{cos(x, stream[l - 1][k]):.4f}" if l else "     -"
    label = f"{l:>5}" if l < m.n_layers else "  top"
    w = f"{aw:16.4f} | {fw:19.4f}" if l < m.n_layers else f"{'-':>16} | {'-':>19}"
    print(f" {label} | {length(x):13.4f} | {w} | {below:>18} | {cos(x, stream[0][k]):14.4f} | {str(top):<27} | {token_similarity(stream[l]):.3f}")
print("  (row l is the row entering layer l; 'top' is the row leaving layer 11, before ln_f)")

# the novel
ids_n = novel_ids(1024, t)
_, st_n = m.forward(ids_n, keep=True)
print("\nthe first 1,024 tokens of the novel:")
print(" enter | mean length | writes / row (att, ff) | mean cos with row below | similarity | Logit Lens: share of seats whose top token is the seat's own token")
rec = {"length": [], "sim": [], "own": [], "cos_below": [], "rel_att": [], "rel_ff": []}
for l in range(m.n_layers + 1):
    X = st_n[l]
    L = np.linalg.norm(X, axis=1)
    sc = m.scores_from(m.top_rows(X))
    own = float((sc.argmax(1) == np.asarray(ids_n)).mean())
    cb = float(np.mean([cos(a, b) for a, b in zip(X, st_n[l - 1])])) if l else float("nan")
    if l < m.n_layers:
        e = m.blocks[l].explode(X)
        ra = float((np.linalg.norm(e["attention"], axis=1) / L).mean()); rf = float((np.linalg.norm(e["feed_forward"], axis=1) / np.linalg.norm(e["add_1"], axis=1)).mean())
    else:
        ra = rf = float("nan")
    for key, v in zip(rec, [float(L.mean()), token_similarity(X), own, cb, ra, rf]):
        rec[key].append(v)
    label = f"{l:>5}" if l < m.n_layers else "  top"
    print(f" {label} | {L.mean():11.4f} | {ra:8.3f}, {rf:8.3f} | {cb:23.4f} | {token_similarity(X):10.3f} | {own:.3f}")
json.dump(rec, open(os.path.join(HERE, "outputs", "stack.json"), "w"), indent=1)
json.dump({"sat_length": [length(s[k]) for s in stream], "sat_cos0": [cos(s[k], stream[0][k]) for s in stream]},
          open(os.path.join(HERE, "outputs", "stack_sat.json"), "w"), indent=1)

# accumulation reference
print("\nthe accumulation reference: a start row and twelve writes of equal length, each unrelated (at right angles) to all the others")
rng = np.random.default_rng(SEED)
Q, _ = np.linalg.qr(rng.normal(size=(128, 13)))
U = Q.T                                    # 13 orthonormal rows: the start row and twelve writes
x = U[0].copy(); rows = [x.copy()]
for l in range(1, 13):
    x = x + U[l]; rows.append(x.copy())
print("  after write | cos with the row one layer below | rule sqrt(l / (l + 1)) | cos with the start row | rule 1 / sqrt(l + 1)")
for l in [1, 2, 3, 6, 9, 12]:
    print(f"  {l:>11} | {cos(rows[l], rows[l - 1]):32.4f} | {np.sqrt(l / (l + 1)):22.4f} | {cos(rows[l], rows[0]):22.4f} | {1 / np.sqrt(l + 1):.4f}")
