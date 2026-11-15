# Lab 15: the hand trace as a notebook.  Cells are separated by '# %%' lines (the percent format that
# Jupyter, VS Code and jupytext open as a notebook).  build_notebook.py runs every cell in order and
# writes lab15.ipynb with the real outputs.  Run from 04-code/ch15.
# %% [markdown]
# # Lab 15: the hand trace as a notebook
# Part 1 runs the hand-sized Tiny and checks it against the pencil trace. Part 2 changes one number in the
# token table, three times, and follows each change to the output. Part 3 is the checkpoint.
# %%
import sys, os, math
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
from hand import hand_tiny, WORDS, IDS, SEAT_WORDS, W_q, W_k, W_v, W_o, W_1, b_1, W_2, b_2
from tiny.embed import TOY_EMBED, TOY_POS
from tiny.attention import softmax
import pencil
TAB = {k: v.tolist() for k, v in dict(W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, W_1=W_1, b_1=b_1, W_2=W_2, b_2=b_2).items()}
print("seed: none needed; every table is written by hand (the book's seed 1865 is not used in this lab)")
# %% [markdown]
# ## Part 1: the trace, by code and by pencil
# %%
m = hand_tiny()
S = m.forward(IDS)
sh = softmax(S)
pen = pencil.trace(TOY_EMBED.tolist(), TOY_POS.tolist(), IDS, TAB)
print("shares (6, 12); every row sums to one:", bool(np.allclose(sh.sum(1), 1)))
print("after mat@5, top three:", [(WORDS[k], round(float(sh[5, k]), 3)) for k in np.argsort(-sh[5])[:3]])
print("largest difference from the pencil trace:", f"{np.abs(sh - np.array(pen['out_shares'])).max():.1e}")
# %%
def steps(m, ids):
    """Every intermediate of the forward pass, by name, as (6, 2) or (6, 6) or (6, 12) tables."""
    X = m.embed(ids); out = [("rows in", X)]
    for l, b in enumerate(m.blocks):
        st = b.explode(X)
        _, s = b.att(b.ln_1(X), return_shares=True)
        out += [(f"L{l} ln_1", st["ln_1"]), (f"L{l} shares", s[0]), (f"L{l} attention", st["attention"]),
                (f"L{l} add_1", st["add_1"]), (f"L{l} ln_2", st["ln_2"]), (f"L{l} feed-forward", st["feed_forward"]),
                (f"L{l} add_2", st["add_2"])]
        X = st["add_2"]
    out += [("ln_f", m.top_rows(X)), ("scores", m.forward(ids)), ("shares", softmax(m.forward(ids)))]
    return out

def follow(m_old, m_new, ids=IDS):
    """For each step, the largest change in each seat's row."""
    a, b = steps(m_old, ids), steps(m_new, ids)
    table = []
    for (name, x), (_, y) in zip(a, b):
        d = np.abs(np.asarray(y) - np.asarray(x)).max(axis=1)
        table.append((name, d))
        print(f"  {name:<16}" + "".join(f"{v:9.4f}" for v in d))
    return table

print(f"  {'step':<16}" + "".join(f"{s + '@' + str(i):>9}" for i, s in enumerate(SEAT_WORDS)))
_ = follow(m, m)
print("  (no change: every entry is zero)")
# %% [markdown]
# ## Part 2, Breakage 1: one number in cat's row
# Cat's row (0, 1) becomes (0, 1.5). Nothing fails. Follow the change.
# %%
E1 = TOY_EMBED.copy(); E1[1, 1] = 1.5
m1 = hand_tiny(E=E1)
print(f"  {'step':<16}" + "".join(f"{s + '@' + str(i):>9}" for i, s in enumerate(SEAT_WORDS)))
t1 = follow(m, m1)
s1 = softmax(m1.forward(IDS))
for i in range(6):
    print(f"  after {SEAT_WORDS[i]}@{i}: top {WORDS[int(sh[i].argmax())]} -> {WORDS[int(s1[i].argmax())]}; "
          f"share of the true next word " + (f"{sh[i, IDS[i + 1]]:.3f} -> {s1[i, IDS[i + 1]]:.3f}" if i < 5 else "-"))
d0 = np.abs(m1.forward(IDS)[0] - S[0])
print("  seat 0's scores: the only column that moves is", [WORDS[k] for k in np.nonzero(d0 > 1e-12)[0]], f"(by {d0.max():.4f}): cat's own row is also an output row")
first = {i: next(name for name, d in t1 if d[i] > 0) if any(d[i] > 0 for _, d in t1) else None for i in range(6)}
print("  first step at which each seat changes:", first)
# %% [markdown]
# ## Part 2, Breakage 2: one number in a row the sentence never uses
# Hat's row (-2, 1) becomes (2, 1). Hat is not in the running sentence.
# %%
E2 = TOY_EMBED.copy(); E2[10, 0] = 2.0
m2 = hand_tiny(E=E2)
print(f"  {'step':<16}" + "".join(f"{s + '@' + str(i):>9}" for i, s in enumerate(SEAT_WORDS)))
t2 = follow(m, m2)
s2 = softmax(m2.forward(IDS))
print("  after mat@5 before:", [(WORDS[k], round(float(sh[5, k]), 3)) for k in np.argsort(-sh[5])[:3]])
print("  after mat@5 after: ", [(WORDS[k], round(float(s2[5, k]), 3)) for k in np.argsort(-s2[5])[:3]])
print("  hat's row now equals looked's row:", E2[10].tolist(), E2[8].tolist(),
      "| their shares are equal at every seat:", bool(np.allclose(s2[:, 10], s2[:, 8])))
# %% [markdown]
# ## Part 2, Breakage 3: one number in a row the sentence uses twice
# The's row (1, 0) becomes (1.1, 0). The appears at seats 0 and 4.
# %%
E3 = TOY_EMBED.copy(); E3[0, 0] = 1.1
m3 = hand_tiny(E=E3)
print(f"  {'step':<16}" + "".join(f"{s + '@' + str(i):>9}" for i, s in enumerate(SEAT_WORDS)))
t3 = follow(m, m3)
s3 = softmax(m3.forward(IDS))
print("  largest change in any share:", f"{np.abs(s3 - sh).max():.4f}", "| top words unchanged:", bool((s3.argmax(1) == sh.argmax(1)).all()))
nxt = IDS[1:]
loss = lambda s: float(np.mean([-math.log(s[i, nxt[i]]) for i in range(5)]))
print(f"  average surprise: hand Tiny {loss(sh):.4f}; breakage 1 {loss(s1):.4f}; breakage 2 {loss(s2):.4f}; breakage 3 {loss(s3):.4f}")
# %% [markdown]
# ## Part 3: the checkpoint
# %%
tests = []
tests.append(("the hand Tiny holds 112 learned numbers", m.n_params == 112))
tests.append(("Tiny's code matches the pencil trace to 1e-4 at every step",
              max(float(np.abs(np.asarray(y) - np.asarray(x)).max()) for (_, x), (_, y) in zip(
                  steps(m, IDS), [("x", pen["x0"])] + sum([[("a", pen[f"layer{l}"][k]) for k in ("ln_1", "shares", "att", "add_1", "ln_2", "ff", "add_2")] for l in range(2)], [])
                  + [("f", pen["ln_f"]), ("s", pen["out_scores"]), ("p", pen["out_shares"])])) < 1e-4))
tests.append(("after mat@5 the top word is 'looked' with share 0.365", WORDS[int(sh[5].argmax())] == "looked" and round(float(sh[5].max()), 3) == 0.365))
tests.append(("breakage 1: seat 0's rows never change up to ln_f (the mask); its scores move only in cat's column (the tie)",
              all(d[0] == 0 for name, d in t1 if name not in ("scores", "shares")) and list(np.nonzero(d0 > 1e-12)[0]) == [1]))
tests.append(("breakage 1: seats 2 to 5 first change at layer 0's shares", all(first[i] == "L0 shares" for i in range(2, 6))))
tests.append(("breakage 2: every row up to ln_f is unchanged, and the shares change", all(d.max() == 0 for name, d in t2 if name not in ("scores", "shares")) and np.abs(s2 - sh).max() > 0))
tests.append(("breakage 3: seats 0 and 4 change in the rows that enter layer 0", t3[0][1][0] > 0 and t3[0][1][4] > 0 and t3[0][1][[1, 2, 3, 5]].max() == 0))
for k, (name, ok) in enumerate(tests, 1):
    print(f"  {k}. {name}: {'pass' if ok else 'FAIL'}")
print("all checkpoint tests pass:", all(ok for _, ok in tests))
