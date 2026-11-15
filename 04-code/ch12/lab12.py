"""Lab 12: the hand block, three breakages that run without an error, and a checkpoint.
Part 1 repeats hand_ffn.py's block on the running sentence and reads mat's neurons.
Part 2 breaks it three ways.  Part 3 tests that catch each break.  Run from 04-code/ch12:  python3 lab12.py"""
from _common import *
from tiny.attention import attention, MultiHead
from tiny.ffn import relu, gelu, FeedForward
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe

out8, _ = attention(X, W_q, W_k, W_v)
R = X + out8
W_1 = np.array([[1, 0, 1], [0, 1, 1]], dtype=float); b_1 = np.array([0, 0, -2], dtype=float)
W_2 = np.array([[0, 0.5], [0.5, 0], [-0.5, 0.5]])

def block(Rx):
    return relu(Rx @ W_1 + b_1) @ W_2

F = block(R)
print("Part 1. The hand block (2 -> 3 -> 2, ReLU) on the running sentence after Chapter 8's head")
print(f"  mat: row {r3(R[5])} -> neuron activations {r3(relu(R[5] @ W_1 + b_1))} -> adds {r3(F[5])} -> new row {r3(R[5] + F[5])}")
print(f"  sat: row {r3(R[2])} -> neuron activations {r3(relu(R[2] @ W_1 + b_1))} -> adds {r3(F[2])}")

print("\nPart 2. Three breakages that run without an error")
nolin = (R @ W_1 + b_1) @ W_2
print("Breakage 1, the activation left out:")
print(f"  mat adds {r3(nolin[5])} instead of {r3(F[5])}; sat adds {r3(nolin[2])} instead of {r3(F[2])}")
a, b = R[2], R[5]
print(f"  additivity check f(sat + mat) - f(sat) - f(mat) + f(0): broken {r3(((np.array([a + b]) @ W_1 + b_1) @ W_2 - (np.array([a]) @ W_1 + b_1) @ W_2 - (np.array([b]) @ W_1 + b_1) @ W_2 + b_1 @ W_2)[0])}"
      f" | correct {r3((block(np.array([a + b])) - block(np.array([a])) - block(np.array([b])) + block(np.zeros((1, 2))))[0])}")
late = relu(R @ W_1 @ W_2 + b_1 @ W_2)
print("Breakage 2, the activation applied after the down table instead of between the tables:")
print(f"  mat adds {r3(late[5])} instead of {r3(F[5])}; sat adds {r3(late[2])} instead of {r3(F[2])}")
print(f"  every output is now zero or positive: {bool((late >= 0).all())}; the correct block's outputs below zero: {int((F < 0).sum())} numbers")

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE); E, P = Embedding(len(tok)), PositionTable()
Xs = E(ids) + P(range(len(ids))); att, ff = MultiHead(n_heads=4), FeedForward()
A = Xs + att(Xs)
def leaky(Ax):
    """A 'batched' block that centres each neuron over the tokens of the sequence before the down table."""
    Hh = ff.hidden(Ax); return (Hh - Hh.mean(0, keepdims=True)) @ ff.W_2 + ff.b_2
ids2 = ids[:-1] + [ids[1]]                                   # the last token replaced by " cat"
X2 = E(ids2) + P(range(len(ids2))); A2 = X2 + att(X2)
d_ok = np.abs(ff(A)[:-1] - ff(A2)[:-1]).max()
d_bad = np.abs(leaky(A)[:-1] - leaky(A2)[:-1]).max()
print("Breakage 3, each neuron centred over the sequence (a well-meant batch statistic):")
print(f"  last token changed ({tok.decode([ids[-1]])!r} -> {tok.decode([ids2[-1]])!r}): earlier rows' block output changes by {d_bad:.1e} (correct block: {d_ok:.1e})")
print(f"  the shapes and sizes look normal: output {leaky(A).shape}, mean length {np.linalg.norm(leaky(A), axis=1).mean():.5f} against {np.linalg.norm(ff(A), axis=1).mean():.5f}")

print("\nPart 3. Checkpoint tests")
tests = []
tests.append(("1. hand block: mat adds (0.352, 0)", np.allclose(F[5], [0.3515, 0], atol=1e-3)))
nl = lambda Rx: (Rx @ W_1 + b_1) @ W_2
add = lambda f: np.abs(f(np.array([a + b])) - f(np.array([a])) - f(np.array([b])) + f(np.zeros((1, 2)))).max()
tests.append(("2. correct block is not additive (catches Breakage 1)", add(block) > 1e-6))
tests.append(("3. Breakage 1 is additive, so test 2 fails on it", add(nl) < 1e-12))
tests.append(("4. some outputs below zero (catches Breakage 2)", (F < 0).any() and not (late < 0).any()))
tests.append(("5. changing the last token leaves earlier rows' block outputs unchanged (catches Breakage 3)", d_ok < 1e-15 and d_bad > 1e-9))
tests.append(("6. Tiny's block: hidden width 512, output width 128, 131,712 numbers", ff.hidden(A).shape[1] == 512 and ff(A).shape[1] == 128 and ff.n_params == 131712))
for name, ok in tests:
    print(f"  {name}: {'pass' if ok else 'FAIL'}")
print("all checkpoint tests pass:", all(ok for _, ok in tests))
