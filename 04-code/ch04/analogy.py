"""King minus man plus woman: arrow arithmetic on hand-made arrows, on GPT-2's table, and on
the embeddings learned from the novel.  Run from 04-code/ch04:  python3 analogy.py"""
import _common, _gpt2
import numpy as np
import tiktoken
from nudge import train, nearest

# 1. Four arrows drawn by hand: across = royalty, up = female
man, woman, king, queen = map(np.array, [(2, 1), (2, 3), (5, 1), (5, 3)])
print("by hand: king - man + woman =", (king - man + woman).tolist(), "= queen", queen.tolist())

# 2. GPT-2's own table (50,257 rows of 768 numbers)
enc = tiktoken.get_encoding("gpt2")
W = _gpt2.tensor("wte.weight")
N = W / np.linalg.norm(W, axis=1, keepdims=True)
tid = lambda s: enc.encode(s)[0]

def ranked(v, exclude=()):
    s = N @ (v / np.linalg.norm(v))
    order = [i for i in np.argsort(-s) if i not in exclude]
    return s, order

print("\nGPT-2: the five rows nearest to a - b + c (cosine)")
tests = [(" king", " man", " woman", " queen"), (" cats", " cat", " dog", " dogs"),
         (" Paris", " France", " Italy", " Rome"), (" walked", " walk", " swim", " swam")]
for a, b, c, want in tests:
    v = W[tid(a)] - W[tid(b)] + W[tid(c)]
    s, order = ranked(v)
    s2, order2 = ranked(v, exclude={tid(a), tid(b), tid(c)})
    show = lambda o: ", ".join(f"{enc.decode([i])!r} {s[i]:.2f}" for i in o[:5])
    r_all, r_ex = order.index(tid(want)) + 1, order2.index(tid(want)) + 1
    print(f"  {a!r} - {b!r} + {c!r}:  {show(order)}")
    print(f"     {want!r} is number {r_all} counting the three input words, number {r_ex} without them;"
          f" cosine({a!r}, {want!r}) alone = {N[tid(a)] @ N[tid(want)]:.2f}")

# 3. The 16-number embeddings learned from one novel
A, vocab, index = train(d=16, epochs=10, seed=1865)
v = A[index["king"]] - A[index["he"]] + A[index["she"]]
print("\nthe novel's 16-number table: king - he + she ->", nearest(A, vocab, index, v, 5))
print("  without the three input words ->", nearest(A, vocab, index, v, 5, exclude={"king", "he", "she"}))

# 4. Do the steps line up?  Compare the step man -> woman with other steps, by cosine
def step(a, b):
    return W[tid(b)] - W[tid(a)]
def cos(u, v):
    return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
base = step(" man", " woman")
print("\nGPT-2: cosine of the step ' man' -> ' woman' with other steps")
for a, b in [(" king", " queen"), (" he", " she"), (" boy", " girl"), (" father", " mother"), (" king", " dog"), (" cat", " mat")]:
    print(f"  {a!r} -> {b!r}: {cos(base, step(a, b)):.2f}")
