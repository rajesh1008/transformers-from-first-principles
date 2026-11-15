"""Chapter 9: a pair of heads doing one-step lookup (an induction pair), built by hand.
Rows are three blocks side by side: which token this is (one-hot), which seat it sits in (one-hot),
and an empty block that the first head fills with "which token came just before me".
Layer 1, the previous-token head: each seat's query points at the seat before it; its value is the
token, and the output table writes it into the empty block.  Layer 2, the lookup head: each token's
query is "my own token"; each key is "the token before me"; so a token finds the earlier places
that came right after another copy of itself, and its value hands over what sat there.
Second part: the same two heads, plus a second first-layer head that looks two seats back, on the
running sentence; then the pair on the first 1,024 tokens of the novel with Tiny's tokenizer.
Run from 04-code/ch09:  python3 induction_toy.py"""
from _common import *
from tiny.attention import attention, softmax, causal_mask

def build_rows(seq, V):
    n = len(seq); W = V + n + V + V                       # token | seat | before me | two before me
    R = np.zeros((n, W))
    R[np.arange(n), seq] = 1
    R[np.arange(n), V + np.arange(n)] = 1
    return R

def look_back(R, V, n, steps, gap=12.0):
    """A head whose query at seat i points at seat i - steps; returns the token it finds,
    written into block 'before me' (steps=1) or 'two before me' (steps=2)."""
    W = R.shape[1]
    c = gap * np.sqrt(n)                                  # after dividing by the square root of the width, the gap is 12
    Wq = np.zeros((W, n)); Wk = np.zeros((W, n)); Wv = np.zeros((W, V))
    for i in range(steps, n):
        Wq[V + i, i - steps] = c
    for j in range(n):
        Wk[V + j, j] = 1.0
    Wv[np.arange(V), np.arange(V)] = 1.0
    out, A = attention(R, Wq, Wk, Wv)
    Wo = np.zeros((V, W)); Wo[np.arange(V), V + n + (steps - 1) * V + np.arange(V)] = 1.0
    return out @ Wo, A

def lookup(R, V, n, use_two=False, gap=12.0):
    """Layer 2: query = my token (and, if use_two, the token before me); key = the token before
    that seat (and the token two before it).  Value = that seat's token."""
    W = R.shape[1]
    d = 2 * V if use_two else V
    c = gap * np.sqrt(d)
    Wq = np.zeros((W, d)); Wk = np.zeros((W, d)); Wv = np.zeros((W, V))
    Wq[np.arange(V), np.arange(V)] = c                   # my token ...
    Wk[V + n + np.arange(V), np.arange(V)] = 1.0         # ... against the token before each seat
    if use_two:
        Wq[V + n + np.arange(V), V + np.arange(V)] = c   # the token before me ...
        Wk[V + n + V + np.arange(V), V + np.arange(V)] = 1.0   # ... against the token two before each seat
    Wv[np.arange(V), np.arange(V)] = 1.0
    return attention(R, Wq, Wk, Wv)

vocab = ["<s>", "the", "cat", "sat", "on", "mat"]
seq = [0] + [1, 2, 3, 4, 1, 5] * 2                       # <s> the cat sat on the mat the cat sat on the mat
names = [vocab[t] for t in seq]
V, n = len(vocab), len(seq)
R0 = build_rows(seq, V)
w1, A1 = look_back(R0, V, n, 1)
R1 = R0 + w1                                             # the head's output is added to the rows
out, A2 = lookup(R1, V, n)
print("sequence:", " ".join(names))
print("layer 1 (previous-token head): share each seat gives to the seat before it:",
      r3([A1[i, i - 1] for i in range(1, n)]))
print("block 'before me' now holds, per seat:", [vocab[int(np.argmax(R1[i, V + n:V + n + V]))] for i in range(n)])

def report(out, A, label):
    print(f"\n{label}")
    print("  seat token  -> guess (share of the guess)   true next")
    right = 0; total = 0
    for i in range(7, n - 1):
        p = out[i] / out[i].sum()
        g = int(np.argmax(p)); top = np.sort(p)[::-1]
        tie = top[0] - top[1] < 1e-3
        guess = "tie: " + " / ".join(vocab[k] for k in np.where(np.abs(p - top[0]) < 1e-3)[0]) if tie else vocab[g]
        ok = (not tie) and g == seq[i + 1]
        right += ok; total += 1
        print(f"  {i:>4} {names[i]:<5} -> {guess:<16} ({top[0]:.3f})   {names[i + 1]}{'  right' if ok else ''}")
    print(f"  right on the second copy: {right} of {total}")
    return right, total

print("\nlayer 2 (lookup head), the row for seat 8 ('cat', second copy):", r3(A2[8]))
report(out, A2, "the pair: one token of context")
w2, _ = look_back(R1, V, n, 2)
R2 = R1 + w2
out2, A2b = lookup(R2, V, n, use_two=True)
report(out2, A2b, "with a second first-layer head looking two seats back: two tokens of context")
outB, AB = lookup(R0, V, n)
print("\nBREAKAGE: remove the previous-token head (block 'before me' stays empty)")
print("  the lookup head's row for seat 8 ('cat'):", r3(AB[8]))
report(outB, AB, "the lookup head alone")

# second setting: the pair on the novel, with Tiny's tokenizer
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text
tok = load_bpe(TINY_BPE)
ids = tok.encode(load_text())[:1024]
uniq = sorted(set(ids)); remap = {t: k for k, t in enumerate(uniq)}
s = [remap[t] for t in ids]; Vn, nn = len(uniq), len(s)
Rn = build_rows(s, Vn)
wn, _ = look_back(Rn, Vn, nn, 1)
Rn1 = Rn + wn
outn, An = lookup(Rn1, Vn, nn)
seen = [i for i in range(1, nn - 1) if any(s[j - 1] == s[i] for j in range(1, i + 1))]    # my token appeared before with a successor
right = sum(int(np.argmax(outn[i])) == s[i + 1] for i in seen)
# the same guess made by counting: the most frequent follower of my token so far (ties to the earliest)
def count_guess(i, k=1):
    c = {}
    for j in range(k, i + 1):
        if s[j - k:j] == s[i - k + 1:i + 1]:
            c[s[j]] = c.get(s[j], 0) + 1
    return max(c, key=lambda t: c[t]) if c else None
agree = sum(int(np.argmax(outn[i])) == count_guess(i) for i in seen)
wn2, _ = look_back(Rn1, Vn, nn, 2)
outn2, _ = lookup(Rn1 + wn2, Vn, nn, use_two=True)
right2 = sum(int(np.argmax(outn2[i])) == s[i + 1] for i in seen)
print(f"\nsecond setting: the first 1,024 tokens of the novel (Tiny's tokenizer, {Vn} different pieces)")
print(f"  seats whose token has appeared before with something after it: {len(seen)} of {nn - 2}")
print(f"  the pair's guess is the true next token at {right} of them ({100 * right / len(seen):.1f} per cent)")
print(f"  with the two-seats-back head as well: {right2} ({100 * right2 / len(seen):.1f} per cent)")
print(f"  the pair's guess equals 'the most frequent follower of this token so far' at {agree} of {len(seen)} seats")
