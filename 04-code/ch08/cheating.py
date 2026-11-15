"""Chapter 8: what the causal mask prevents.  (1) Leakage on the running sentence: change the
last word and count which output rows change.  (2) A hand-built head that reads one place ahead,
used to 'predict' the next token of the novel with and without the mask.
Run from 04-code/ch08:  python3 cheating.py"""
from _common import *
from tiny.attention import attention, softmax, causal_mask
from tiny.corpus import load_text
from tiny.tokenizer import load_bpe

# (1) leakage
X2 = X.copy(); X2[5] = [-2, 1]                 # 'the cat sat on the hat': hat's toy row is (-2, 1)
for causal in (True, False):
    a, _ = attention(X, W_q, W_k, W_v, causal=causal)
    b, _ = attention(X2, W_q, W_k, W_v, causal=causal)
    changed = [WORDS[i] if i != 5 else "mat/hat" for i in range(6) if not np.allclose(a[i], b[i])]
    print(f"mask {'on ' if causal else 'off'}: rows whose output changes when 'mat' becomes 'hat': {len(changed)} of 6 {changed}")
a, _ = attention(X, W_q, W_k, W_v, causal=False); b, _ = attention(X2, W_q, W_k, W_v, causal=False)
print("  without the mask, the first 'the' moves from", tuple(round(float(v), 3) for v in a[0]), "to", tuple(round(float(v), 3) for v in b[0]))

# (2) the look-ahead head on the novel
tok = load_bpe(TINY_BPE)
ids = np.array(tok.encode(load_text()))
n, V = 64, len(tok)
windows = len(ids) // n
P = np.eye(n)                                  # keys: which seat a token sits in
ahead = np.roll(np.eye(n), 1, axis=1) * 20.0   # query of seat i points at seat i+1, strongly
right = {True: 0, False: 0}; total = 0
for w in range(windows):
    seq = ids[w * n:(w + 1) * n]
    values = np.eye(V)[seq]                    # value: a row with a 1 in the place of its own id
    for causal in (True, False):
        scores = ahead @ P.T
        if causal:
            scores = np.where(causal_mask(n), scores, -np.inf)
        guess = (softmax(scores) @ values).argmax(axis=1)
        right[causal] += int((guess[:-1] == seq[1:]).sum())
    total += n - 1
print(f"\nthe novel in {windows} windows of {n} Tiny tokens; {total:,} next-token guesses")
print(f"  look-ahead head, mask off: {right[False]:,} right ({right[False] / total:.1%})")
print(f"  same head, mask on:        {right[True]:,} right ({right[True] / total:.1%})")
same = int(sum((ids[w*n:(w+1)*n][:-1] == ids[w*n:(w+1)*n][1:]).sum() for w in range(windows)))
print(f"  (next token equal to the current one, for comparison: {same:,}, {same / total:.1%})")
