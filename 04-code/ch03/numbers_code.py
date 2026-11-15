"""Numbers and code through three vocabularies: Tiny's, GPT-2's and o200k.
Run from 04-code/ch03:  python3 numbers_code.py"""
import _common
from collections import Counter
import tiktoken
from tiny.tokenizer import load_bpe
import os

tiny = load_bpe(os.path.join(os.path.dirname(_common.__file__), "..", "tiny", "tiny_bpe_2048.txt"))
gpt2, o200k = tiktoken.get_encoding("gpt2"), tiktoken.get_encoding("o200k_base")
pieces = {"Tiny": lambda s: [tiny.vocab[i].decode("utf-8", "replace") for i in tiny.encode(s)],
          "GPT-2": lambda s: [gpt2.decode([i]) for i in gpt2.encode(s)],
          "o200k": lambda s: [o200k.decode([i]) for i in o200k.encode(s)]}
show = lambda ps: "|".join(p.replace(" ", "\u2423").replace("\n", "\u21b5") for p in ps)

# 1. How many tokens does each whole number from 0 to 9,999 need (written after a space)?
print("tokens needed for ' 0' to ' 9999' (the space included):")
for name, f in pieces.items():
    c = Counter(len(f(f" {n}")) for n in range(10000))
    print(f"  {name:<6}", "  ".join(f"{k} tokens: {c[k]:>5,}" for k in sorted(c)))

# 2. How each vocabulary cuts a few numbers
print("\nhow the numbers are cut:")
print(f"  {'text':<14} {'Tiny':<22} {'GPT-2':<16} {'o200k':<16}")
for s in [" 7", " 42", " 1865", " 1866", " 1871", " 2026", " 3141", " 12345", " 1000000", " 1,000,000"]:
    print(f"  {s!r:<14} {show(pieces['Tiny'](s)):<22} {show(pieces['GPT-2'](s)):<16} {show(pieces['o200k'](s)):<16}")

# 3. Where GPT-2 and o200k cut the four-digit numbers 1000 to 9999
def pattern(ps):                               # e.g. ' 1865' -> ['␣18', '65'] -> '2+2' (digits per token)
    return "+".join(str(n) for n in (sum(ch.isdigit() for ch in p) for p in ps) if n)
for name in ["GPT-2", "o200k"]:
    c = Counter(pattern(pieces[name](f" {n}")) for n in range(1000, 10000))
    print(f"\n{name}: how the 9,000 four-digit numbers are cut (digits in each token)")
    for p, k in c.most_common(6):
        print(f"  {p:<8} {k:>5,}")
    if len(c) > 6:
        print(f"  ({len(c) - 6} rarer patterns: {sum(k for p, k in c.most_common()[6:]):,} numbers)")

# 4. A sum, token by token
s = "1865 + 6 = 1871"
print(f"\n{s!r}:")
for name, f in pieces.items():
    print(f"  {name:<6} {show(f(s))}")

# 5. A few lines of Python
CODE = '''def count_pairs(words):
    counts = {}
    for w in words:
        for a, b in zip(w, w[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts
'''
print(f"\na six-line Python function, {len(CODE.encode())} bytes:")
for name, f in pieces.items():
    ps = f(CODE)
    blank = sum(1 for p in ps if p.strip() == "")
    print(f"  {name:<6} {len(ps):>3} tokens ({len(CODE.encode()) / len(ps):.2f} bytes per token); "
          f"{blank} of them only spaces or line breaks")
for name, f in pieces.items():
    print(f"  {name:<6} line 5 starts: {show(f(CODE.splitlines()[4] + chr(10))[:6])}")
