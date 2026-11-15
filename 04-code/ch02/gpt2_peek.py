"""What GPT-2's 50,257 entries look like.  Run from 04-code/ch02:  python3 gpt2_peek.py"""
import _common
import tiktoken
from _common import SENTENCE

gpt2 = tiktoken.get_encoding("gpt2")                # comparison only
ids = gpt2.encode(SENTENCE)
print("the running sentence in GPT-2 ids:", ids)
show = lambda i: repr(gpt2.decode_single_token_bytes(i).decode("utf-8", "replace"))
print("the first ten entries:  ", " ".join(show(i) for i in range(10)))
print("entries 256 to 265:     ", " ".join(show(i) for i in range(256, 266)))
print("entries 50,000 to 50,009:", " ".join(show(i) for i in range(50000, 50010)))
toks = [gpt2.decode_single_token_bytes(i) for i in range(gpt2.n_vocab - 1)]
space = sum(t.startswith(b" ") for t in toks)
print(f"entries that begin with a space: {space:,} of {len(toks):,} ({space/len(toks):.0%})")
longest = sorted(toks, key=len, reverse=True)[:3]
for t in longest:
    t = t.decode("utf-8", "replace")
    print(f"a longest entry, {len(t)} characters: {t[:24]!r}...")
