"""Why nobody stores the table.  Run from 04-code/ch01:  python3 sizes.py"""
import tiktoken
enc = tiktoken.get_encoding("gpt2")
V = enc.n_vocab
print(f"GPT-2 vocabulary: {V:,} tokens")
print(f'"the cat sat on the mat" -> {enc.encode("the cat sat on the mat")}  ({len(enc.encode("the cat sat on the mat"))} tokens)')
print(f"pair grid (one previous token):    {V**2:,} cells")
print(f"triple grid (two previous tokens): {V**3:,} cells")
gpt2_small = 124_000_000
print(f"GPT-2 small learned numbers:       about {gpt2_small:,}")
print(f"the pair grid alone is {V**2 / gpt2_small:.0f} times larger than GPT-2 small")
