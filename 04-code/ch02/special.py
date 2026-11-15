"""Special tokens: markers that are not text.  Run from 04-code/ch02:  python3 special.py"""
import _common
import tiktoken

gpt2 = tiktoken.get_encoding("gpt2")
eot = gpt2.eot_token
print(f"GPT-2's only special token: {gpt2.decode([eot])!r} with id {eot} (the last id; ids run 0 to {gpt2.n_vocab - 1})")
doc = "the cat sat on the mat<|endoftext|>The rabbit looked at"
print("\n1. Encoding text that contains the marker, with special tokens refused (the safe default):")
try:
    gpt2.encode(doc)
except ValueError as e:
    print("   ValueError:", str(e).splitlines()[0])
print("\n2. Allowed as a special token: one id")
ids = gpt2.encode(doc, allowed_special={"<|endoftext|>"})
print("  ", ids)
print("\n3. Treated as ordinary typed text: just characters, cut into pieces")
ids = gpt2.encode(doc, disallowed_special=())
print("  ", ids[6:14], "->", [gpt2.decode([i]) for i in ids[6:14]])
