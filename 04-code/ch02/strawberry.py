"""Why "strawberry" is hard: the model sees pieces, not letters.
Run from 04-code/ch02:  python3 strawberry.py   (tiktoken is used for comparison only)"""
import _common
import tiktoken

for name in ["gpt2", "cl100k_base", "o200k_base"]:
    enc = tiktoken.get_encoding(name)
    for text in ["strawberry", " strawberry", "Strawberry", "STRAWBERRY"]:
        ids = enc.encode(text)
        pieces = " | ".join(enc.decode([i]).replace(" ", "␣") for i in ids)
        print(f"{name:<12} {text!r:<15} {len(ids)} token(s): {pieces:<28} ids {ids}")
    print()
word = "strawberry"
print(f"letters in {word!r}: {len(word)}; the letter r appears {word.count('r')} times")
enc = tiktoken.get_encoding("gpt2")
for i in enc.encode(word):
    p = enc.decode([i]); print(f"  piece {p!r:<8} id {i:>5}  contains {p.count('r')} r")

spelled = " ".join(word)                     # "s t r a w b e r r y"
ids = enc.encode(spelled)
print(f"\nspelled out, {spelled!r}: {len(ids)} tokens:",
      " | ".join(enc.decode([i]).replace(" ", "␣") for i in ids))
