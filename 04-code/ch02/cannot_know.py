"""What a tokenizer cannot know.  Run from 04-code/ch02:  python3 cannot_know.py"""
import _common
import tiktoken

gpt2 = tiktoken.get_encoding("gpt2")
def show(texts):
    for t in texts:
        ids = gpt2.encode(t)
        print(f"  {t!r:<12} -> {ids!s:<22} {' | '.join(gpt2.decode([i]).replace(' ', '␣') for i in ids)}")
print("The same word, five ids:"); show(["cat", " cat", "Cat", " Cat", " CAT"])
print("\nThe same word in two places of the running sentence:"); show(["the", " the"])
print("\nNumbers are cut wherever the merges happened to fall:"); show(["1865", " 1865", " 1866", " 12345", " 3.14159"])
print("\nNear-identical words, unrelated ids:"); show([" rabbit", " rabbits", " Rabbit", " rabbi"])
