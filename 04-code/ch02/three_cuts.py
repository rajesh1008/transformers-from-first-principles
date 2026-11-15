"""Three ways to cut the running sentence.  Run from 04-code/ch02:  python3 three_cuts.py"""
import _common
import tiktoken
from _common import SENTENCE
from tiny.tokenizer import split_words

gpt2 = tiktoken.get_encoding("gpt2")               # comparison only
chars = list(SENTENCE)
words = split_words(SENTENCE)
pieces = [gpt2.decode([t]) for t in gpt2.encode(SENTENCE)]
print(f"characters ({len(chars)}):", " ".join("␣" if c == " " else c for c in chars))
print(f"words      ({len(words)}):", " | ".join(words))
print(f"GPT-2      ({len(pieces)}):", " | ".join(p.replace(" ", "␣") for p in pieces))
