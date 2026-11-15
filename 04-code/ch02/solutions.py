"""Numerical answers for the Chapter 2 exercises.  Run from 04-code/ch02:  python3 solutions.py"""
import _common
from collections import Counter
import tiktoken
from tiny.corpus import load_text
from tiny.tokenizer import WordTokenizer, CharTokenizer, TOY_VOCAB

tok = WordTokenizer(TOY_VOCAB); text = load_text(); gpt2 = tiktoken.get_encoding("gpt2")
print("Ex 1:", tok.encode("a dog sat on a hat"), tok.encode("the rabbit looked at the cat"), repr(tok.decode([7, 2, 3, 5, 4])))
def coverage(tok, text):
    ids = tok.encode(text); unk = tok.stoi[tok.unk]
    return sum(i != unk for i in ids) / len(ids)
print(f"Ex 6: coverage {coverage(tok, text):.1%}")
cands = ["important", "although", "remember", "beautiful", "question", "everything", "together", "however",
         "probably", "different", "suddenly", "anything", "something", "morning", "children", "rabbit",
         "garden", "queen", "turtle", "curious", "dinner", "kitchen", "window", "pocket", "minute"]
found = [w for w in cands if len(gpt2.encode(" " + w)) == 1 and len(gpt2.encode(w)) > 1][:5]
print("Ex 7:", [(w, gpt2.encode(" " + w), [gpt2.decode([i]) for i in gpt2.encode(w)]) for w in found])
print("Ex 8: toy", len(tok.itos), "characters", len(CharTokenizer(text).itos))
single = [n for n in range(1, 2001) if len(gpt2.encode(f" {n}")) == 1]
first_two = next(n for n in range(1, 2001) if len(gpt2.encode(f" {n}")) > 1)
print(f"Ex 9: {len(single)} of 2,000 are one token with a leading space; the first needing two is {first_two}")
top = Counter(gpt2.encode(text, disallowed_special=())).most_common(10)
shown = [gpt2.decode_single_token_bytes(i) for i, _ in top]
print("Ex 10:", shown, "begin with a space:", sum(b.startswith(b" ") for b in shown))
print("       the curly quotes are three bytes each:", list("“".encode()), list("’".encode()))
print("Ex 11:", repr(" ".join(__import__("tiny.tokenizer", fromlist=["x"]).split_words("don't stop, Alice."))))
