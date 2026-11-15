"""How long is the novel, measured three ways?  Run from 04-code/ch02:  python3 corpus_lengths.py"""
import _common
import tiktoken
from tiny.corpus import load_text
from tiny.tokenizer import CharTokenizer, split_words

text = load_text()
gpt2 = tiktoken.get_encoding("gpt2")
chars = CharTokenizer(text)
words = split_words(text)
g = gpt2.encode(text, disallowed_special=())
print(f"UTF-8 bytes:      {len(text.encode('utf-8')):>8,}")
print(f"characters:       {len(text):>8,}   vocabulary {len(chars.itos):>6,} different characters")
print(f"words and marks:  {len(words):>8,}   vocabulary {len(set(words)):>6,} different words and marks")
print(f"GPT-2 tokens:     {len(g):>8,}   vocabulary {gpt2.n_vocab:>6,} (used here: {len(set(g)):,})")
print(f"characters per GPT-2 token: {len(text)/len(g):.2f}")
non_ascii = sorted(c for c in chars.itos if ord(c) > 127)
print(f"characters outside plain ASCII: {' '.join(non_ascii)}")
