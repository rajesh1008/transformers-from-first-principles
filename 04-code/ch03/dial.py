"""Vocabulary size as a dial.  Train on the novel at several sizes and measure the novel and
its sequel (never seen in training).  Run from 04-code/ch03:  python3 dial.py"""
import _common
from _common import SENTENCE
import tiktoken
from tiny.corpus import load_text, load_heldout
from tiny.tokenizer import train_bpe, BPETokenizer

text, sequel = load_text(), load_heldout()
nb, sb = len(text.encode("utf-8")), len(sequel.encode("utf-8"))
print(f"novel {nb:,} bytes; sequel {sb:,} bytes\n")
print(" size  merges | novel tokens  bytes/token | sequel tokens  bytes/token | the sentence")
rows = [(256, 2), (300, 2), (500, 2), (1000, 2), (2048, 2), (5000, 1), (50000, 1)]
for size, min_count in rows:
    m = train_bpe(text, size - 1, min_count=min_count)
    tok = BPETokenizer(m)
    n, s = len(tok.encode(text)), len(tok.encode(sequel))
    sent = "|".join(tok.piece(i) for i in tok.encode(SENTENCE))
    print(f"{len(tok):>5} {len(m):>7} | {n:>12,} {nb/n:>11.2f}  | {s:>13,} {sb/s:>11.2f}  | {sent}"
          + ("   (asked for 50,000: the novel ran out of pairs)" if size == 50000 else ""))
gpt2 = tiktoken.get_encoding("gpt2")
n, s = len(gpt2.encode(text)), len(gpt2.encode(sequel))
sent = "|".join(gpt2.decode([i]).replace(" ", "\u2423") for i in gpt2.encode(SENTENCE))
print(f"GPT-2 {gpt2.n_vocab:,} | {n:>12,} {nb/n:>11.2f}  | {s:>13,} {sb/s:>11.2f}  | {sent}")
