"""Lab 3: train Tiny's vocabulary on the novel, save it, check it, then break it by merging too few times.
Run from 04-code/ch03:  python3 lab_bpe.py"""
import _common, os
from _common import SENTENCE, prompt
from tiny.corpus import load_text, load_heldout
from tiny.tokenizer import train_bpe, BPETokenizer, load_bpe, TINY_VOCAB_SIZE

text, sequel = load_text(), load_heldout()
merges = train_bpe(text, TINY_VOCAB_SIZE - 1)
tok = BPETokenizer(merges)
path = os.path.join(os.path.dirname(_common.__file__), "..", "tiny", "tiny_bpe_2048.txt")
tok.save(path)
tok = load_bpe(path)                                     # from now on, Tiny reads its vocabulary from this file
print(f"saved {len(merges):,} merges to tiny/tiny_bpe_2048.txt; reloaded vocabulary size {len(tok):,}")

ids = tok.encode(text)
print(f"novel: {len(text.encode()):,} bytes -> {len(ids):,} tokens; round trip exact: {tok.decode(ids) == text}")
ids = tok.encode(sequel)
print(f"sequel: {len(sequel.encode()):,} bytes -> {len(ids):,} tokens; round trip exact: {tok.decode(ids) == sequel}")
p = tok.encode(prompt)
print(f"prompt {prompt!r} -> {p} = {[tok.piece(i) for i in p]}")

print("\nBreakage: train with only 10 merges")
weak = BPETokenizer(train_bpe(text, 256 + 10))
w = weak.encode(SENTENCE)
print(f"  {SENTENCE!r} -> {len(w)} tokens: {'|'.join(weak.piece(i) for i in w)}")
n_weak, n_tiny = len(weak.encode(text)), len(tok.encode(text))
print(f"  novel: {n_weak:,} tokens instead of {n_tiny:,} ({n_weak / n_tiny:.2f} times as many)")
window = round(1024 * n_weak / n_tiny)                 # text that fills 1,024 of Tiny's tokens
print(f"  text that fills 1,024 Tiny tokens needs {window:,} of these; attention would compare "
      f"{window ** 2:,} pairs of positions instead of {1024 ** 2:,}")
print(f"  round trip still exact: {weak.decode(weak.encode(text)) == text}")

ids = tok.encode(SENTENCE)
print(f"\ncheckpoint: {SENTENCE!r} -> {ids}")
