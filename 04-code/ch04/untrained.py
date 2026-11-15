"""Where This Bites: rows that training never reached.  Tiny's unused rows, and GPT-2's rows that
sit closest to the table's average direction.  Run from 04-code/ch04:  python3 untrained.py"""
import _common, _gpt2
from _common import TINY_BPE
import numpy as np
import tiktoken
from collections import Counter
from tiny.corpus import load_text, load_heldout
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE

tok = load_bpe(TINY_BPE)
used = Counter(tok.encode(load_text()))
never = [i for i in range(TINY_VOCAB_SIZE) if i not in used]
print(f"Tiny: {len(never)} of {TINY_VOCAB_SIZE:,} ids never occur in the novel: "
      f"{sum(i < 256 for i in never)} single bytes, {sum(256 <= i < 2047 for i in never)} merged pieces, "
      f"{sum(i == 2047 for i in never)} special token")
seq = Counter(tok.encode(load_heldout()))
print(f"  the sequel uses {sum(seq[i] for i in never):,} tokens with those ids, "
      f"{len([i for i in never if seq[i]])} different ones; examples: "
      f"{[tok.piece(i) for i in never if seq[i]][:6]}")

enc = tiktoken.get_encoding("gpt2")
W = _gpt2.tensor("wte.weight").astype(float)
N = W / np.linalg.norm(W, axis=1, keepdims=True)
m = W.mean(0) / np.linalg.norm(W.mean(0))
c = N @ m
top = np.argsort(-c)[:100]
single = [i for i in top if len(enc.decode_single_token_bytes(int(i))) == 1]
print(f"\nGPT-2: of the 100 rows closest to the table's average direction, {len(single)} are single bytes "
      f"(control characters and fragments of characters)")
wordy = [enc.decode([int(i)]) for i in top if len(enc.decode_single_token_bytes(int(i))) > 4]
print(f"  the first two longer ones: {wordy[:2]}")
print(f"  cosine with the average direction: those 100 rows {c[top].mean():.2f} on average; "
      f"' cat' {c[3797]:.2f}; all rows {c.mean():.2f}")
text = load_text() + load_heldout()
counts = Counter(enc.encode(text))
print(f"  how often the 100 appear in the novel and its sequel together: {sum(counts[int(i)] for i in top)}")
