"""Chapter 3 Scoreboard.  Timings on the book's build machine (CPU only), as of November 2026.
Run from 04-code/ch03:  python3 scoreboard.py"""
import _common, time
from tiny.corpus import load_text
from tiny.tokenizer import train_bpe, BPETokenizer, TINY_VOCAB_SIZE
text = load_text()
t0 = time.perf_counter(); merges = train_bpe(text, TINY_VOCAB_SIZE - 1); t_train = time.perf_counter() - t0
tok = BPETokenizer(merges)
t0 = time.perf_counter(); n = len(tok.encode(text)); t_cold = time.perf_counter() - t0
t0 = time.perf_counter(); n = len(tok.encode(text)); t_warm = time.perf_counter() - t0
print(f"training {len(merges):,} merges on the novel: {t_train:.2f} s")
print(f"encoding the novel ({n:,} tokens), first time: {t_cold*1000:.0f} ms (about {round(n/t_cold, -4):,.0f} tokens per second)")
print(f"encoding it again, with every chunk cached: {t_warm*1000:.0f} ms (about {round(n/t_warm, -4):,.0f} tokens per second)")
print(f"vocabulary: {len(tok):,} ids; learned numbers (weights) in Tiny so far: 0")
