"""Chapter 2 Scoreboard. Timings on the book's build machine (CPU only), as of November 2026;
the reference laptop in the book is a CPU machine with 16 GB.  Run: python3 scoreboard.py"""
import _common, time
from tiny.corpus import load_text
from tiny.tokenizer import WordTokenizer, TOY_VOCAB, CharTokenizer
text = load_text()
tok, ch = WordTokenizer(TOY_VOCAB), CharTokenizer(text)
for name, enc in [("toy word tokenizer", tok.encode), ("character tokenizer", ch.encode)]:
    t0 = time.perf_counter(); n = len(enc(text)); dt = time.perf_counter() - t0
    print(f"{name:<20} {n:>7,} tokens in {dt*1000:5.0f} ms  (about {round(n/dt, -4):,.0f} tokens per second)")
print("learned numbers in Tiny so far: 0 (a tokenizer is a fixed table, not trained weights)")
