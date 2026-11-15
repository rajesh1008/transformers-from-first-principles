"""Chapter 1 Scoreboard: what the counting model stores and how fast it writes.
Timings are for the book's build machine (CPU only); as of November 2026.
Reference laptop in the book: CPU, 16 GB. Run from 04-code/ch01:  python3 scoreboard.py"""
import time, random
from bigram import pairs, next_word_shares, SEED

stored = sum(len(row) for row in pairs.values())          # one stored count per pair ever seen
random.seed(SEED)
n, out = 20_000, ["the"]
t0 = time.perf_counter()
for _ in range(n):
    shares = next_word_shares(out[-1]) or next_word_shares("the")
    out.append(random.choices(list(shares), weights=list(shares.values()))[0])
dt = time.perf_counter() - t0
print(f"seed = {SEED}")
print(f"stored numbers (pair counts): {stored:,}")
print(f"generation speed: about {round(n / dt, -2):,.0f} words per second")
