"""Chapter 7: numbers for the Check Yourself box and the exercises (Appendix A).
Run from 04-code/ch07:  python3 solutions.py   (add --measure to retime exercise 11)"""
import _common
from _common import SENTENCE, TINY_BPE, recorded, prompt
import math, time
import numpy as np
from tiny.corpus import load_text
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import TOY_EMBED, TOY_TURN, Embedding, apply_rotary, agreement_table
from tiny.linalg import agreement_table_fast, matmul_tiled, intensity, work, check_shapes

R = {w: TOY_EMBED[TOY_VOCAB.index(w)] for w in TOY_VOCAB}
X = np.array([R[w] for w in SENTENCE.split()])
print("CY1:", check_shapes((6, 2), (2, 3)))
print("CY2:", (np.array([R["the"], R["mat"]]) @ np.array([R["sat"], R["on"]]).T).tolist())
print("CY3:", check_shapes((7, 128), (128, 7)), check_shapes((128, 7), (7, 128)))

print("1:", (np.array([R["sat"], R["on"]]) @ np.array([R["the"], R["cat"], R["mat"]]).T).tolist())
for a, b in [((3, 4), (4, 5)), ((5, 4), (3, 4)), ((1, 128), (128, 2048)), ((2048, 128), (1, 128))]:
    try: print("2:", a, b, "->", check_shapes(a, b))
    except ValueError as e: print("2:", e)
print("3:", (X.T @ X).tolist())
V = TOY_EMBED
print("4:", (V @ V.T).shape, work(12, 2, 12))
def turn_matrix(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s], [-s, c]])
R90 = turn_matrix(math.radians(90))
print("5:", np.round(R90, 3).tolist(), np.round(R90 @ np.array([1.0, 0.0]), 3).tolist(),
      np.round(R90 @ R90, 3).tolist())
n = 1024 * 1024
print(f"6: {n:,} numbers; {n * 8 / 2**20:.0f} MiB at 64 bits, {n * 2 / 2**20:.0f} MiB at 16 bits; "
      f"x 12 heads x 12 layers at 16 bits: {n * 2 * 144 / 2**20:.0f} MiB")
h = 4096
b = 295 * h / (h - 2 * 295)
print(f"7: {intensity(1, h, h):.1f} and {intensity(64, h, h):.1f}; break-even rows on H100 SXM ~ {b:.1f} "
      f"(check {intensity(math.ceil(b), h, h):.1f})")

tok = load_bpe(TINY_BPE)
emb = Embedding(TINY_VOCAB_SIZE, seed=1865)
P = emb(tok.encode(prompt))
TP = agreement_table_fast(P)
print("8:", tok.encode(prompt), TP.shape, f"{TP.sum():.4f}", bool(np.allclose(TP, agreement_table(P))))
C = X - X.mean(axis=0)
print("9:", np.round(X.mean(axis=0), 3).tolist(), np.round(C, 3).tolist(), np.round(C.sum(axis=0), 12).tolist())
pair = np.stack([apply_rotary(X, range(6), speeds=TOY_TURN), apply_rotary(X, range(100, 106), speeds=TOY_TURN)])
tabs = pair @ pair.transpose(0, 2, 1)
print("10:", pair.shape, "->", tabs.shape, f"largest difference {np.abs(tabs[0] - tabs[1]).max():.1e}")

def measure():
    out = {}
    for n in [256, 512, 1024, 2048]:
        W = np.random.default_rng(1865).normal(0, 0.02, size=(n, 128)); best = 1e9
        for _ in range(20):
            t0 = time.perf_counter(); W @ W.T; best = min(best, time.perf_counter() - t0)
        out[str(n)] = best
    return out
t = recorded("solutions", measure)
print("11:", ", ".join(f"{n}: {t[n] * 1000:.2f} ms" for n in ["256", "512", "1024", "2048"]),
      f"(measured {t['measured']})")
wrong = (X * X).sum(axis=0)
print("12:", wrong.shape, wrong.tolist(), (X / wrong).shape, np.round((X / wrong)[2], 3).tolist())
W = emb(tok.encode(load_text())[:1024])
print("13:", bool(np.allclose(matmul_tiled(W, W.T, 100), W @ W.T)))
