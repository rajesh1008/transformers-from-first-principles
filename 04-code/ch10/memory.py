"""Chapter 10, section 1: where the memory goes.  While a model writes one token at a time it keeps,
for every earlier token, every layer and every key-value head, one key and one value.  This script
counts those numbers per token for four variants of Tiny's layer and of GPT-2 small's, and then
builds the caches for real on the novel to check the arithmetic against the bytes NumPy allocates.
Run from 04-code/ch10:  python3 memory.py"""
from _common import *
from tiny.attention import MultiHead
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text

def per_token(layers, kv_heads, d_head):
    return layers * kv_heads * d_head * 2                 # a key and a value per key-value head per layer

print("numbers kept per token while writing (keys + values, all layers)")
print(f"{'model / variant':<44}{'kv heads':>9}{'per token':>11}{'bytes at 16 bits':>18}")
rows = [("Tiny, 1 layer, 4 heads of 32: multi-head", 1, 4, 32),
        ("Tiny, grouped-query, 2 key-value heads", 1, 2, 32),
        ("Tiny, multi-query, 1 key-value head", 1, 1, 32),
        ("GPT-2 small, 12 layers, 12 heads of 64", 12, 12, 64),
        ("GPT-2 small shape, grouped-query, 4 kv heads", 12, 4, 64),
        ("GPT-2 small shape, multi-query, 1 kv head", 12, 1, 64)]
for name, L, kv, d in rows:
    n = per_token(L, kv, d)
    print(f"{name:<44}{kv:>9}{n:>11,}{n * 2:>18,}")

print("\nthe whole cache for a 1,024-token window, GPT-2 small's shape, 16-bit numbers:")
for name, kv in (("multi-head (12 kv heads)", 12), ("grouped-query (4)", 4), ("multi-query (1)", 1)):
    print(f"  {name:<26} {per_token(12, kv, 64) * 1024 * 2 / 2**20:6.1f} MiB")
W = 256
print(f"  sliding window of {W}, multi-head: {per_token(12, 12, 64) * W * 2 / 2**20:6.1f} MiB, whatever the length")
for n in (1024, 8192, 65536):
    full = per_token(12, 12, 64) * n * 2 / 2**20
    print(f"  at {n:>6,} tokens: full cache {full:8.1f} MiB | window of {W}: {per_token(12, 12, 64) * min(n, W) * 2 / 2**20:5.1f} MiB")

# real caches: the keys and values Tiny's layer would keep for the first 1,024 tokens of the novel
tok = load_bpe(TINY_BPE)
ids = tok.encode(load_text())[:1024]
X = Embedding(len(tok))(ids) + PositionTable()(range(len(ids)))
print("\nreal caches for Tiny's layer on the first 1,024 tokens of the novel (float64, as NumPy stores them):")
for kv in (4, 2, 1):
    m = MultiHead(n_heads=4, n_kv_heads=kv)
    K, V = X @ m.W_k, X @ m.W_v
    print(f"  {kv} key-value head{'s' if kv > 1 else ' '}: keys {K.shape} + values {V.shape} = {K.nbytes + V.nbytes:>9,} bytes"
          f" = {(K.nbytes + V.nbytes) // 1024 // 8} numbers per token")
