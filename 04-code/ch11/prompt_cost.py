"""Chapter 11, Where This Bites: why a longer prompt costs more than linearly.  Counts the operations
GPT-2 small's shape does to read (prefill) a causal prompt of n tokens: the parts that grow with n (the
four attention tables and the feed-forward block, per token) and the part that grows with n squared (the
scores and the blend).  Pure counting.  Run from 04-code/ch11:  python3 prompt_cost.py"""
from _common import *
L, d, ff = 12, 768, 3072
per_token_linear = L * 2 * (4 * d * d + 2 * d * ff)            # Q, K, V, output tables and the feed-forward block
print("GPT-2 small's shape (12 layers, width 768, feed-forward 3,072), reading a causal prompt of n tokens")
print(f"operations per token that do not depend on n: {per_token_linear / 1e6:.1f} million")
print(f"{'n':>7} | {'linear part':>16} | {'attention part':>16} | attention share | per token (millions)")
for n in (1024, 4096, 8192, 16384, 65536, 131072):
    lin = per_token_linear * n
    att = L * 4 * d * n * (n + 1) / 2                      # QK^T and PV, causal: about n^2 / 2 pairs, 4 d each
    print(f"{n:>7,} | {lin:>16.3e} | {att:>16.3e} | {att / (lin + att):>14.1%} | {(lin + att) / n / 1e6:>8.1f}")
cross = per_token_linear / (L * 4 * d / 2)
print(f"attention equals the rest at about n = {cross:,.0f} tokens; doubling n beyond that nearly quadruples the attention part")
