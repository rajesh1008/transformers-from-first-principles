"""Chapter 8: what one head costs, counted from the shapes.  Run from 04-code/ch08:  python3 cost.py"""
def head_cost(N, m, d):
    proj = 3 * 2 * N * m * d                 # Q, K, V: three (N, m) @ (m, d) products
    scores = 2 * N * N * d                   # Q @ K.T
    blend = 2 * N * N * d                    # shares @ V
    return proj, scores, blend
for name, N, m, d in (("running sentence, hand head", 6, 2, 2), ("Tiny, one full window", 1024, 128, 128),
                      ("GPT-2 small, one head, full window", 1024, 768, 64), ("GPT-2 small, one head, 8,192 tokens", 8192, 768, 64)):
    p, s, b = head_cost(N, m, d); tot = p + s + b
    print(f"{name}: N={N:,} m={m} d={d}")
    print(f"   making Q, K, V {p:>16,}  ({p / tot:.0%})   scores {s:>16,}  ({s / tot:.0%})   blend {b:>16,}  ({b / tot:.0%})")
    print(f"   share table {N * N:,} numbers = {N * N * 2 / 2**20:,.1f} MiB at 16 bits")
print("\nlength at which the two N x N steps cost as much as making Q, K and V (N = 1.5 x m):")
for m in (128, 768, 4096):
    print(f"   model width {m:>5}: N = {int(1.5 * m):,}")
