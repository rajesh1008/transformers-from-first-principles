"""Chapter 11, section 1: where the bytes go.  Counts, for one head of width 64 in 16-bit numbers, the
bytes a plain attention moves to and from main memory (write scores, read them for softmax, write shares,
read them for the blend) against a tiled version that reads Q, K and V and writes the output once per
query block, and the arithmetic each does.  Pure counting, no timing.  Run:  python3 traffic.py"""
from _common import *

d, B = 64, 2                       # head width, bytes per number (16-bit)
Br = 128                           # query rows per block for the tiled count
print("one head, width 64, 16-bit numbers, no causal saving (counts are for the full table)")
print(f"{'n':>7} | {'FLOPs':>15} | {'plain bytes':>15} | {'tiled bytes':>13} | {'plain FLOP/B':>12} | {'tiled FLOP/B':>12} | n x n table")
for n in (1024, 4096, 16384, 65536):
    flops = 4 * n * n * d                          # Q K^T and P V: 2 n^2 d each
    io_qkvo = 4 * n * d * B                        # read Q, K, V, write O once
    plain = io_qkvo + 4 * n * n * B                # write S, read S, write P, read P
    tiled = n * d * B * 2 + (n // Br) * 2 * n * d * B   # read Q and write O once; re-read K and V per query block
    print(f"{n:>7,} | {flops:>15,} | {plain:>15,} | {tiled:>13,} | {flops / plain:>12.1f} | {flops / tiled:>12.1f} | {n * n * B / 2**20:,.0f} MiB")
print("\nGPT-2 small, 12 heads x 12 layers, n = 1,024: score and share tables written and read per forward pass,")
print(f"  plain: {12 * 12 * 4 * 1024 * 1024 * B / 2**30:.2f} GiB of traffic for scores and shares alone;"
      f" Q, K, V and outputs: {12 * 12 * 4 * 1024 * 64 * B / 2**20:.1f} MiB")
