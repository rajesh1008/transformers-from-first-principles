"""Chapter 45: distributed training on paper.  Pure arithmetic, no model; every assumption is printed with its result.
A. Training memory per learned number with mixed precision and AdamW (2 bytes 16-bit weights + 2 bytes 16-bit gradients
   + 12 bytes of 32-bit master copy and two moments = 16), and how the ZeRO stages divide it over N data-parallel ranks.
B. The pipeline bubble: with p stages and m micro-batches in a GPipe-style schedule, idle share (p-1)/(m+p-1).
C. Bytes on the wire per step for data, tensor and pipeline parallelism, for Tiny and for an illustrative 70-billion-number
   dense model (80 layers, width 8,192), and the seconds they take at assumed link speeds."""
GB = 1e9
def zero(P, N):
    return {"plain DP": 16 * P, "ZeRO-1 (optimiser state)": (4 + 12 / N) * P, "ZeRO-2 (+ gradients)": (2 + 14 / N) * P, "ZeRO-3 / FSDP (+ weights)": 16 * P / N}
print("A. training memory, excluding activations (bytes)")
for name, P in (("Tiny", 2_766_592), ("illustrative 70B dense", 70e9)):
    for N in (1, 8, 64):
        z = zero(P, N); print(f"  {name:24s} N={N:3d}: " + "; ".join(f"{k} {v / GB:,.3f} GB" for k, v in z.items()))
print("B. pipeline bubble share (p-1)/(m+p-1)")
for p in (4, 8, 16):
    print(f"  p={p:2d}: " + "  ".join(f"m={m:2d} {(p - 1) / (m + p - 1):.3f}" for m in (1, 4, 8, 16, 32, 64)))
print("C. bytes per step and time at assumed link speeds")
P = 70e9; L, D = 80, 8192; tokens = 8 * 8192          # per data-parallel rank per step: 8 sequences of 8,192 tokens (assumed)
dp = 2 * (64 - 1) / 64 * P * 2                         # ring all-reduce of 16-bit gradients over 64 ranks, per rank
print(f"  DP over 64 ranks, all-reduce of 16-bit gradients: {dp / GB:,.1f} GB per rank per step; at 50 GB/s (assumed network link) {dp / 50e9:,.1f} s")
tp_one = tokens * D * 2                                # one all-reduce of the activations, 16-bit
tp = 4 * L * tp_one * 2 * (8 - 1) / 8                  # 2 all-reduces per layer forward, 2 backward, tensor parallel over 8
print(f"  TP over 8 ranks, 4 all-reduces per layer per step of {tokens:,} tokens: {tp / GB:,.1f} GB per rank; at 450 GB/s (assumed in-machine link) {tp / 450e9:,.2f} s; at 50 GB/s {tp / 50e9:,.1f} s")
pp = tokens * D * 2 * 2                                # one hand-off forward and one gradient back per stage boundary
print(f"  PP, per stage boundary per step (activations forward + gradients back, 16-bit): {pp / GB:,.2f} GB")
flops = 6 * P * tokens
print(f"  arithmetic per rank per step (6ND, before splitting): {flops:.3e}; at 400 TFLOP/s sustained (assumed) {flops / 400e12:,.0f} s")
P = 2_766_592; tok = 16 * 256
print(f"  Tiny, DP over 4 ranks: {2 * 3 / 4 * P * 4 / 1e6:,.2f} MB per rank per step (32-bit gradients); 6ND per step of {tok:,} tokens {6 * P * tok:.3e}")
