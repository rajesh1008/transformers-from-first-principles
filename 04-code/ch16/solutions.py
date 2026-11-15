"""Worked answers for Chapter 16's exercises that have numbers (Exercises 1 to 8)."""
import time
from _common import load_config, c, SEED   # first: one thread
import numpy as np
from reader import spec, tiny_spec, count, total, kv_numbers_per_token
from tiny.model import Tiny

print("Exercise 1: a GPT-2-family config with n_embd 1024, n_layer 24, n_head 16, vocab 50,257, n_positions 1,024")
g = spec(dict(model_type="gpt2", n_embd=1024, n_layer=24, n_head=16, vocab_size=50257, n_positions=1024))
for part, f, n in count(g):
    print(f"  {part:26s} {f:44s} {c(n):>13s}")
print(f"  total {c(total(g))}")

q = spec(load_config("Qwen3-0.6B")); sm = spec(load_config("SmolLM2-135M")); gp = spec(load_config("GPT-2 small"))
print("\nExercise 2: Qwen3-0.6B key-value cache at 4,096 tokens, 16-bit")
print(f"  {c(kv_numbers_per_token(q))} numbers x 2 bytes x 4,096 = {c(kv_numbers_per_token(q) * 2 * 4096)} bytes")

print("\nExercise 3: SmolLM2-135M weights at 4-bit (half a byte per number, scales left out)")
print(f"  {c(total(sm))} x 0.5 = {c(total(sm) // 2)} bytes")

print("\nExercise 4: feed-forward to attention ratio (weight tables only, per layer)")
for name, s in [("GPT-2 small", gp), ("SmolLM2-135M", sm), ("Qwen3-0.6B", q), ("Tiny", tiny_spec())]:
    att = s["d"] * s["heads"] * s["head_dim"] * 2 + 2 * s["d"] * s["kv_heads"] * s["head_dim"]
    ffn = (3 if s["gated"] else 2) * s["d"] * s["d_ff"]
    print(f"  {name:13s} {c(ffn)} / {c(att)} = {ffn / att:.2f}")

print("\nExercise 5: Tiny with d_ff 256 and RMSNorm")
s5 = tiny_spec(d_ff=256, norm="rms")
print(f"  hand {c(total(s5))}; Tiny(d_ff=256, norm='rms').n_params = {c(Tiny(d_ff=256, norm='rms').n_params)}")

print("\nExercise 6: Tiny untied")
s6 = tiny_spec(tied=False)
print(f"  hand {c(total(s6))}; Tiny(tied=False).n_params = {c(Tiny(tied=False).n_params)}")

print("\nExercise 7: Tiny forward time against tokens (one thread, fastest of five, 2026-11-10)")
m = Tiny(); rng = np.random.default_rng(SEED)
for n in (256, 512, 1024):
    ids = rng.integers(0, 2048, size=n); best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); m.forward(ids); best = min(best, time.perf_counter() - t0)
    lin = 2 * n * (12 * (4 * 128 * 128 + 2 * 128 * 512) + 128 * 2048)
    quad = 2 * 12 * 2 * n * n * 128
    print(f"  {n:5d} tokens: {best * 1e3:7.1f} ms; hand FLOPs {lin / 1e9:.2f} G proportional to n + {quad / 1e9:.2f} G proportional to n x n")

print("\nExercise 8: longest context for Qwen3-0.6B, batch 1, 16-bit, in 8 GB (weights + cache only)")
w = 2 * total(q); per = 2 * kv_numbers_per_token(q)
print(f"  (8,000,000,000 - {c(w)}) / {c(per)} = {c((8_000_000_000 - w) // per)} tokens; the config's limit is {c(q['context'])}")
