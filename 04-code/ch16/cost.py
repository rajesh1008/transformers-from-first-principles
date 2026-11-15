"""Arithmetic by hand from a configuration, then a measured check on Tiny (one CPU thread, fastest of
five, 10 November 2026).  FLOP = floating-point operation; one multiply-add counts as two."""
import time
from _common import load_config, save_json, MODELS, c, SEED   # first: sets one thread
import numpy as np
from reader import spec, tiny_spec, total, non_embedding
from tiny.model import Tiny

def per_token_flops(s, n_ctx):
    """Forward FLOPs for one token that can see n_ctx earlier tokens (causal, so on average a seat in
    a pass of n tokens sees about n/2): 2 per learned number in the layers, plus scores and blend
    (2 x 2 x layers x n_ctx x query width), plus the output scores (2 x width x vocab)."""
    layers = 2 * non_embedding(s)
    attn = 2 * 2 * s["layers"] * n_ctx * s["heads"] * s["head_dim"]
    out = 2 * s["d"] * s["vocab"]
    return layers, attn, out

out = {}
print("Forward FLOPs per token, by hand (a token seeing 512 earlier tokens):")
for name, s in [("Tiny", tiny_spec())] + [(n, spec(load_config(n))) for n in MODELS]:
    a, b, o = per_token_flops(s, 512)
    t = a + b + o
    print(f"  {name:13s} layers {c(a):>13s} + attention {c(b):>11s} + output {c(o):>13s} = {c(t):>13s}"
          f"  ({t / total(s):.2f} per learned number)")
    out[name] = dict(layers=a, attention=b, output=o, total=t)

# ---- Tiny over 1,024 tokens: the exact multiply-adds of Tiny's code, which computes the full
# 1,024 x 1,024 score table and masks it afterwards (so attention costs n x n, not n x n / 2) -------
n, d, L, f, V = 1024, 128, 12, 512, 2048
madds = {"queries, keys, values, output table": L * 4 * n * d * d,
         "scores (full table, then masked)": L * n * n * d,
         "blend": L * n * n * d,
         "feed-forward": L * 2 * n * d * f,
         "next-token scores": n * d * V}
flops = 2 * sum(madds.values())
print(f"\nTiny, one forward pass over {n:,} tokens, multiply-adds by hand:")
for k, v in madds.items():
    print(f"  {k:38s} {c(v):>14s}")
print(f"  total FLOPs (x2)                       {c(flops):>14s}")

m = Tiny()
ids = np.random.default_rng(SEED).integers(0, 2048, size=n)
def fastest(fn, k=5):
    best = 1e9
    for _ in range(k):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best
t_tiny = fastest(lambda: m.forward(ids))
A = np.random.default_rng(SEED).normal(size=(1024, 1024)); B = np.random.default_rng(SEED + 1).normal(size=(1024, 1024))
t_mm = fastest(lambda: A @ B)
rate_mm = 2 * 1024 ** 3 / t_mm
rate_tiny = flops / t_tiny
print(f"\nMeasured, one thread, fastest of five (seed {SEED}, 2026-11-10):")
print(f"  plain 1,024 x 1,024 table product: {t_mm * 1e3:.1f} ms = {rate_mm / 1e9:.1f} GFLOP/s")
print(f"  Tiny forward over 1,024 tokens:    {t_tiny * 1e3:.1f} ms = {rate_tiny / 1e9:.1f} GFLOP/s"
      f" ({rate_tiny / rate_mm * 100:.0f} per cent of the table product's rate)")
print(f"  hand prediction at the table product's rate: {flops / rate_mm * 1e3:.1f} ms")

# ---- training cost, by the 6 x N x D rule (2 forward + 4 backward per learned number per token) ----
D_novel = 44360
N = m.n_params
train = 6 * N * D_novel
print(f"\nTraining Tiny once through the novel ({c(D_novel)} tokens), 6 x N x D: 6 x {c(N)} x {c(D_novel)} = {train:.3e} FLOPs")
print(f"  at Tiny's measured forward rate: {train / rate_tiny:.0f} s; at the table product's rate: {train / rate_mm:.0f} s"
      f"  (attention and the softmax left out of 6ND; Chapter 24 measures the real time)")
out["tiny_1024"] = dict(madds=madds, flops=flops, t_tiny_s=t_tiny, t_mm_s=t_mm, rate_mm=rate_mm,
                        rate_tiny=rate_tiny, train_novel_flops=train)

# ---- decoding is limited by reading the weights: bytes per token / memory rate -----------------------
copy_rate = 32.1e9    # Chapter 11's measured copy rate on the same machine (all cores), bytes per second
print(f"\nDecoding one token at a time reads every weight once per token. At Chapter 11's measured copy rate"
      f" ({copy_rate / 1e9:.1f} GB/s):")
for name in ["Tiny"] + list(MODELS):
    s = tiny_spec() if name == "Tiny" else spec(load_config(name))
    b16 = 2 * total(s)
    print(f"  {name:13s} 16-bit weights {b16 / 1e6:8.1f} MB -> at most {copy_rate / b16:8.0f} tokens per second")
save_json("cost", out)
