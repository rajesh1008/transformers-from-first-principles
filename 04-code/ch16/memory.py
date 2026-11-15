"""Memory by hand from a configuration: the weights at four number formats, the key-value cache per
token and at full context, a training rule of thumb, and one measured check on Tiny (the peak memory
NumPy allocates during a forward pass over 1,024 tokens, against the hand sizes of its largest tables)."""
import tracemalloc
from _common import load_config, save_json, MODELS, c, SEED   # first: sets one thread
import numpy as np
from reader import spec, tiny_spec, total, kv_numbers_per_token
from tiny.model import Tiny

def gb(b): return f"{b / 1e9:,.3f} GB"
def mb(b): return f"{b / 1e6:,.1f} MB"
out = {}
specs = [("Tiny", tiny_spec())] + [(n, spec(load_config(n))) for n in MODELS]
print("Weights: learned numbers x bytes per number (64-bit 8, 32-bit 4, 16-bit 2, 8-bit 1, 4-bit 0.5)")
for name, s in specs:
    N = total(s)
    row = {bits: N * bits / 8 for bits in (64, 32, 16, 8, 4)}
    print(f"  {name:13s} {c(N):>12s}  " + "  ".join(f"{bits}-bit {mb(b):>10s}" for bits, b in row.items()))
    out[name] = dict(N=N, weights=row)

print("\nKey-value cache: 2 (key and value) x layers x key-value heads x head width numbers per token")
for name, s in specs:
    per = kv_numbers_per_token(s)
    b16 = per * 2
    full = b16 * s["context"]
    w16 = out[name]["N"] * 2
    print(f"  {name:13s} 2 x {s['layers']} x {s['kv_heads']} x {s['head_dim']} = {c(per):>6s} numbers per token;"
          f" 16-bit {c(b16):>7s} bytes; full context {c(s['context']):>6s} tokens: {mb(full):>9s}"
          f" (weights in 16-bit {mb(w16)}; cache/weights {full / w16:.2f})")
    if s["kv_heads"] != s["heads"]:
        print(f"  {'':13s} with one key-value head per query head it would be {c(per * s['heads'] // s['kv_heads'])} per token"
              f" ({s['heads'] // s['kv_heads']}x)")
    out[name].update(kv_per_token=per, kv_full_16bit=full)

print("\nTraining, rule of thumb (mixed precision with Adam, as counted in the ZeRO paper, 1910.02054):")
print("  16-bit weights 2 + 16-bit nudges 2 + 32-bit copy 4 + two 32-bit running averages 4 + 4 = 16 bytes per learned number,")
print("  before activations")
for name, s in specs:
    print(f"  {name:13s} {gb(16 * out[name]['N']):>12s}")
    out[name]["train_16B"] = 16 * out[name]["N"]

# Activations, from the 2022 activation-recomputation paper's per-layer count (2205.05198):
# s*b*h*(34 + 5*a*s/h) bytes per layer for 16-bit training without recomputation (s seats, b batch,
# h width, a heads).  For GPT-2 small at s = 1,024, b = 1:
s = specs[1 + list(MODELS).index("GPT-2 small")][1]
S, H, A = 1024, s["d"], s["heads"]
per_layer = S * H * (34 + 5 * A * S / H)
print(f"\nGPT-2 small activations (2205.05198 formula), 1,024 seats, batch 1: {S}x{H}x(34 + 5x{A}x{S}/{H}) = "
      f"{c(int(per_layer))} bytes per layer; x {s['layers']} layers = {gb(per_layer * s['layers'])}")
out["gpt2_activations_bytes"] = per_layer * s["layers"]

# ---- measured: Tiny's forward pass over 1,024 tokens, 64-bit numbers --------------------------------
m = Tiny()
ids = np.random.default_rng(SEED).integers(0, 2048, size=1024)
weights_bytes = sum(t.nbytes for t in m.tables().values())
m.forward(ids[:8])                      # warm up
tracemalloc.start()
S_out = m.forward(ids)
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
n, d, h, f, V = 1024, 128, 4, 512, 2048
hand = {"scores, one layer (4 heads x 1,024 x 1,024)": h * n * n * 8,
        "next-token scores (1,024 x 2,048)": n * V * 8,
        "feed-forward middle (1,024 x 512)": n * f * 8,
        "one row table (1,024 x 128)": n * d * 8}
print(f"\nTiny, measured (seed {SEED}, 64-bit, 1,024 random ids, 2026-11-10):")
print(f"  weights held by the model: {c(weights_bytes)} bytes = {c(m.n_params)} x 8")
print(f"  peak extra memory during forward(): {c(peak)} bytes ({mb(peak)})")
for k, v in hand.items():
    print(f"  hand size of {k:42s} {c(v):>12s} bytes")
out["tiny_measured"] = dict(weights_bytes=weights_bytes, forward_peak=peak, hand=hand)
save_json("memory", out)
