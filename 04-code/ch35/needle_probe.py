"""Chapter 35, 35.5: why SmolLM2 misses a key written near the end.  For one key at depths 0.5 and 1.0 in a 512-token haystack,
print the model's top choice and the true token's share at each answer position (teacher forcing).  CPU, one thread, 32-bit."""
import sys; sys.argv = [sys.argv[0], "smol", "512"]
exec(open("needle.py").read().split("R = {")[0])
for d in (0.5, 0.75, 1.0):
    k = keys[0]; ids = build([(d, k)]); A = E(f" {k}"); full = ids + A; lp = table(full)
    rows = []
    for j, p in enumerate(range(len(full) - len(A) - 1, len(full) - 1)):
        top = int(lp[p].argmax()); rows.append(f"{tk.decode([full[p + 1]])!r}: share {float(lp[p, full[p + 1]].exp()):.3f}, top {tk.decode([top])!r} {float(lp[p, top].exp()):.3f}")
    print(f"depth {d}, key {k}: " + " | ".join(rows))
    print("   text before the question:", repr(tk.decode(ids[-40:])))
