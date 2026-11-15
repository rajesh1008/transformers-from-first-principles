"""Chapter 9: heads that do nothing, measured.  GPT-2 small reads 256 tokens of the novel (the
opening of Chapter I); every one of its 144 heads is removed alone (set to zero before the output
table) and the change in loss recorded.  The least useful heads are then removed together, in
batches of 24, 48, 72 and 96, and the loss measured again on a *different* text, 256 tokens of the
opening of Through the Looking-Glass (Chapter 1), so that heads are not chosen and judged on the
same words.  The same numbers of heads chosen at random (seeds 1866 to 1870) are the control.
Both texts are public domain and probably in GPT-2's training data; this measures loss, not
memorisation.  Run from 04-code/ch09:  python3 gpt2_prune.py   (about six minutes)"""
from _common import *
import time
from _gpt2run import run, token_losses, H, L
from _bpe import encode
from tiny.corpus import load_text

def opening(path_text, marker, n=256):
    i = path_text.index(marker)
    ids = encode(path_text[i:i + 4000].replace("\r", ""))
    return [50256] + ids[:n - 1]

alice = load_text()
glass_raw = open(os.path.join(HERE, "..", "data", "looking_glass_gutenberg_raw.txt"), encoding="utf-8").read()
A_ids = opening(alice, "Alice was beginning")
G_ids = opening(glass_raw, "One thing was certain")
t0 = time.perf_counter()
xa, _, cache_a = run(A_ids); base_a = token_losses(xa, A_ids).mean()
xg, _, cache_g = run(G_ids); base_g = token_losses(xg, G_ids).mean()
print(f"selection text: Alice, Chapter I opening, {len(A_ids)} tokens; loss {base_a:.3f}")
print(f"test text: Through the Looking-Glass, Chapter 1 opening, {len(G_ids)} tokens; loss {base_g:.3f}")
eff = np.zeros((L, H))
for l in range(L):
    for h in range(H):
        xr, _, _ = run(A_ids, remove=[(l, h)], cache=cache_a)
        eff[l, h] = token_losses(xr, A_ids).mean() - base_a
flat = eff.ravel()
print("\neach head removed alone, change in loss on the selection text:")
print(f"  largest increases: " + ", ".join(f"{i // 12}.{i % 12} (+{flat[i]:.3f})" for i in np.argsort(-flat)[:5]))
print(f"  heads that change the loss by less than 0.01 either way: {(np.abs(flat) < 0.01).sum()} of 144")
print(f"  heads whose removal *lowers* the loss: {(flat < 0).sum()} (largest fall {flat.min():.3f})")
print("  per layer, heads changing the loss by less than 0.01:", [int((np.abs(eff[l]) < 0.01).sum()) for l in range(L)])
order = np.argsort(flat)                                      # least useful first
print("\nremoving the least useful heads together (chosen on Alice, judged on Looking-Glass):")
print("  heads removed   loss on Alice   loss on Looking-Glass   random heads on Looking-Glass (mean of 5, range)")
rows = []
for k in (24, 48, 72, 96):
    rem = [(int(i) // 12, int(i) % 12) for i in order[:k]]
    la = token_losses(run(A_ids, remove=rem, cache=cache_a)[0], A_ids).mean()
    lg = token_losses(run(G_ids, remove=rem, cache=cache_g)[0], G_ids).mean()
    rnd = []
    for s in range(5):
        pick = np.random.default_rng(SEED + 1 + s).choice(144, size=k, replace=False)
        rr = [(int(p) // 12, int(p) % 12) for p in pick]
        rnd.append(token_losses(run(G_ids, remove=rr, cache=cache_g)[0], G_ids).mean())
    rows.append((k, la, lg, np.mean(rnd), np.min(rnd), np.max(rnd)))
    print(f"  {k:>13}   {la:>13.3f}   {lg:>21.3f}   {np.mean(rnd):.3f} ({np.min(rnd):.3f} to {np.max(rnd):.3f})")
print(f"  {0:>13}   {base_a:>13.3f}   {base_g:>21.3f}   (no heads removed)")
np.savez(os.path.join(HERE, "outputs", "gpt2_prune.npz"), eff=eff, rows=np.array(rows), base=np.array([base_a, base_g]))
print(f"\n(run time {time.perf_counter() - t0:.0f} s on the book's machine, 10 November 2026)")
