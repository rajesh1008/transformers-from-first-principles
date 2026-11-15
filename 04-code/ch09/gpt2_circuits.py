"""Chapter 9 cameo: GPT-2 small's heads working in pairs.  A stretch of 50 random tokens (seed 1865)
is shown twice after GPT-2's start token.  The first time nothing can be predicted; the second time
a model that can look up "what came after this token last time" predicts almost everything.
For every head we measure (a) its previous-token score, the average share each seat gives the seat
before it, and (b) its induction score, the average share a seat in the second copy gives to the
seat just after its own earlier copy.  Then we remove heads (Chapter 9's three ways) and measure the
second copy's loss.  Comparison only: GPT-2's numbers, not Tiny's.
Run from 04-code/ch09:  python3 gpt2_circuits.py   (about two minutes on the book's machine)"""
from _common import *
import time
from _gpt2run import run, token_losses, H, L, decoder

rng = np.random.default_rng(SEED)
T = 50
rand = rng.integers(1000, 50000, size=T).tolist()
ids = [50256] + rand + rand                          # <|endoftext|> then the same 50 tokens twice
dec = decoder()
print("first ten random tokens:", [dec(i) for i in rand[:10]])
t0 = time.perf_counter()
x, shares, cache = run(ids, keep_shares=True)
loss = token_losses(x, ids)
first, second = loss[1:T].mean(), loss[T + 1:2 * T].mean()
print(f"\nloss per token, first copy {first:.3f}, second copy {second:.3f}  (natural log; uniform guessing over 50,257 = {np.log(50257):.3f})")
S = np.stack(shares)                                 # (12 layers, 12 heads, 101, 101)
idx = np.arange(T + 1, 2 * T + 1)
induction = S[:, :, idx, idx - T + 1].mean(-1)       # share given to the seat after my earlier copy
prev = S[:, :, np.arange(1, len(ids)), np.arange(0, len(ids) - 1)].mean(-1)
order = np.dstack(np.unravel_index(np.argsort(-induction, axis=None), induction.shape))[0]
top_ind = [tuple(map(int, o)) for o in order[:5]]
print("\ntop five induction heads (layer.head: score):", ", ".join(f"{l}.{h}: {induction[l, h]:.3f}" for l, h in top_ind))
print(f"heads with induction score above 0.3: {(induction > 0.3).sum()} of 144")
po = np.dstack(np.unravel_index(np.argsort(-prev, axis=None), prev.shape))[0]
top_prev = [tuple(map(int, o)) for o in po[:6]]
print("top six previous-token heads:", ", ".join(f"{l}.{h}: {prev[l, h]:.3f}" for l, h in top_prev))
print(f"layer 4, head 11 (Chapter 8's previous-token head): previous-token score {prev[4, 11]:.3f}")
print(f"heads with previous-token score above 0.5: {(prev > 0.5).sum()} of 144")

def second_loss(remove, how="zero"):
    xr, _, _ = run(ids, remove=remove, how=how, cache=cache)
    return token_losses(xr, ids)[T + 1:2 * T].mean()

print("\nremoving heads, loss on the second copy (baseline", f"{second:.3f})")
for how in ("zero", "mean"):
    print(f"  the five induction heads, {how:>4}: {second_loss(top_ind, how):.3f}")
crng = np.random.default_rng(SEED + 1)
ctrl = []
for _ in range(20):
    pick = crng.choice(144, size=5, replace=False)
    ctrl.append(second_loss([(int(p) // 12, int(p) % 12) for p in pick]))
print(f"  five random heads, zero, 20 draws (seed {SEED + 1}): mean {np.mean(ctrl):.3f}, lowest {np.min(ctrl):.3f}, highest {np.max(ctrl):.3f}")
print(f"  BREAKAGE, the five induction heads removed the naive way (64 columns zeroed after the output table): {second_loss(top_ind, 'naive'):.3f}")

# heads in a circuit: remove early previous-token heads and watch the induction heads lose their pattern
early_prev = [(l, h) for l, h in top_prev if l < min(l2 for l2, _ in top_ind)][:3]
xr, sh2, _ = run(ids, remove=early_prev, keep_shares=True)
S2 = np.zeros_like(S); S2[L - len(sh2):] = np.stack(sh2)
ind2 = S2[:, :, idx, idx - T + 1].mean(-1)
print("\nremove the previous-token heads", ", ".join(f"{l}.{h}" for l, h in early_prev), "(zero):")
print("  induction scores before -> after:", ", ".join(f"{l}.{h}: {induction[l, h]:.3f} -> {ind2[l, h]:.3f}" for l, h in top_ind))
print(f"  loss on the second copy: {token_losses(xr, ids)[T + 1:2 * T].mean():.3f}")

# every head removed alone, the right way and the naive way
eff_zero = np.zeros((L, H)); eff_naive = np.zeros((L, H))
for l in range(L):
    for h in range(H):
        eff_zero[l, h] = second_loss([(l, h)]) - second
        eff_naive[l, h] = second_loss([(l, h)], "naive") - second
r = np.corrcoef(eff_zero.ravel(), eff_naive.ravel())[0, 1]
tz = set(np.argsort(-eff_zero.ravel())[:5].tolist()); tn = set(np.argsort(-eff_naive.ravel())[:5].tolist())
print("\nevery head removed alone (144 runs each way):")
print("  five most harmful, correct removal:", ", ".join(f"{i // 12}.{i % 12} (+{eff_zero.ravel()[i]:.3f})" for i in np.argsort(-eff_zero.ravel())[:5]))
print("  five most harmful, naive removal  :", ", ".join(f"{i // 12}.{i % 12} (+{eff_naive.ravel()[i]:.3f})" for i in np.argsort(-eff_naive.ravel())[:5]))
print(f"  agreement between the two ways: correlation {r:.2f}; heads in common among the top five: {len(tz & tn)}")
print(f"  heads whose correct removal changes the second-copy loss by less than 0.01: {(np.abs(eff_zero) < 0.01).sum()} of 144")
np.savez(os.path.join(HERE, "outputs", "gpt2_circuits.npz"), induction=induction, prev=prev, eff_zero=eff_zero, eff_naive=eff_naive,
         ind_after=ind2, S_ind=S[top_ind[0][0], top_ind[0][1]], S_prev=S[top_prev[0][0], top_prev[0][1]], loss=loss)
print(f"\n(run time {time.perf_counter() - t0:.0f} s on the book's machine, 10 November 2026)")
