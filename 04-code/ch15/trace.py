"""Chapter 15, sections 1 to 6: the running sentence through the hand-sized Tiny, every step printed.
The numbers come from pencil.py (plain Python, no eps, full precision, shown to three places).
Run from 04-code/ch15:  python3 trace.py > outputs/trace.txt"""
from _common import *
import math
from hand import WORDS, IDS, SEAT_WORDS, W_q, W_k, W_v, W_o, W_1, b_1, W_2, b_2
from tiny.embed import TOY_EMBED, TOY_POS
import pencil

TAB = {k: v.tolist() for k, v in dict(W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, W_1=W_1, b_1=b_1, W_2=W_2, b_2=b_2).items()}
E, P = TOY_EMBED.tolist(), TOY_POS.tolist()
T = pencil.trace(E, P, IDS, TAB)
f3 = lambda v: "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
seat = lambda i: f"{SEAT_WORDS[i]}@{i}"

print("THE PLAN: the hand-sized Tiny")
print("  vocabulary: the toy vocabulary of twelve entries:", WORDS)
print("  running sentence ids:", IDS, "->", SEAT_WORDS)
print("  width 2, one head of width 2, feed-forward 2 -> 3 -> 2 with ReLU, RMSNorm (gain 1), two layers, tied output")
counts = {"token table": 24, "position table": 12, "per layer: ln_1 gain": 2, "per layer: W_q, W_k, W_v, W_o": 16,
          "per layer: ln_2 gain": 2, "per layer: W_1, b_1, W_2, b_2": 17, "ln_f gain": 2}
total = 24 + 12 + 2 * (2 + 16 + 2 + 17) + 2
print("  learned numbers:", ", ".join(f"{k} {v}" for k, v in counts.items()), f"-> total {total}")

print("\nSTEP 1  tokens to rows: token row + seat row")
for i in range(6):
    print(f"  seat {i} {SEAT_WORDS[i]:>3} id {IDS[i]:>2}: token {f3(T['tok'][i])} + seat {f3(T['pos'][i])} = {f3(T['x0'][i])}")

for l in range(2):
    L = T[f"layer{l}"]
    X = T["x0"] if l == 0 else T["layer0"]["add_2"]
    print(f"\n===== LAYER {l} =====")
    print(f"STEP {2 + 6 * l}a  RMSNorm (ln_1): divide each row by the root of the average of its two squares")
    for i in range(6):
        a, b = X[i]
        print(f"  {seat(i):>7}: {f3(X[i])}  rms sqrt(({a:.3f}^2 + {b:.3f}^2) / 2) = {L['rms_1'][i]:.3f}  -> {f3(L['ln_1'][i])}")
    print(f"STEP {2 + 6 * l}b  query, key and value rows (Chapter 8's tables)")
    for i in range(6):
        print(f"  {seat(i):>7}: query {f3(L['q'][i])}  key {f3(L['k'][i])}  value {f3(L['v'][i])}")
    print(f"STEP {3 + 6 * l}  scores (query . key), divided by sqrt(2), masked, softmax")
    print("           " + "".join(f"{seat(j):>9}" for j in range(6)))
    for i in range(6):
        print(f"  {seat(i):>7}: " + "".join(f"{L['scaled'][i][j]:9.3f}" if j <= i else f"{'masked':>9}" for j in range(6)))
    print("  shares:")
    for i in range(6):
        print(f"  {seat(i):>7}: " + "".join(f"{L['shares'][i][j]:9.3f}" if j <= i else f"{'':>9}" for j in range(6)) + f"   sum {sum(L['shares'][i]):.3f}")
    s = 2
    print(f"  sat in detail: scores " + ", ".join(f"{L['scores'][s][j]:.3f}" for j in range(3)) + " -> / 1.414 -> "
          + ", ".join(f"{L['scaled'][s][j]:.3f}" for j in range(3)) + " -> e^ " + ", ".join(f"{math.exp(L['scaled'][s][j]):.3f}" for j in range(3))
          + f" (total {sum(math.exp(L['scaled'][s][j]) for j in range(3)):.3f}) -> shares " + ", ".join(f"{L['shares'][s][j]:.3f}" for j in range(3)))
    print(f"STEP {4 + 6 * l}  blend of values (shares x values), then W_o (the identity) and the first addition")
    for i in range(6):
        print(f"  {seat(i):>7}: blend {f3(L['blend'][i])}  row {f3(X[i])} + blend = {f3(L['add_1'][i])}")
    print(f"STEP {5 + 6 * l}  RMSNorm (ln_2), the feed-forward block, the second addition")
    for i in range(6):
        print(f"  {seat(i):>7}: {f3(L['add_1'][i])} -> ln_2 {f3(L['ln_2'][i])} | matches {f3(L['match'][i])} | ReLU {f3(L['active'][i])} | writes {f3(L['ff'][i])} | row out {f3(L['add_2'][i])}")
    fired = [seat(i) for i in range(6) if L['active'][i][2] > 0]
    print(f"  neuron 3 (key (1, 1), bias -2) fires for: {fired if fired else 'nobody'}")
    print("  neurons active per seat:", [sum(1 for h in L['active'][i] if h > 0) for i in range(6)])

print("\n===== OUTPUT =====")
print("STEP 14  final RMSNorm (ln_f) on the top rows")
for i in range(6):
    print(f"  {seat(i):>7}: top {f3(T['top'][i])}  rms {T['rms_f'][i]:.3f} -> {f3(T['ln_f'][i])}")
print("STEP 15  scores against all twelve rows of the token table (tied output), then shares")
nexts = IDS[1:] + [None]
loss_terms = []
for i in range(6):
    sc, sh = T["out_scores"][i], T["out_shares"][i]
    order = sorted(range(12), key=lambda k: -sh[k])
    top3 = ", ".join(f"{WORDS[k]} {sh[k]:.3f}" for k in order[:3])
    line = f"  after {seat(i):>7}: top three {top3}"
    if nexts[i] is not None:
        k = nexts[i]; rank = order.index(k) + 1
        loss_terms.append(-math.log(sh[k]))
        line += f" | true next '{WORDS[k]}' share {sh[k]:.3f}, rank {rank} of 12, surprise {-math.log(sh[k]):.3f}"
    print(line)
i = 5
print("  after mat in full (word: score, share):")
print("   " + "; ".join(f"{WORDS[k]} {T['out_scores'][i][k]:+.3f} {T['out_shares'][i][k]:.3f}" for k in range(12)))
print(f"  average surprise over the five predictions with a true next word: {sum(loss_terms) / 5:.3f}"
      f" (an even split over twelve gives ln 12 = {math.log(12):.3f})")
print("  shares sum to one at every seat:", all(abs(sum(r) - 1) < 1e-12 for r in T["out_shares"]))

# the counting of arithmetic for 'the plan'
mult = add = expn = roots = 0
n, d = 6, 2
emb_add = n * d
add += emb_add
per_layer = {}
norm_ops = n * (d + 1)               # squares and one divide per number, roughly: count squares and divides
mm = 3 * n * d * d                   # q, k, v: n rows x 2 x 2 multiplies
pairs = n * (n + 1) // 2             # unmasked pairs
score_mult = pairs * d
blend_mult = pairs * d
wo_mult = n * d * d
ff_mult = n * (d * 3 + 3 * d)
per_mult = mm + score_mult + blend_mult + wo_mult + ff_mult
print("\ncounting the pencil work (multiplications only, per layer):")
print(f"  q, k, v {mm}; scores {score_mult} (only the {pairs} unmasked pairs); blends {blend_mult}; W_o {wo_mult}; feed-forward {ff_mult}; total {per_mult}")
print(f"  exponentials per layer: {pairs}; square roots per layer: {2 * n}")
out_mult = n * 12 * d
print(f"  two layers {2 * per_mult} multiplications + output scores {out_mult} = {2 * per_mult + out_mult}; exponentials {2 * pairs + n * 12}; square roots {2 * 2 * n + n}")

save_json("trace", {"x0": T["x0"], "layer0": T["layer0"], "layer1": T["layer1"], "top": T["top"], "ln_f": T["ln_f"],
                    "out_scores": T["out_scores"], "out_shares": T["out_shares"], "loss": sum(loss_terms) / 5,
                    "learned_numbers": total})
