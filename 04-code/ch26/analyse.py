"""Chapter 26: every table in the chapter from the saved runs (outputs/seeds_*.json, the tables in outputs/ckpt/, Chapter 24's
and Chapter 25's 2,500-step tables).  One thread, 32-bit.
  "Carroll" and "Looking-Glass" in Parts 1 to 4 are the 16-window held-out samples every training script logs (Chapter 24's,
  2,048 predictions each), not Chapter 25's every-window scores.
  Part 1  repeats: fingerprints of three runs of one configuration; Chapter 24's 2,500-step run repeated from nothing
  Part 2  threads and the one-ulp nudge: first step whose batch loss differs from the one-thread run, and how the gap grows
  Part 3  the 3 x 3 grid of starting seeds and order seeds; five starting seeds alone, five order seeds alone
  Part 4  lr 1e-3 against 1.4e-3 on five seeds: single-seed verdicts, every pair and every three-seed subset
  Part 5  where two models disagree: per-token KL and top-choice agreement on the first 64 Looking-Glass windows by pool frequency, for pairs that
          differ by starting seed, by order seed, by thread count, by the nudge, and for Chapter 25's 2,500-step order pair
Writes outputs/analyse.json."""
import _common
import os, json, glob, itertools, math
import numpy as np
import torch
torch.set_num_threads(1)
from _common import HERE
from tiny.seed import fingerprint
from tiny.eval import load
O = {}
R = {}
for f in glob.glob(os.path.join(HERE, "outputs", "seeds_i*.json")):
    d = json.load(open(f)); R[d["tag"]] = d
def fin(tag, k=2):
    return R[tag]["final"][k]                      # 1 Carroll held-out, 2 Looking-Glass
print(f"{len(R)} runs of 400 steps found")
# Part 1
reps = ["i1865_o1865_t1", "i1865_o1865_t1_r2", "i1865_o1865_t1_r3"]
O["repeats"] = {t: dict(fp=R[t]["fingerprint"], carroll=fin(t, 1), glass=fin(t)) for t in reps}
print("Part 1. three runs of one configuration:", {t: R[t]["fingerprint"] for t in reps},
      "identical losses at every step:", R[reps[0]]["losses"] == R[reps[1]]["losses"] == R[reps[2]]["losses"])
a = torch.load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
p = os.path.join(HERE, "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
if os.path.exists(p):
    b = torch.load(p)
    same = all(torch.equal(a[k], b[k]) for k in a)
    O["repeat_ch24"] = dict(fp_ch24=fingerprint(a), fp_repeat=fingerprint(b), identical=same,
                            run=json.load(open(os.path.join(HERE, "outputs", "run_wsd_1e-3_2000.json")))["branches"]["2000"]["final"])
    print("  Chapter 24's 2,500-step run repeated:", O["repeat_ch24"])
# Part 2
base = R["i1865_o1865_t1"]["losses"]
O["drift"] = {}
for t in ("i1865_o1865_t2", "i1865_o1865_t4", "i1865_o1865_t4_r2", "i1865_o1865_t1_nudge"):
    if t not in R:
        continue
    l = R[t]["losses"]
    d = np.abs(np.array(l) - np.array(base))
    first = int(np.argmax(d > 0)) if (d > 0).any() else None
    marks = {s: float(d[s - 1]) for s in (1, 2, 5, 10, 25, 50, 100, 200, 400)}
    O["drift"][t] = dict(first_step=None if first is None else first + 1, at=marks, carroll=fin(t, 1), glass=fin(t),
                         d_carroll=fin(t, 1) - fin("i1865_o1865_t1", 1), d_glass=fin(t) - fin("i1865_o1865_t1"),
                         fp=R[t]["fingerprint"], seconds=R[t]["seconds"])
    print(f"Part 2. {t}: first differing step {O['drift'][t]['first_step']}; |batch-loss gap| at steps 1/10/50/100/400: "
          + " / ".join(f"{marks[s]:.2g}" for s in (1, 10, 50, 100, 400))
          + f"; final Carroll {fin(t, 1):.4f} ({O['drift'][t]['d_carroll']:+.4f}) Looking-Glass {fin(t):.4f} ({O['drift'][t]['d_glass']:+.4f}); {R[t]['seconds']:.0f} s")
if "i1865_o1865_t4" in R and "i1865_o1865_t4_r2" in R:
    O["t4_pair_identical"] = R["i1865_o1865_t4"]["losses"] == R["i1865_o1865_t4_r2"]["losses"]
    print("  two four-thread runs identical to each other:", O["t4_pair_identical"], "| fingerprints", R["i1865_o1865_t4"]["fingerprint"], R["i1865_o1865_t4_r2"]["fingerprint"])
# Part 3
S3 = (1865, 7, 11)
for k, nm in ((1, "carroll"), (2, "glass")):
    G = np.array([[fin(f"i{i}_o{o}_t1", k) for o in S3] for i in S3])
    gm = G.mean(); ri = G.mean(1) - gm; co = G.mean(0) - gm; res = G - gm - ri[:, None] - co[None, :]
    O[f"grid_{nm}"] = dict(table=G.tolist(), mean=gm, sd_all=float(G.std(ddof=1)), sd_init_means=float(G.mean(1).std(ddof=1)),
                           sd_order_means=float(G.mean(0).std(ddof=1)), sd_resid=float(np.sqrt((res ** 2).sum() / 4)),
                           ss_init=float(3 * (ri ** 2).sum()), ss_order=float(3 * (co ** 2).sum()), ss_resid=float((res ** 2).sum()))
    print(f"Part 3. {nm} grid (rows starting seed 1865/7/11, columns order seed 1865/7/11):")
    for i, row in zip(S3, G):
        print(f"    init {i:5d}: " + "  ".join(f"{v:.4f}" for v in row) + f"   row mean {row.mean():.4f}")
    print("    column means " + "  ".join(f"{v:.4f}" for v in G.mean(0)) + f";  all nine: mean {gm:.4f} sd {G.std(ddof=1):.4f}")
    g = O[f"grid_{nm}"]
    print(f"    sums of squares: starting seed {g['ss_init']:.4f}, order seed {g['ss_order']:.4f}, left over {g['ss_resid']:.4f}")
S5 = (1865, 7, 11, 21, 42)
for k, nm in ((1, "carroll"), (2, "glass")):
    ini = [fin(f"i{s}_o1865_t1", k) for s in S5]
    ordr = [fin(f"i1865_o{s}_t1", k) for s in S5]
    both = [fin(f"i{s}_o{s}_t1", k) for s in S5]
    O[f"five_{nm}"] = dict(init_only=ini, order_only=ordr, both=both, sd_init=float(np.std(ini, ddof=1)),
                           sd_order=float(np.std(ordr, ddof=1)), sd_both=float(np.std(both, ddof=1)))
    print(f"  {nm}: five starting seeds (order 1865) " + ", ".join(f"{v:.4f}" for v in ini) + f"  sd {np.std(ini, ddof=1):.4f}")
    print(f"  {nm}: five order seeds (start 1865)   " + ", ".join(f"{v:.4f}" for v in ordr) + f"  sd {np.std(ordr, ddof=1):.4f}")
    print(f"  {nm}: five seeds, both changed       " + ", ".join(f"{v:.4f}" for v in both) + f"  sd {np.std(both, ddof=1):.4f}")
    O[f"five_{nm}"]["pred_sd_both"] = float(math.sqrt(O[f"five_{nm}"]["sd_init"] ** 2 + O[f"five_{nm}"]["sd_order"] ** 2))
    print(f"    spreads added as squares: sqrt({O[f'five_{nm}']['sd_init']:.4f}^2 + {O[f'five_{nm}']['sd_order']:.4f}^2) = {O[f'five_{nm}']['pred_sd_both']:.4f}")
# Part 4
for k, nm in ((1, "carroll"), (2, "glass")):
    A = np.array([fin(f"i{s}_o{s}_t1", k) for s in S5])
    B = np.array([fin(f"i{s}_o{s}_t1_lr1.4e-3", k) for s in S5])
    dlt = B - A
    mean, sd = float(dlt.mean()), float(dlt.std(ddof=1))
    pair = [(i, j, float(B[j] - A[i])) for i in range(5) for j in range(5)]
    wrong_unpaired = sum(1 for *_, v in pair if np.sign(v) != np.sign(mean))
    subsets = list(itertools.combinations(range(5), 3))
    three = [float(dlt[list(c)].mean()) for c in subsets]
    three_sd = [float(dlt[list(c)].std(ddof=1)) for c in subsets]
    resolved3 = sum(1 for m_, s_ in zip(three, three_sd) if abs(m_) > 2 * s_ / math.sqrt(3))
    O[f"lr_{nm}"] = dict(lr1e3=A.tolist(), lr14e3=B.tolist(), diffs=dlt.tolist(), mean=mean, sd=sd, se=sd / math.sqrt(5),
                         single_seed_wrong=int((np.sign(dlt) != np.sign(mean)).sum()), unpaired_25_wrong=wrong_unpaired,
                         three_seed_means=three, three_seed_wrong=sum(1 for v in three if np.sign(v) != np.sign(mean)),
                         three_seed_resolved=resolved3, spread_lr1e3=float(A.std(ddof=1)))
    L = O[f"lr_{nm}"]
    print(f"Part 4. {nm}: lr 1e-3 " + ", ".join(f"{v:.4f}" for v in A) + " | lr 1.4e-3 " + ", ".join(f"{v:.4f}" for v in B))
    print(f"    differences (1.4e-3 minus 1e-3, same seed) " + ", ".join(f"{v:+.4f}" for v in dlt)
          + f"; mean {mean:+.4f}, sd {sd:.4f}, se {sd / math.sqrt(5):.4f}")
    print(f"    single seeds pointing the other way: {L['single_seed_wrong']} of 5; unpaired pairs (seed a for one, b for the other): "
          f"{wrong_unpaired} of 25 the other way; three-seed means the other way {L['three_seed_wrong']} of 10, "
          f"resolved at 2 se {resolved3} of 10")
# Part 5
from _pool import heldout_sets, corpora, POOL_BOOKS
c = corpora()
pool = np.concatenate([np.asarray(c["books"][b]) for b in POOL_BOOKS])
cnt = np.bincount(pool, minlength=2048)
from tiny.eval import windows
G = torch.tensor(np.asarray(windows(c["glass"])))[:64]   # the first 64 non-overlapping Looking-Glass windows (8,192 predictions)
tgt = G[:, 1:].reshape(-1).numpy()
bands = [(0, 99, "0-99"), (100, 999, "100-999"), (1000, 9999, "1,000-9,999"), (10000, 10 ** 9, "10,000+")]
def logp(path):
    tt = load(path)
    with torch.no_grad():
        return torch.log_softmax(tt.scores(G[:, :-1]).double(), -1).reshape(-1, 2048)
CK = os.path.join(HERE, "outputs", "ckpt")
pairs = {"starting seed (1865 vs 7, order 1865)": ("seeds_i1865_o1865_t1.pt", "seeds_i7_o1865_t1.pt"),
         "order seed (1865 vs 7, start 1865)": ("seeds_i1865_o1865_t1.pt", "seeds_i1865_o7_t1.pt"),
         "threads (1 vs 4)": ("seeds_i1865_o1865_t1.pt", "seeds_i1865_o1865_t4.pt"),
         "one-ulp nudge": ("seeds_i1865_o1865_t1.pt", "seeds_i1865_o1865_t1_nudge.pt"),
         "2,500 steps, order 1865 vs 7": (os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"),
                                          os.path.join(HERE, "..", "ch25", "outputs", "ckpt", "wsd_1e-3_2000_o7_b2000.pt"))}
O["disagree"] = {}
print("Part 5. two models on the first 64 Looking-Glass windows: KL (nats per prediction) and top-choice agreement, by pool count of the true token")
for nm, (x, y) in pairs.items():
    px = logp(os.path.join(CK, x) if not os.path.isabs(x) else x); py = logp(os.path.join(CK, y) if not os.path.isabs(y) else y)
    kl = (px.exp() * (px - py)).sum(-1).numpy()
    agree = (px.argmax(-1) == py.argmax(-1)).numpy()
    row = dict(kl=float(kl.mean()), agree=float(agree.mean()), bands={})
    for lo, hi, bn in bands:
        m = (cnt[tgt] >= lo) & (cnt[tgt] <= hi)
        row["bands"][bn] = dict(n=int(m.sum()), kl=float(kl[m].mean()), agree=float(agree[m].mean()))
    O["disagree"][nm] = row
    print(f"  {nm:34s} KL {row['kl']:.3g}, top choice agrees {row['agree']:.3f};  by band: "
          + "; ".join(f"{bn} {v['kl']:.2g}/{v['agree']:.2f}" for bn, v in row["bands"].items()))
print("  tokens per band:", {bn: v["n"] for bn, v in row["bands"].items()})
json.dump(O, open(os.path.join(HERE, "outputs", "analyse.json"), "w"), indent=1)
