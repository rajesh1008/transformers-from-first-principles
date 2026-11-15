"""Lab 28: predict, break three decoders, run the checkpoints.  Chapter 24's 2,500-step model, 32-bit, one thread, seed 1865.
Makes no API call; reads the key from ../../.env into this process only to check that its string appears in no file here."""
from _common import *
import torch, glob
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G

CK = os.path.join(HERE, "..", "ch24", "outputs", "ckpt")
tt = load(os.path.join(CK, "wsd_1e-3_2000_b2000.pt")); ama = load(os.path.join(CK, "wsd_1e-3_2000_b500.pt"))
t = tok(); P = t.encode("the cat sat on the")
s = G.next_scores(tt, P); p = G.probs(s)
print("Part 1. predict, then read")
q = np.array([0.5, 0.3, 0.2]); Hq = -(q * np.log(q)).sum()
print(f"  table 0.5 / 0.3 / 0.2: entropy {Hq:.3f} nats; surprises " + " / ".join(f"{-np.log(x):.3f}" for x in q))
print(f"  0.9 coin, 100 tosses, 80 heads: surprise {80 * -np.log(0.9) + 20 * -np.log(0.1):.2f} nats, {(80 * -np.log(0.9) + 20 * -np.log(0.1)) / 100:.3f} per toss; entropy {-(0.9 * np.log(0.9) + 0.1 * np.log(0.1)):.3f}")
qt, nt = G.typical_shares(s, 0.2); top = np.argsort(-p)
print(f"  after the prompt, typical tau 0.2 keeps {nt} tokens; is the line break (top share {p[top[0]]:.4f}) among them? {bool(qt[top[0]] > 0)}")
print(f"  rank of the best-ranked kept token: {int(np.where(qt[top] > 0)[0][0]) + 1}")

print("\nPart 2. three breakages")
sc, k = G.contrastive_scores(s, G.next_scores(ama, P), alpha=1e-300); j = int(np.argmax(sc))
print(f"  B1 contrastive with no plausibility cut picks {t.decode([j])!r}, expert share {p[j]:.2e} (rank {int(np.where(top == j)[0][0]) + 1} of 2048)")
o = np.argsort(-np.log(p))                       # wrong: order by surprise alone (smallest first), not by |surprise - entropy|
wrong = set(o[:int(np.searchsorted(np.cumsum(p[o]), 0.2) + 1)].tolist())
qp, _ = G.shares(s, top_p=0.2)
print(f"  B2 typical sampling sorted by surprise alone keeps {len(wrong)} tokens, the same set as top-p 0.2: {wrong == set(np.where(qp > 0)[0].tolist())}")
def miro(sign, target=3.0, n=40):
    rng = streams(SEED)["sample"]; ids = list(P); mu = 2 * target; su = []
    for _ in range(n):
        sc_ = G.next_scores(tt, ids); qq, _ = G.mirostat_step(sc_, mu); nx = int(rng.choice(len(qq), p=qq))
        x = -np.log(G.probs(sc_)[nx]); mu -= sign * 0.1 * (x - target); su.append(x); ids.append(nx)
    return np.mean(su[:20]), np.mean(su[20:]), mu
a = miro(+1); b = miro(-1)
print(f"  B3 Mirostat target 3.0, correct update: surprise {a[0]:.3f} then {a[1]:.3f}, final bar {a[2]:.2f}; "
      f"sign flipped: {b[0]:.3f} then {b[1]:.3f}, final bar {b[2]:.2f}")

print("\nPart 3. checkpoints")
ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"[{'pass' if cond else 'FAIL'}] {name}")
check("typical sampling at tau 1.0 gives the plain shares", np.allclose(G.typical_shares(s, 1.0)[0], p))
check("eta-sampling at eps 0 keeps every token", G.eta_shares(s, 0.0)[1] == 2048)
g, _ = G.generate(tt, P, 20, None)
ids = list(P)
for _ in range(20):
    sc_, _ = G.contrastive_scores(G.next_scores(tt, ids), G.next_scores(ama, ids), alpha=1.0); ids.append(int(np.argmax(sc_)))
check("contrastive decoding with alpha 1.0 equals greedy for 20 tokens", ids == g)
opts = [[int(x)] for x in top[:5]]
check("exact and masked choice agree when no option shares a first token", np.allclose(G.exact_option_shares(tt, P, opts), G.local_mask_probs(tt, P, opts)))
r1, r2 = streams(SEED)["sample"], streams(SEED)["sample"]; i1, i2 = list(P), list(P); mu = 200.0
for _ in range(10):
    sc_ = G.next_scores(tt, i1); qq, _ = G.mirostat_step(sc_, mu); i1.append(int(r1.choice(2048, p=qq)))
i2, _ = G.generate(tt, P, 10, r2, T=1.0)
check("Mirostat with a bar of 200 nats equals plain sampling for 10 tokens", i1 == i2)
key = None
for line in open(os.path.join(HERE, "..", "..", ".env")):
    if line.strip().startswith("ANTHROPIC_API_KEY="):
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
found = [f for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f) and key and key in open(f, errors="ignore").read()]
check("the API key was read from .env and its string appears in no file under 04-code/ch28", bool(key) and not found)
del key
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
assert all(ok)
