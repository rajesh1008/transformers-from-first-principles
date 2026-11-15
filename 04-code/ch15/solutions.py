"""Chapter 15: worked answers to the exercises that have a number in them.
Run from 04-code/ch15: python3 solutions.py > outputs/solutions.txt"""
from _common import *
from hand import hand_tiny, hand_tables, layer_tables, IDS, WORDS, CONFIG, W_1, b_1, W_2
from tiny.embed import TOY_EMBED, TOY_POS
from tiny.attention import softmax
from tiny.model import Tiny
from tiny.norm import token_similarity

def surprise(S):
    sh = softmax(S); return float(np.mean([-np.log(sh[i, IDS[i + 1]]) for i in range(5)]))

print("Exercise 1 (by hand): cat@1's layer 0 shares from scaled scores 2.600 and 1.697")
e = np.exp([2.600, 1.697]); print(f"  e^ {e[0]:.3f}, {e[1]:.3f}; total {e.sum():.3f}; shares {e[0]/e.sum():.3f}, {e[1]/e.sum():.3f}")

print("Exercise 2 (by hand): RMSNorm of on@3's row (0.5, -1)")
r = np.array([0.5, -1.0]); rms = np.sqrt((r ** 2).mean()); print(f"  rms {rms:.3f}; normed {r3(r / rms)}")

print("Exercise 3 (by hand): the feed-forward block on the@0's layer 0 row after ln_2, (1.359, 0.391)")
h = np.array([1.359, 0.391]); mt = h @ W_1 + b_1; a = np.maximum(mt, 0)
print(f"  matches {r3(mt)}; ReLU {r3(a)}; writes {r3(a @ W_2)}")

print("Exercise 4 (by hand): the score for 'dog' after mat@5, top row after ln_f (1.042, 0.957), dog's row (0, 2)")
print(f"  {np.dot([1.042, 0.957], TOY_EMBED[WORDS.index('dog')]):.3f}")

m = hand_tiny(); S0 = m.forward(IDS)
print(f"\nbaseline: top words {[WORDS[k] for k in S0.argmax(1)]}; average surprise {surprise(S0):.4f}")

print("Exercise 5 (in code): W_q set to the identity in both layers")
t = hand_tables()
for l in range(2): t[f"h.{l}.att.W_q"] = np.eye(2)
m5 = Tiny(**CONFIG).load(t); S5 = m5.forward(IDS)
_, sh5 = m5.blocks[0].att(m5.blocks[0].ln_1(m5.embed(IDS)), return_shares=True)
print(f"  top words {[WORDS[k] for k in S5.argmax(1)]}; average surprise {surprise(S5):.4f}; sat@2's layer 0 shares {r3(sh5[0][2][:3])}")

print("Exercise 6 (in code): neuron 3's bias -2 changed to -1")
t = hand_tables()
for l in range(2): t[f"h.{l}.ff.b_1"] = np.array([0.0, 0.0, -1.0])
m6 = Tiny(**CONFIG).load(t); X = m6.embed(IDS)
for l, b in enumerate(m6.blocks):
    st = b.explode(X)
    act = st["ln_2"] @ W_1 + np.array([0, 0, -1.0])
    print(f"  layer {l}: neuron 3 matches {r3(act[:, 2])}; fires for {[f'{WORDS[IDS[i]]}@{i}' for i in range(6) if act[i, 2] > 0]}")
    X = st["add_2"]
S6 = m6.forward(IDS); print(f"  top words {[WORDS[k] for k in S6.argmax(1)]}; average surprise {surprise(S6):.4f}")

print("Exercise 7 (in code): the same tables stacked three and four layers deep")
for L in (1, 2, 3, 4):
    t = {"E": TOY_EMBED.astype(float), "P": TOY_POS.astype(float), "ln_f.gain": np.ones(2)}
    for l in range(L):
        for k, v in layer_tables().items(): t[f"h.{l}.{k}"] = np.array(v, dtype=float)
    cfg = dict(CONFIG); cfg["n_layers"] = L
    mL = Tiny(**cfg).load(t); X = mL.embed(IDS); sims = [token_similarity(X)]
    for b in mL.blocks:
        X = b(X); sims.append(token_similarity(X))
    S = mL.forward(IDS)
    print(f"  {L} layer(s): token similarity by height {[round(s, 3) for s in sims]}; top words {sorted(set(WORDS[k] for k in S.argmax(1)))}; "
          f"average surprise {surprise(S):.4f}; share of 'looked' after mat@5 {softmax(S)[5, WORDS.index('looked')]:.3f}")
