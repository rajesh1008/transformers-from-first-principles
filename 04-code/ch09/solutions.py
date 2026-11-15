"""Chapter 9: answers to the exercises with numerical answers (Appendix A).
Run from 04-code/ch09:  python3 solutions.py"""
from _common import *
from tiny.attention import attention, softmax, causal_mask, Head, MultiHead
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe

Q, K, V = X @ W_q, X @ W_k, X @ W_v
M = causal_mask(6)
heads = [softmax(np.where(M, np.outer(Q[:, h], K[:, h]), -np.inf)) for h in (0, 1)]
print("1. on's row, head 1:", r3(heads[0][3, :4]), " head 2:", r3(heads[1][3, :4]))
print("2. on's outputs side by side:", r3([heads[0][3] @ V[:, 0], heads[1][3] @ V[:, 1]]))
rng = np.random.default_rng(SEED)
ok = True
for _ in range(1000):
    wq, wk = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
    s = (X[5] @ wq) @ (X @ wk).T
    ok &= np.isclose(s[0], (s[2] + s[3]) / 2)
print("3. score for 'the' is halfway between 'sat' and 'on' for 1,000 random head tables:", bool(ok),
      "| best share for both at once: 1/4 (two thes tie with them)")
outs = [np.array([1.0, 1.0]), np.array([0.0, 2.0]), np.array([-1.0, 1.0])]
Wo = np.array([[1, 0], [0, 1], [1, 1], [2, 0], [0, 0], [1, -1]], dtype=float)
cat_ = np.concatenate(outs) @ Wo
blocks = sum(o @ Wo[2 * h:2 * h + 2] for h, o in enumerate(outs))
print("4. concatenated:", cat_.tolist(), " by blocks:", blocks.tolist(), " head 2 removed:", (outs[0] @ Wo[:2] + outs[2] @ Wo[4:]).tolist())
print("5. heads, width, shares per 1,024 window:", [(h, 128 // h, h * 1024 * 1024) for h in (1, 2, 4, 8, 16, 32, 64, 128)])
d = 768; h = 12
print(f"6. GPT-2 small: head width {d // h}, divide by {np.sqrt(d // h):.0f}; query+key+value {3 * d * d:,}, output {d * d:,}, total {4 * d * d:,} per layer")
print("7. seat 9 (sat) -> on, share 1.000; seat 11 (the) -> cat, share 0.667 (see induction_toy.txt)")
k = np.array([0.3, -1.2, 2.0, 0.5])
for q in (0.7, -0.7):
    sh = softmax(q * k)
    print(f"8. query {q}: shares {r3(sh)}, favourite key {k[np.argmax(sh)]}")
tok = load_bpe(TINY_BPE); ids = tok.encode(SENTENCE)
Xt = Embedding(len(tok))(ids) + PositionTable()(range(len(ids)))
_, A1 = Head()(Xt, return_shares=True)
m1 = MultiHead(n_heads=1, d_head=128)
_, A2 = m1(Xt, return_shares=True)
print("9. MultiHead(1 x 128) shares equal Head() shares:", bool(np.allclose(A1, A2[0])),
      "| outputs equal:", bool(np.allclose(Head()(Xt), m1(Xt))), "(MultiHead multiplies by W_o)")
m = MultiHead(n_heads=4)
o, _ = m.head_outputs(Xt)
contrib = [o[h] @ m.W_o[32 * h:32 * (h + 1)] for h in range(4)]
print("10. the four contributions add up to the layer's output:", bool(np.allclose(sum(contrib), m(Xt))))
