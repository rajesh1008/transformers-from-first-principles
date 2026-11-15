"""Chapter 18, sections 18.3 to 18.7: the nudge dY arriving at layer 0's head, pushed back by hand through the
output table, the blend, the softmax, the scores, the query, key and value tables and the norm, to the rows
entering layer 0.  Every step is computed with the rules of the chapter (tiny.attention.head_backward) and
compared with PyTorch autograd's nudge on the same intermediate, and the table nudges with Chapter 17's.
64-bit; nothing random."""
import _common, json, os
import numpy as np
from hand import hand_tiny, IDS
from tiny.attention import head_forward, head_backward
from tiny.train import torch_grads
from _common import traced, npy, SEATS, f4, row, save_json, CH17_OUT

m = hand_tiny()
b = m.blocks[0]
L, k = traced(m, IDS); L.backward()
dY = npy(k["Y"].grad).squeeze()
X = m.embed(IDS[:-1]); A = b.ln_1(X)
f = head_forward(A, b.att.W_q, b.att.W_k, b.att.W_v, b.att.W_o)
g = head_backward(f, dY)
auto = {"O": k["O"], "P": k["P"], "S": k["S"], "Q": k["Q"], "K": k["K"], "V": k["V"], "A": k["a"]}

def show(name, T, kk=5, mask=False):
    print(f"{name}:")
    labels = ["row 0", "row 1"] if name.startswith("dW") else SEATS
    for s, r in enumerate(np.atleast_2d(T)):
        cells = "  ".join((f4(v, kk) if (not mask or j <= s) else "      .") for j, v in enumerate(r))
        print(f"  {labels[s]:<6} {cells}")

ladder = []
def check(name, mine, theirs):
    d = float(np.abs(mine - theirs).max()); ladder.append((name, d))
    print(f"  by hand = autograd to {d:.1e}")

print("Step 1, the output table (Y = O W_o, W_o the identity)")
show("dW_o = O^T dY", g["W_o"]); check("W_o", g["W_o"], torch_grads(m, IDS)[1]["h.0.att.W_o"])
show("dO = dY W_o^T = dY", g["O"]); check("O", g["O"], npy(auto["O"].grad))

print("\nStep 2, through the blend (O = P V)")
show("dP = dO V^T (only the seats each row could see matter)", g["P"], mask=True)
check("P (seen seats)", np.tril(g["P"]), np.tril(npy(auto["P"].grad).squeeze()))
show("dV = P^T dO", g["V"]); check("V", g["V"], npy(auto["V"].grad).squeeze())

print("\nStep 3, through the softmax, row by row: dS = P x (dP - sum over the row of P x dP)")
rs = (f["P"] * g["P"]).sum(-1)
print("  the row-sum term, one number per seat:", " ".join(f4(v, 5) for v in rs))
show("dS", g["S"], mask=True); check("S", g["S"], npy(auto["S"].grad).squeeze())
print(f"  every row of dS adds to zero: largest row sum {np.abs(g['S'].sum(-1)).max():.1e}")

print("\nStep 4, through the scores (S = Q K^T / sqrt 2)")
show("dQ = dS K / sqrt 2", g["Q"]); check("Q", g["Q"], npy(auto["Q"].grad).squeeze())
show("dK = dS^T Q / sqrt 2", g["K"]); check("K", g["K"], npy(auto["K"].grad).squeeze())

print("\nStep 5, into the three tables (Q = A W_q and so on: dW = A^T dQ)")
G = torch_grads(m, IDS)[1]
for name in ("W_q", "W_k", "W_v"):
    print(f"  d{name} = A^T d{name[-1].upper()} = {np.round(g[name], 4).tolist()}  (Chapter 17's autograd {np.round(G['h.0.att.' + name], 4).tolist()})")
    check(name, g[name], G["h.0.att." + name])

print("\nStep 6, back to the rows the head read: A fed Q, K and V, so its nudge is the sum of three")
parts = {n: g[n.upper()] @ f["W_" + n].T for n in ("q", "k", "v")}
for s in range(5):
    print(f"  {SEATS[s]:<6} via Q {row(parts['q'][s], 5)} + via K {row(parts['k'][s], 5)} + via V {row(parts['v'][s], 5)} = {row(g['A'][s], 5)}")
check("A", g["A"], npy(auto["A"].grad))

print("\nStep 7, through ln_1 (RMSNorm) and the skip path, to the rows entering layer 0")
r = np.sqrt((X ** 2).mean(-1, keepdims=True) + 1e-5); u = g["A"] * b.ln_1.gain
through = u / r - X * (u * X).sum(-1, keepdims=True) / (2 * r ** 3)
dX = dY + through
ch17 = json.load(open(os.path.join(CH17_OUT, "head_nudges.json")))["seat_nudges"]
for s in range(5):
    print(f"  {SEATS[s]:<6} skip {row(dY[s])} + through the head {row(through[s])} = {row(dX[s])}   Chapter 17: {row(ch17[s])}")
check("X", dX, np.array(ch17[:5]))
print("\nagreement, step by step (largest difference from autograd):")
for name, d in ladder:
    print(f"  {name:<15} {d:.1e}")
save_json("backward_head", dict(dY=dY.tolist(), g={kk: v.tolist() for kk, v in g.items()}, rowsum=rs.tolist(),
                                ladder=ladder, dX=dX.tolist(), through=through.tolist(), parts={kk: v.tolist() for kk, v in parts.items()}))
