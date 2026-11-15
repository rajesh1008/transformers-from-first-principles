"""Chapter 17, section 17.4: backpropagation on a three-node network, by hand, then checked twice.

    input x --(w1, b1)--> z = w1 x + b1 --ReLU--> h = max(0, z) --(w2, b2)--> y = w2 h + b2
    loss = (y - t) squared, t the target

Four learned numbers (w1, b1, w2, b2).  Values chosen for hand arithmetic; nothing random.
The backward pass is the chain rule applied from the loss back to each number: every step multiplies
the nudge arriving from above by the local slope of that step."""
import _common
import torch

x, t = 2.0, 2.0
w1, b1, w2, b2 = 0.5, -0.5, 1.5, 0.0

def forward(w1, b1, w2, b2, x=x, t=t):
    z = w1 * x + b1
    h = max(0.0, z)
    y = w2 * h + b2
    return z, h, y, (y - t) ** 2

z, h, y, L = forward(w1, b1, w2, b2)
print("forward:  z = w1 x + b1 = 0.5 x 2 - 0.5 =", z, "| h = max(0, z) =", h, "| y = w2 h + b2 = 1.5 x 0.5 + 0 =", y,
      "| loss = (y - t)^2 = (0.75 - 2)^2 =", L)
dy = 2 * (y - t)
dw2, db2 = dy * h, dy * 1.0
dh = dy * w2
dz = dh * (1.0 if z > 0 else 0.0)
dw1, db1 = dz * x, dz * 1.0
print("backward: d loss/d y = 2(y - t) =", dy)
print("          d loss/d w2 = d loss/d y x h =", dw2, "| d loss/d b2 = d loss/d y x 1 =", db2)
print("          d loss/d h  = d loss/d y x w2 =", dh)
print("          d loss/d z  = d loss/d h x (1 because z > 0) =", dz)
print("          d loss/d w1 = d loss/d z x x =", dw1, "| d loss/d b1 = d loss/d z x 1 =", db1)

def wig(i, hh=1e-6):
    p = [w1, b1, w2, b2]; q = list(p)
    p[i] += hh; q[i] -= hh
    return (forward(*p)[3] - forward(*q)[3]) / (2 * hh)
print("wiggled:  ", [round(wig(i), 6) for i in range(4)])
P = torch.tensor([w1, b1, w2, b2], dtype=torch.float64, requires_grad=True)
zz = P[0] * x + P[1]; yy = P[2] * torch.relu(zz) + P[3]; LL = (yy - t) ** 2
LL.backward()
print("autograd: ", [round(float(g), 6) for g in P.grad])

print("\none step of gradient descent, learning rate 0.05:")
lr = 0.05
new = [w1 - lr * dw1, b1 - lr * db1, w2 - lr * dw2, b2 - lr * db2]
print("  numbers", [round(v, 4) for v in new], "loss", round(forward(*new)[3], 4), "(was", L, ")")
print("\nlearning rates, 30 steps each:")
for lr in (0.05, 0.3, 1.0, 1.2):
    p = [w1, b1, w2, b2]; Ls = []; dead = None
    for k in range(30):
        zz_, hh_, yy_, LL_ = forward(*p); Ls.append(LL_)
        if zz_ <= 0 and dead is None:
            dead = k
        d_y = 2 * (yy_ - t); d_z = d_y * p[2] * (1.0 if zz_ > 0 else 0.0)
        g = [d_z * x, d_z, d_y * hh_, d_y]
        p = [a - lr * b for a, b in zip(p, g)]
    note = "" if dead is None else f"; the ReLU switched off at step {dead} and never came back (z = {zz_:.3g}), b2 now {p[3]:.4g}"
    print(f"  lr {lr}: loss " + " ".join(f"{v:.3g}" for v in Ls[:6]) + f" ... step 29 {Ls[-1]:.3g}{note}")
print("\nthe nudge that is exactly zero: start with b1 = -1.5, so z = -0.5 and the ReLU is off")
z2, h2, y2, L2 = forward(w1, -1.5, w2, b2)
print(f"  z {z2}, h {h2}, y {y2}, loss {L2}; d loss/d w1 = d loss/d b1 = 0, d loss/d w2 = 0 (h = 0); only b2 moves:",
      [round(v, 6) for v in [(forward(w1+1e-6,-1.5,w2,b2)[3]-forward(w1-1e-6,-1.5,w2,b2)[3])/2e-6,
                             (forward(w1,-1.5+1e-6,w2,b2)[3]-forward(w1,-1.5-1e-6,w2,b2)[3])/2e-6,
                             (forward(w1,-1.5,w2+1e-6,b2)[3]-forward(w1,-1.5,w2-1e-6,b2)[3])/2e-6,
                             (forward(w1,-1.5,w2,b2+1e-6)[3]-forward(w1,-1.5,w2,b2-1e-6)[3])/2e-6]])
