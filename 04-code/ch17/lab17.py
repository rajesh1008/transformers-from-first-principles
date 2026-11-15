"""Lab 17: nudge one learned number to the bottom of its hill, break gradient descent three ways, then
pass the checkpoint.  64-bit NumPy and PyTorch; nothing random.  Run from 04-code/ch17."""
import _common
import numpy as np
import torch
from hand import hand_tiny, IDS, WORDS
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.train import wiggle_grads, train

# ---------- Part 1: one learned number, by hand ----------
m = hand_tiny()
S, stream = m.forward(IDS, keep=True)
TOP = m.top_rows(stream[-1])                 # each seat's top row after the final norm; mat's row changes nothing upstream
E0 = m.E.table.copy()

def loss_of_x(x):
    e = E0.copy(); e[4, 0] = x
    z = TOP[:5] @ e.T
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-lp[np.arange(5), IDS[1:]].mean())

def slope_of_x(x, h=1e-5):
    return (loss_of_x(x + h) - loss_of_x(x - h)) / (2 * h)

print("Part 1: x is mat's first number in the hand Tiny; learning rate 2")
x = -1.0
for step in range(16):
    if step % 3 == 0 or step == 15:
        print(f"  step {step:>2}: x {x:+.4f}  loss {loss_of_x(x):.4f}  slope {slope_of_x(x):+.5f}")
    x -= 2.0 * slope_of_x(x)
print(f"  the bottom is at x = 1.6303 with loss 2.9126; we reached x = {x:.4f}, loss {loss_of_x(x):.4f}")

# ---------- Part 2: breakages ----------
print("\nBreakage 1, wrong sign (x <- x + lr x slope):")
x = -1.0
for step in range(4):
    print(f"  step {step}: x {x:+.4f}  loss {loss_of_x(x):.4f}")
    x += 2.0 * slope_of_x(x)

print("\nBreakage 2, learning rate too large for the hand Tiny (all 112 numbers, lr 8):")
hist = train(hand_tiny(), lambda s: IDS, 6, 8.0)
print("  loss by step:", ", ".join(f"{L:.4g}" for _, L in hist))

print("\nBreakage 3, a row used in three places, nudge kept from one place only:")
tt = TorchTiny.from_numpy(hand_tiny())
L, G = tt.grads(IDS)
X = (tt.t["E"][torch.tensor(IDS)] + tt.t["P"][:6]).detach().requires_grad_(True)
Ls = -torch.log_softmax(tt.scores_from_rows(X[:-1]), -1).gather(-1, torch.tensor(IDS[1:]).unsqueeze(-1)).mean()
Ls.backward(); gX = X.grad.numpy()
out_only = G["E"][0] - gX[0] - gX[4]
W = wiggle_grads(hand_tiny(), IDS, names=["E"])
print(f"  the's nudge, output use only: {np.round(out_only, 4)}; all three uses added: {np.round(G['E'][0], 4)}; wiggled: {np.round(W['E'][0], 4)}")

print("\nBreakage 4, wiggle too small in 32-bit (E['cat'][0] of the hand Tiny, true nudge -0.218176):")
m32 = hand_tiny()
for k, v in m32.tables().items():
    v[...] = v.astype(np.float32)
def loss32(val):
    e = m32.E.table; old = e[1, 0]; e[1, 0] = val
    tt32 = TorchTiny.from_numpy(m32, dtype=torch.float32)
    with torch.no_grad():
        out = float(tt32.loss(IDS))
    e[1, 0] = old
    return np.float32(out)
for h in (1e-2, 1e-3, 1e-5, 1e-7):
    s = (loss32(np.float32(0 + h)) - loss32(np.float32(0 - h))) / np.float32(2 * h)
    print(f"  h = {h:g}: wiggled slope {float(s):+.6f}")

# ---------- Part 3: the checkpoint ----------
print("\nPart 3: checkpoint")
tests = []
def check(name, ok):
    tests.append(ok); print(f"  {'PASS' if ok else 'FAIL'}  {name}")
check("the slope of the one-number loss at x = -1 is -0.22601", abs(slope_of_x(-1.0) + 0.22601) < 1e-5)
check("one step at learning rate 2 lowers the loss", loss_of_x(-1.0 - 2 * slope_of_x(-1.0)) < loss_of_x(-1.0))
pw = torch.tensor([0.5, -0.5, 1.5, 0.0], dtype=torch.float64, requires_grad=True)
y = pw[2] * torch.relu(pw[0] * 2.0 + pw[1]) + pw[3]; ((y - 2.0) ** 2).backward()
check("three-node network: the nudge on w1 is -7.5", abs(float(pw.grad[0]) + 7.5) < 1e-12)
check("hand Tiny: the nudge on E['cat'][0] is -0.218176", abs(G["E"][1, 0] + 0.218176) < 5e-7)
check("the's three nudges add to its total", np.allclose(gX[0] + gX[4] + out_only, G["E"][0]) and np.allclose(G["E"][0], W["E"][0], atol=1e-7))
check("hand Tiny: the position row for seat 5 gets exactly zero nudge", bool(np.all(G["P"][5] == 0)))
t = Tiny(); ids = _common.novel_ids()[:1025]
check("Tiny unchanged: 2,766,592 numbers, loss 7.6334 on 1,024 predictions, PyTorch mirror agrees",
      t.n_params == 2766592 and abs(t.loss(ids) - 7.6334) < 5e-5 and abs(float(TorchTiny.from_numpy(t).loss(ids).detach()) - t.loss(ids)) < 1e-9)
print(f"{sum(tests)} of {len(tests)} passed")
assert all(tests)
