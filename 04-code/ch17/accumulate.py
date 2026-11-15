"""Chapter 17, section 17.10: PyTorch adds each backward pass's nudges into .grad rather than replacing them.
Train the hand Tiny for 200 steps at learning rate 0.5 twice, with .grad cleared before each backward pass and
without.  64-bit; nothing random."""
import _common
import torch
from hand import hand_tiny, IDS
from tiny.torch_tiny import TorchTiny

def run(clear, steps=200, lr=0.5):
    tt = TorchTiny.from_numpy(hand_tiny()); out = []
    for s in range(steps):
        if clear:
            for v in tt.t.values():
                v.grad = None
        L = tt.loss(IDS); L.backward(); out.append(float(L.detach()))
        with torch.no_grad():
            for v in tt.t.values():
                v -= lr * v.grad
    return out, float(torch.sqrt(sum((v.grad ** 2).sum() for v in tt.t.values())))
for clear in (True, False):
    Ls, gl = run(clear)
    print(f"{'cleared   ' if clear else 'not cleared'}: loss at steps 0, 10, 50, 100, 150, 199: "
          + ", ".join(f"{Ls[i]:.4g}" for i in (0, 10, 50, 100, 150, 199)) + f"; lowest {min(Ls):.4g}" + f"; length of the last .grad {gl:.3g}")
