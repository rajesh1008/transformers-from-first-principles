"""Chapter 19, sections 19.2 and 19.3: a plain chain of 48 layers, each  x -> act(x W),  W of 128 x 128 drawn
with spread s, no norms and no residual path.  256 rows of 128 numbers with spread 1 go in.  Forward: the spread
of the rows after layers 1, 6, 12, 24, 48.  Backward: a nudge of spread 1 is put on the output and carried back
down (times act'(x W), times W transposed); its spread on the rows entering layer 1 is reported.  64-bit."""
import _common
import numpy as np
from _common import SEED, save_json, rms
from tiny.ffn import ACTIVATIONS
ACT = {"linear": (lambda z: z, lambda z: np.ones_like(z)),
       "relu": (lambda z: np.maximum(z, 0), lambda z: (z > 0).astype(float))}
def gelu_d(z, h=1e-5):
    f = ACTIVATIONS["gelu"]
    return (f(z + h) - f(z - h)) / (2 * h)
ACT["gelu"] = (ACTIVATIONS["gelu"], gelu_d)
D, DEPTH = 128, 48
SPREADS = [("0.02", 0.02), ("1/sqrt(128)", 1 / np.sqrt(D)), ("sqrt(2)/sqrt(128)", np.sqrt(2 / D)), ("0.2", 0.2)]
SHOW = (1, 6, 12, 24, 48)
out = {}
with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    for act in ("linear", "relu", "gelu"):
        f, fd = ACT[act]
        print(f"\n{act}:  spread of the rows after layer " + ", ".join(map(str, SHOW)) + "   |  nudge reaching layer 1")
        for name, s in SPREADS:
            rng = np.random.default_rng(SEED)
            Ws = [rng.normal(0, s, size=(D, D)) for _ in range(DEPTH)]
            x = rng.normal(0, 1, size=(256, D))
            pres, fwd = [], [1.0]
            for W in Ws:
                z = x @ W; pres.append(z); x = f(z); fwd.append(float(rms(x)))
            d = np.random.default_rng(SEED + 1).normal(0, 1, size=x.shape)
            back = [float(rms(d))]
            for W, z in zip(reversed(Ws), reversed(pres)):
                d = (d * fd(z)) @ W.T; back.append(float(rms(d)))
            back = back[::-1]                       # back[l] = spread of the nudge on the rows entering layer l+1
            out[f"{act}|{name}"] = dict(s=s, forward=fwd, backward=back)
            print(f"  s = {name:18s} " + "  ".join(f"{fwd[k]:9.3g}" for k in SHOW) + f"   |  {back[0]:9.3g}")
save_json("chain", out)
