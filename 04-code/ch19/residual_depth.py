"""Chapter 19, section 19.4: scaling residual branches by depth.  Tiny-shaped stacks (d 128, four heads,
d_ff 512, pre-norm) of 12, 24, 48 and 96 layers, each layer drawn with its usual seed (1865 + 100 x layer),
under "gpt2" (every table 0.02) and "scaled" (W_o and W_2 divided by sqrt(2 x layers)).  One forward and
backward pass (64-bit) on four windows of 129 tokens of the novel."""
import _common
import numpy as np
from tiny.model import Tiny
from tiny.init import initialise
from _common import novel_ids, tok, save_json, SEED, probe
T = tok(); nov = np.array(novel_ids(T))
starts = np.random.default_rng(SEED).integers(0, len(nov) - 129, size=4)
B = np.stack([nov[a:a + 129] for a in starts])
out = {}
print("layers  start    stream out   growth  last layer's change   nudge on stream: layer 0 / top   loss")
for L in (12, 24, 48, 96):
    for scheme in ("gpt2", "scaled"):
        m = initialise(Tiny(n_layers=L), scheme)
        P = probe(m, B)
        s = P["stream"]
        last = np.sqrt(P["att"][-1] ** 2 + P["ff"][-1] ** 2) / s[-2]
        growth = s[-1] / s[0]
        ratio = P["nudge"][0] / P["nudge"][-1]
        out[f"{L}|{scheme}"] = dict(stream=s, nudge=P["nudge"], att=P["att"], ff=P["ff"], loss=P["loss"])
        print(f"{L:5d}   {scheme:7s} {s[-1]:9.4f}   {growth:7.2f}x      {last:7.4f}          {P['nudge'][0]:.3g} / {P['nudge'][-1]:.3g} ({ratio:.1f}x)"
              f"    {P['loss']:.4f}")
save_json("residual_depth", out)
