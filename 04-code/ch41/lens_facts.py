"""Chapter 41, 41.2: the logit lens on the 30 taught facts (Chapter 38's fine-tuned Tiny, base beside it).  At the answer's animal name (the site
of Chapter 40), the rank of the taught verb among the 2,048 tokens when each layer's rows are read through the last norm and the tied table;
the number of animals whose verb is first at layer l; the layer at which it first becomes first.  32-bit, seed 1865."""
from lib import *
import numpy as np
torch.set_num_threads(1)
exec(open("facts_common.py").read())
out = {}
for nm, tt in (("fine-tuned", ft), ("base", base)):
    first = []; at = np.zeros(13, int); ranks = np.zeros((len(names), 13), int)
    for i, a in enumerate(names):
        e = P[a]
        with torch.no_grad():
            r = forward(tt, e["ids"][:e["site"] + 1])["resid"]
            for l in range(13):
                lg = tt._norm(r[l][0, -1], "ln_f") @ tt.t["E"].T
                ranks[i, l] = int((lg > lg[e["verb"]]).sum()) + 1
        at += ranks[i] == 1; f = [l for l in range(13) if ranks[i, l] == 1]; first.append(f[0] if f else None)
    out[nm] = dict(first_one=at.tolist(), median_rank=np.median(ranks, 0).tolist(), first=first)
    print(f"{nm:10s} animals with the taught verb as the lens's first choice, by layer 0..12: " + " ".join(str(x) for x in at))
    print(f"{nm:10s} median rank of the taught verb by layer 0..12: " + " ".join(f"{x:g}" for x in np.median(ranks, 0)))
    ff = [x for x in first if x is not None]; print(f"{nm:10s} animals for which the verb is ever first: {len(ff)} of 30; median first layer {np.median(ff) if ff else None}")
save_json("lens_facts", out)
