"""Chapter 43: does a patch grid explain the hosted model's image-token counts?  For each measured size, compare the count with
ceil(side / p)^2 + c for patch sides p from 14 to 32 and the constant c that fits best; report the p that fits every size
below the model's resize limit exactly.  Reads outputs/image_tokens_*.json."""
import json, glob, math, os
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "image_tokens_*.json"))):
    R = json.load(open(f)); S = R["sizes"]; cap = max(s["tokens"] for s in S); under = [s for s in S if s["tokens"] < cap]
    for p in range(14, 33):
        cs = {s["tokens"] - math.ceil(s["side"] / p) ** 2 for s in under}
        if len(cs) == 1:
            c = cs.pop(); side = round(math.sqrt(cap - c)) * p
            print(f"{R['model']}: patch side {p} px fits all {len(under)} sizes below the limit exactly, with {c} extra tokens per image; "
                  f"the plateau of {cap} tokens = {round(math.sqrt(cap - c))}^2 + {c}, an image resized to about {side} px a side")
