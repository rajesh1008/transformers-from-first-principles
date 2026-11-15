"""Chapter 43, 'Where this bites': how many input tokens a hosted model charges for one image, by image size.
The first test digit (8 x 8 pixels) is scaled up with nearest-neighbour to square PNGs of 8 ... 3,072 pixels a side and sent to
Anthropic's token-counting endpoint (/v1/messages/count_tokens; nothing is generated) with the question "Which digit is this?";
the count for the question alone is subtracted.  Model id MODEL (default claude-haiku-4-5-20251001).
The key is read from ../../.env into this process only; it is never printed or written.  Writes outputs/image_tokens_MODEL.json."""
import os, io, json, base64, urllib.request
from PIL import Image
import numpy as np
from vit import data, HERE
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
def _key():
    for l in open(os.path.join(HERE, "..", "..", ".env")):
        if l.startswith("ANTHROPIC_API_KEY="): return l.split("=", 1)[1].strip().strip('"')
KEY = _key()
def count(content):
    body = json.dumps(dict(model=MODEL, messages=[dict(role="user", content=content)])).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens", data=body, method="POST",
                                 headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["input_tokens"]
_, _, Xte, yte = data(); img = (Xte[0].numpy() * 255).astype(np.uint8)
q = [dict(type="text", text="Which digit is this?")]; base = count(q); R = dict(model=MODEL, label=int(yte[0]), question_only=base, sizes=[])
print(f"model {MODEL}; the question alone counts {base} input tokens")
for side in (8, 32, 64, 128, 256, 512, 768, 1024, 1568, 2048, 2576, 3072):
    b = io.BytesIO(); Image.fromarray(img).resize((side, side), Image.NEAREST).save(b, "PNG")
    n = count([dict(type="image", source=dict(type="base64", media_type="image/png", data=base64.b64encode(b.getvalue()).decode()))] + q)
    R["sizes"].append(dict(side=side, pixels=side * side, tokens=n - base))
    print(f"{side:5d} x {side:<5d} {side*side:>10,d} pixels  image tokens {n-base:6,d}  (pixels per token {side*side/max(1,n-base):8.1f})", flush=True)
json.dump(R, open(os.path.join(HERE, "outputs", f"image_tokens_{MODEL}.json"), "w"), indent=1)
