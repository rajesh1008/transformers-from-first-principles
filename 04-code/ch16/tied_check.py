"""Is Qwen3-0.6B's stored output table a copy of its token table?  The config says tied; the file
stores both.  Read three rows of each straight from the weights file with range requests (2 KB per
row in 16-bit) and compare them number by number.  Fetched 10 November 2026."""
import json, struct, urllib.request
import numpy as np
from _common import save_json
UA = {"User-Agent": "Mozilla/5.0 (book research; tfp)"}
URL = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors"

def get(a, b):
    r = urllib.request.Request(URL, headers=dict(UA, Range=f"bytes={a}-{b}"))
    return urllib.request.urlopen(r, timeout=60).read()

n = struct.unpack("<Q", get(0, 7))[0]
head = json.loads(get(8, 8 + n - 1))
start = 8 + n
def row(name, i):
    t = head[name]; d = t["shape"][1]; off = start + t["data_offsets"][0] + i * d * 2
    raw = np.frombuffer(get(off, off + d * 2 - 1), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)       # bfloat16 -> float32, exactly
res = {}
for i in (0, 1000, 151935):
    a, b = row("model.embed_tokens.weight", i), row("lm_head.weight", i)
    same = bool(np.array_equal(a, b))
    print(f"row {i:6d}: token table {np.round(a[:3], 4)} ... output table {np.round(b[:3], 4)} ... identical: {same}")
    res[i] = same
print("every row read is identical:", all(res.values()))
save_json("tied_check", res)
