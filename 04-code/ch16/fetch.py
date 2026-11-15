"""Download three public configuration files and the header of each model's weights file.
A .safetensors file begins with 8 bytes giving the header's length, then a JSON header naming
every table with its shape and number format.  Two small range requests read the header without
downloading the weights.  Run once; the results are kept in configs/ (fetched 10 November 2026)."""
import json, os, struct, urllib.request
from _common import CONFIGS, MODELS
UA = {"User-Agent": "Mozilla/5.0 (book research; tfp)"}

def get(url, rng=None):
    h = dict(UA)
    if rng:
        h["Range"] = "bytes=%d-%d" % rng
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
        return r.read()

for name, stem in MODELS.items():
    repo = stem.replace("_", "/", 1)
    base = f"https://huggingface.co/{repo}/resolve/main/"
    cfg = get(base + "config.json")
    open(os.path.join(CONFIGS, stem + ".config.json"), "wb").write(cfg)
    n = struct.unpack("<Q", get(base + "model.safetensors", (0, 7)))[0]
    head = json.loads(get(base + "model.safetensors", (8, 8 + n - 1)))
    meta = head.pop("__metadata__", None)
    tables = {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in head.items()}
    last = max(v["data_offsets"][1] for v in head.values())
    json.dump({"repo": repo, "header_bytes": n, "data_bytes": last, "metadata": meta, "tables": tables},
              open(os.path.join(CONFIGS, stem + ".header.json"), "w"), indent=1)
    print(f"{name:13s} {repo:30s} config {len(cfg):4d} bytes; header {n:,} bytes; {len(tables)} tables; weights data {last:,} bytes")
