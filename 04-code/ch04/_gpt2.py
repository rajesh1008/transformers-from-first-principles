"""Read tensors from GPT-2 small's public weight file with nothing but NumPy.
A .safetensors file is an 8-byte length, a JSON header naming every tensor, then raw numbers.
The file (548 MB) is downloaded once to ~/.cache/tfp/gpt2 (or $TFP_GPT2_DIR)."""
import os, json, struct, urllib.request
import numpy as np

DIR = os.environ.get("TFP_GPT2_DIR", os.path.expanduser("~/.cache/tfp/gpt2"))
URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
PATH = os.path.join(DIR, "model.safetensors")

def _ensure():
    if not os.path.exists(PATH):
        os.makedirs(DIR, exist_ok=True)
        print(f"downloading GPT-2 small weights (548 MB) to {DIR} ...")
        urllib.request.urlretrieve(URL, PATH)

def header():
    _ensure()
    with open(PATH, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def tensor(name):
    """One tensor by name, e.g. 'wte.weight' (the token embedding table, 50,257 x 768)."""
    h, start = header()
    info = h[name]
    a, b = info["data_offsets"]
    assert info["dtype"] == "F32"
    arr = np.memmap(PATH, dtype="<f4", mode="r", offset=start + a, shape=tuple(info["shape"]))
    return np.array(arr)
