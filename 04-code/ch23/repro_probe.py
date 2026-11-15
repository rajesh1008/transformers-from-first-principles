"""Why two runs of one seeded configuration differ (tick 16 finding).  Trains s1 for 12 steps four times in one
process: twice with MixedTiny (BF16 products) and twice in plain 32-bit, and twice more with BF16 products whose
inputs are copied to fresh contiguous memory first.  Prints the first step at which each pair differs.  One thread."""
import _common
import numpy as np, torch
torch.set_num_threads(1)
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam, warmup_cosine
from tiny.precision import MixedTiny, train_mixed
from tiny import data as D
from _pool import shards
from sizes import make
st = D.ShardStream(shards(), seed=1865)
X = torch.tensor(np.stack([st.next_batch() for _ in range(12)]))
def run(kind):
    m = make("s1")
    tt = MixedTiny.from_numpy(m, compute=torch.bfloat16) if kind == "bf16" else TorchTiny.from_numpy(m, dtype=torch.float32)
    return [L for _, L in train_mixed(tt, lambda s: X[s], 12, Adam(1e-3), warmup_cosine(2, 12, 0.1), mode="mixed")]
def first_diff(a, b):
    d = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return (d[0], max(abs(x - y) for x, y in zip(a, b))) if d else (None, 0.0)
for kind in ("bf16", "fp32"):
    a, b = run(kind), run(kind)
    print(f"{kind}: first differing step {first_diff(a, b)[0]}, largest difference {first_diff(a, b)[1]:.3g}")
import os
print("torch", torch.__version__, "| bf16 matmul backend:", torch.backends.mkldnn.is_available(), "| threads", torch.get_num_threads())
