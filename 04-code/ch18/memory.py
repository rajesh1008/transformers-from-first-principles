"""Chapter 18, box "Why training needs more memory than inference": what autograd keeps between the forward and
the backward pass.  Tiny (2,766,592 learned numbers) in 32-bit on the novel's first 1,025 tokens (1,024
predictions).  torch.autograd.graph.saved_tensors_hooks counts every tensor the backward pass will read, once per
distinct storage; the same forward pass under torch.no_grad() keeps nothing.  The tables and their nudges are
counted separately.  Nothing random."""
import _common
import torch
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from _common import novel_ids, save_json
m = Tiny(); ids = novel_ids()[:1025]
tt = TorchTiny.from_numpy(m, dtype=torch.float32)
tables = sum(v.numel() * v.element_size() for v in tt.t.values())
seen, sizes = {}, []
def pack(x):
    key = (x.untyped_storage().data_ptr(), x.untyped_storage().nbytes())
    if key not in seen:
        seen[key] = x.untyped_storage().nbytes(); sizes.append((tuple(x.shape), seen[key]))
    return x
with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
    L = tt.loss(ids)
saved = sum(seen.values())
param_keys = {(v.untyped_storage().data_ptr(), v.untyped_storage().nbytes()) for v in tt.t.values()}
saved_acts = sum(b for k, b in seen.items() if k not in param_keys)
n, H = 1024, m.blocks[0].att.n_heads
share_tables = sum(b for s, b in sizes if s[-2:] == (n, n))
count0 = []
with torch.no_grad(), torch.autograd.graph.saved_tensors_hooks(lambda x: count0.append(x.untyped_storage().nbytes()) or x, lambda x: x):
    L0 = tt.loss(ids)
L.backward()
grads = sum(v.grad.numel() * v.grad.element_size() for v in tt.t.values())
print(f"loss {L.item():.4f} (no_grad: {L0.item():.4f})")
print(f"learned tables, 32-bit:                {tables/1e6:7.1f} MB")
print(f"their nudges after backward():         {grads/1e6:7.1f} MB")
print(f"kept for the backward pass (not tables): {saved_acts/1e6:5.1f} MB in {len(seen) - len(param_keys & set(seen))} tensors")
print(f"  of which {n} x {n} tables (shares and masks):{share_tables/1e6:7.1f} MB ({share_tables/saved_acts:.0%})")
print(f"  one {n} x {n} table in 32-bit is {n*n*4/1e6:.1f} MB; {m.n_layers} layers x {H} heads of them is {m.n_layers*H*n*n*4/1e6:.0f} MB")
print(f"kept under torch.no_grad():            {sum(count0)/1e6:7.1f} MB in {len(count0)} tensors")
save_json("memory", dict(tables_mb=tables/1e6, grads_mb=grads/1e6, saved_mb=saved_acts/1e6, share_mb=share_tables/1e6,
                         n_saved=len(seen), loss=L.item()))
