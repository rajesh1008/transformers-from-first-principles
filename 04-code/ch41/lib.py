"""Chapter 41 helpers (copy of ../ch40/lib.py): a forward pass of Tiny that records every layer's residual rows and every head's attention shares,
and can zero a head's output or overwrite a residual row.  Same arithmetic as tiny/torch_tiny.py (checked in lab40.py)."""
from _common import *
import math, torch
from tiny.eval import load
CH24 = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
CH38 = os.path.join(HERE, "..", "ch38", "outputs", "ckpt", "sft_3e-4_600.pt")
def load_ft():
    tt = load(CH24); sd = torch.load(CH38)
    for k in tt.t: tt.t[k] = sd[k].float().detach()
    return tt
def forward(tt, ids, ablate=(), patch=None, want_att=False, mlp_off=(), naive=(), meanz=None, grab=None):
    """ids: (n,) or (b,n).  Returns dict: resid (list of 13 tensors: rows entering layer 0 ... rows after layer 11, before the final
    norm), att (list of (b,h,n,n) if want_att), logits (b,n,V).  ablate: set of (layer, head) whose output is zeroed.
    patch: (layer, seat, row) overwrites the rows entering that layer at that seat."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    if ids.dim() == 1: ids = ids[None]
    n = ids.shape[-1]; H = tt.config["n_heads"]; t = tt.t
    x = t["E"][ids] + t["P"][:n]; res = []; atts = []
    for l in range(tt.config["n_layers"]):
        if patch is not None and patch[0] == l: x = x.clone(); x[:, patch[1]] = patch[2]
        res.append(x)
        h = tt._norm(x, f"h.{l}.ln_1"); p = f"h.{l}.att."
        sp = lambda T: T.reshape(*T.shape[:-1], H, T.shape[-1] // H).transpose(-3, -2)
        Q, K, V = sp(h @ t[p + "W_q"]), sp(h @ t[p + "W_k"]), sp(h @ t[p + "W_v"])
        s = (Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])).masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), float("-inf"))
        a = torch.softmax(s, -1)
        if want_att: atts.append(a)
        z = a @ V                                                  # (b,h,n,dh)
        for (ll, hh) in ablate:
            if ll == l:
                z = z.clone(); z[:, hh] = 0 if meanz is None else meanz[(ll, hh)]
        if grab is not None: grab[l] = z.detach()
        out = z.transpose(-3, -2).reshape(*z.shape[:-3], n, -1) @ t[p + "W_o"]
        for (ll, hh) in naive:
            if ll == l: out = out.clone(); out[..., hh * (out.shape[-1] // H):(hh + 1) * (out.shape[-1] // H)] = 0
        x = x + out
        f = tt._ffn(tt._norm(x, f"h.{l}.ln_2"), l)
        x = x + (f * 0 if l in mlp_off else f)
    res.append(x)
    xf = tt._norm(x, "ln_f"); logits = xf @ t["E"].T
    return dict(resid=res, att=atts, logits=logits)
