"""Tiny in PyTorch (Chapter 17).  The same arithmetic as tiny/model.py, written with torch tensors so
that PyTorch's autograd can compute the nudge (the gradient) for every learned number.

Nothing here is a new model.  `TorchTiny.from_numpy(m)` copies every table of a NumPy Tiny (any of the
switches of Chapter 15: norm "layer" or "rms", activation "gelu" or "relu", tied or untied, any width,
heads and layers) into torch tensors, in 64-bit by default, and `loss(ids)` computes exactly the
number `Tiny.loss(ids)` computes: the average surprise of each true next token.  `grads(ids)` returns
the gradient of that loss for every table, keyed by the names `Tiny.tables()` uses, as NumPy arrays,
so a script can compare them number for number with a nudge measured by wiggling (tiny/train.py).

`to_numpy(m)` writes the (trained) tensors back into the NumPy Tiny, so the NumPy model stays the
reference and everything the earlier chapters measure can be measured again after training.
Rotary positions, windows, key-value sharing and dropped heads are not mirrored (Tiny's defaults use
none of them)."""
import math
import torch

GELU_C = math.sqrt(2.0 / math.pi)


def _gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(GELU_C * (x + 0.044715 * x ** 3)))   # GPT-2's tanh form, as tiny/ffn.py


ACT = {"gelu": _gelu, "relu": torch.relu}


class TorchTiny:
    def __init__(self, tables, config, dtype=torch.float64):
        self.config = dict(config)
        self.dtype = dtype
        self.t = {k: torch.tensor(v, dtype=dtype, requires_grad=True) for k, v in tables.items()}

    @classmethod
    def from_numpy(cls, m, dtype=torch.float64):
        cfg = dict(n_layers=m.n_layers, n_heads=m.blocks[0].att.n_heads, d_model=m.d_model,
                   norm="rms" if not hasattr(m.blocks[0].ln_1, "bias") else "layer",
                   activation=m.blocks[0].ff.act_name, tied=m.tied, eps=m.blocks[0].ln_1.eps,
                   final_norm=m.ln_f is not None)
        return cls(m.tables(), cfg, dtype=dtype)

    def to_numpy(self, m):
        m.load({k: v.detach().cpu().numpy().astype(float) for k, v in self.t.items()})
        return m

    def parameters(self):
        return list(self.t.values())

    # ---- the forward pass, step for step as tiny/model.py -----------------------------------------
    def _norm(self, x, prefix):
        eps = self.config["eps"]
        if self.config["norm"] == "layer":
            m = x.mean(-1, keepdim=True)
            v = ((x - m) ** 2).mean(-1, keepdim=True)
            return (x - m) / torch.sqrt(v + eps) * self.t[prefix + ".gain"] + self.t[prefix + ".bias"]
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * self.t[prefix + ".gain"]

    def _attention(self, x, l):
        n, d = x.shape[-2], x.shape[-1]
        h = self.config["n_heads"]
        p = f"h.{l}.att."
        def split(T):                                    # (..., n, h * dh) -> (..., h, n, dh)
            return T.reshape(*T.shape[:-1], h, T.shape[-1] // h).transpose(-3, -2)
        Q, K, V = split(x @ self.t[p + "W_q"]), split(x @ self.t[p + "W_k"]), split(x @ self.t[p + "W_v"])
        scores = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        shares = torch.softmax(scores, dim=-1)
        out = (shares @ V).transpose(-3, -2)            # (..., n, h, dh)
        out = out.reshape(*out.shape[:-2], -1)
        return out @ self.t[p + "W_o"]

    def _ffn(self, x, l):
        p = f"h.{l}.ff."
        up = x @ self.t[p + "W_1"] + self.t[p + "b_1"]
        return ACT[self.config["activation"]](up) @ self.t[p + "W_2"] + self.t[p + "b_2"]

    def scores(self, ids):
        """ids: a list, or a (batch, n) LongTensor.  Returns next-token scores (..., n, vocab)."""
        ids = torch.as_tensor(ids, dtype=torch.long)
        n = ids.shape[-1]
        return self.scores_from_rows(self.t["E"][ids] + self.t["P"][:n])

    def scores_from_rows(self, x):
        """Next-token scores from the rows entering layer 0 (token row plus position row at each seat).
        Chapter 17 uses it to ask for the nudge on each seat's row itself."""
        for l in range(self.config["n_layers"]):
            x = x + self._attention(self._norm(x, f"h.{l}.ln_1"), l)
            x = x + self._ffn(self._norm(x, f"h.{l}.ln_2"), l)
        if self.config["final_norm"]:
            x = self._norm(x, "ln_f")
        W = self.t["E"].T if self.config["tied"] else self.t["W_out"]
        return x @ W

    def loss(self, ids):
        """Average surprise (natural log) of each true next token: seats 0 .. n-2 predict 1 .. n-1.
        ids may be one sequence or a batch of equal-length sequences."""
        ids = torch.as_tensor(ids, dtype=torch.long)
        S = self.scores(ids[..., :-1])
        logp = torch.log_softmax(S, dim=-1)
        target = ids[..., 1:]
        return -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()

    def grads(self, ids):
        """The loss and the gradient of the loss for every table, as NumPy arrays by table name."""
        for v in self.t.values():
            v.grad = None
        L = self.loss(ids)
        L.backward()
        return float(L.detach()), {k: v.grad.detach().cpu().numpy().copy() for k, v in self.t.items()}
