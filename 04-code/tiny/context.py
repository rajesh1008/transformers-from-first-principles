"""Context-window tools (Chapter 35).

  token_losses(scores, ids)                       loss (nats) of every next token in one pass; scores(ids) -> (n, vocab)
  sliding_losses(scores, ids, window)             each token predicted from at most the previous `window` tokens, positions
                                                  renumbered from zero (one pass per prediction: exact, slow)
  by_band(losses, bands)                          mean loss in each (first, last) band of positions
  rope_periods(head_dim, base=10000.0, scale=1.0) the period in tokens of each rotary pair (positions divided by `scale`)
  unturned_pairs(head_dim, base, length)          how many pairs never complete one turn inside `length` tokens
  untrained_rows(P, P0, tol=0.0)                  first row of a learned position table that never moved from its start
"""
import math
import torch

def token_losses(scores, ids):
    with torch.no_grad():
        lp = torch.log_softmax(scores(ids[:-1]).double(), -1)
    return -lp[torch.arange(len(ids) - 1), torch.as_tensor(ids[1:])]

def sliding_losses(scores, ids, window):
    out = torch.empty(len(ids) - 1, dtype=torch.float64)
    for t in range(len(ids) - 1):
        ctx = ids[max(0, t - window + 1):t + 1]
        with torch.no_grad():
            out[t] = -torch.log_softmax(scores(ctx).double()[-1], -1)[ids[t + 1]]
    return out

def by_band(losses, bands):
    return {f"{a}-{b}": float(losses[a:b + 1].mean()) for a, b in bands if a < len(losses)}

def rope_periods(head_dim, base=10000.0, scale=1.0):
    return [2 * math.pi * scale / base ** (-2 * i / head_dim) for i in range(head_dim // 2)]

def unturned_pairs(head_dim, base, length, scale=1.0):
    return sum(p > length for p in rope_periods(head_dim, base, scale))

def untrained_rows(P, P0, tol=0.0):
    moved = (torch.as_tensor(P) - torch.as_tensor(P0, dtype=torch.as_tensor(P).dtype)).norm(dim=1)
    still = (moved <= tol).nonzero().flatten()
    return int(still[0]) if len(still) else None
