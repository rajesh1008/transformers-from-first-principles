"""Distillation and pruning tools for Tiny (Chapter 34).  Tables are Tiny's: linear maps stored [in, out], the token table 'E'
and position table 'P' left alone; feed-forward unit j is column j of W_1, b_1[j] and row j of W_2.

  kd_loss(student_scores, teacher_scores, T=1.0)   T^2 x KL(teacher || student) on tables softened by T, mean over positions
  channel_masks(tables, frac)                      zero the smallest-|w| frac of each output channel's inputs (per column)
  unit_masks(tables, n_layers, frac)               remove the frac of feed-forward units with the smallest |W_1 col| x |W_2 row|
  apply_masks(tables, masks)                       multiply in place; call after every optimiser step, or Adam refills them
  nonzero(tables, masks)                           numbers left after pruning
"""
import torch

def kd_loss(ps, pt, T=1.0):
    ls = torch.log_softmax(ps / T, -1); lt = torch.log_softmax(pt / T, -1)
    return (lt.exp() * (lt - ls)).sum(-1).mean() * T * T

def _linear(k, v):
    return v.dim() == 2 and k not in ("E", "P")

def channel_masks(tables, frac):
    out = {}
    for k, w in tables.items():
        if _linear(k, w):
            idx = w.detach().abs().argsort(0)[:int(w.shape[0] * frac)]
            m = torch.ones_like(w, dtype=w.dtype); m.scatter_(0, idx, 0.0); out[k] = m.detach()
    return out

def unit_masks(tables, n_layers, frac):
    out = {}
    for l in range(n_layers):
        p = f"h.{l}.ff."; W1, W2 = tables[p + "W_1"].detach(), tables[p + "W_2"].detach()
        drop = (W1.norm(dim=0) * W2.norm(dim=1)).argsort()[:int(W1.shape[1] * frac)]
        m1 = torch.ones_like(W1); m1[:, drop] = 0; mb = torch.ones_like(tables[p + "b_1"].detach()); mb[drop] = 0
        m2 = torch.ones_like(W2); m2[drop, :] = 0
        out[p + "W_1"], out[p + "b_1"], out[p + "W_2"] = m1, mb, m2
    return out

@torch.no_grad()
def apply_masks(tables, masks):
    for k, m in masks.items():
        tables[k].mul_(m)

def nonzero(tables, masks):
    return sum(v.numel() for v in tables.values()) - sum(int((m == 0).sum()) for m in masks.values())
