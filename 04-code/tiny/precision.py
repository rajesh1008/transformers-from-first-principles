"""Number formats and mixed-precision training for Tiny (Chapter 21).

Nothing here changes Tiny or the modules of Chapters 14 to 20.  The module adds:

  fmt_info(name)              the bits, range and spacing of FP32, BF16, FP16, FP8 (E4M3, E5M2) and FP4 (E2M1)
  round_to(x, dtype)          round a 32-bit tensor to a 16- or 8-bit format and back (nearest)
  round_bf16_stochastic(x, g) round a 32-bit tensor to BF16 up or down at random, in proportion to the distance
  mx_round(x, block, fmt)     block scaling in the style of the microscaling (MX) formats: one power-of-two scale
                              per block of `block` numbers, each number stored in an 8- or 4-bit format
  MixedTiny                   TorchTiny whose table products run in a 16-bit format while the master tables,
                              the residual stream, the norms, the softmax and the loss stay in 32-bit
  LossScaler                  dynamic loss scaling: multiply the loss by S, divide the nudges by S, skip the
                              step and halve S if any nudge is not finite, double S after `interval` good steps
  train_mixed(...)            Chapter 20's train_opt with a precision mode:
                                "fp32"          everything 32-bit (Chapter 20)
                                "mixed"         MixedTiny in bf16 or fp16, 32-bit master tables and optimiser
                                "weights16"     every table stored in BF16, the whole forward and backward in BF16,
                                                no 32-bit master copy; the optimiser's step is computed in 32-bit
                                                and rounded back to BF16, to nearest or stochastically
Seed 1865 for the stochastic rounding generator unless another is given."""
import math
import numpy as np
import torch
from tiny.torch_tiny import TorchTiny, ACT

# ---- formats -----------------------------------------------------------------------------------------
FORMATS = {                     # name: (exponent bits, mantissa bits, torch dtype or None, has inf)
    "FP32": (8, 23, torch.float32, True),
    "BF16": (8, 7, torch.bfloat16, True),
    "FP16": (5, 10, torch.float16, True),
    "FP8 E5M2": (5, 2, torch.float8_e5m2, True),
    "FP8 E4M3": (4, 3, torch.float8_e4m3fn, False),
    "FP4 E2M1": (2, 1, None, False),
}


def fmt_info(name):
    """Largest finite number, smallest normal, smallest subnormal, and the gap above 1 (machine epsilon)."""
    e, m, dt, _ = FORMATS[name]
    if dt is not None:
        fi = torch.finfo(dt)
        return dict(bits=1 + e + m, exp=e, man=m, max=float(fi.max), min_normal=float(fi.smallest_normal),
                    min_sub=float(fi.smallest_normal) * 2.0 ** -m, eps=float(fi.eps))
    bias = 2 ** (e - 1) - 1                      # FP4 E2M1: values 0, 0.5, 1, 1.5, 2, 3, 4, 6
    return dict(bits=1 + e + m, exp=e, man=m, max=6.0, min_normal=2.0 ** (1 - bias), min_sub=0.5, eps=0.5)


FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def round_fp4(x):
    """Round to the nearest FP4 (E2M1) value, keeping the sign; ties to the lower magnitude."""
    a = x.abs().clamp(max=6.0)
    idx = (a[..., None] - FP4_GRID).abs().argmin(-1)
    return torch.sign(x) * FP4_GRID[idx]


def round_to(x, name):
    """Round a 32-bit tensor to the format `name` and return it as 32-bit again."""
    dt = FORMATS[name][2]
    if name == "FP32":
        return x.clone()
    if dt is None:
        return round_fp4(x)
    if name == "FP8 E4M3":                        # E4M3 has no infinity: saturate at 448 as the scaled recipes do
        x = x.clamp(-448.0, 448.0)
    return x.to(dt).to(torch.float32)


def round_bf16_stochastic(x, gen):
    """Round 32-bit x to BF16, up or down at random so that the expected result is x.  BF16 is the top 16 bits
    of FP32, so adding a random number below 2^16 to the bit pattern and cutting the low 16 bits rounds up with
    probability equal to the distance already travelled towards the next BF16 number."""
    bits = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, x.shape, generator=gen, dtype=torch.int32)
    out = ((bits + r) & ~0xFFFF).view(torch.float32)
    return out.to(torch.bfloat16)


def scaled_round(x, name, amax=None):
    """Per-tensor scaling for an 8- or 4-bit format: divide by a scale that maps the largest |x| to the
    format's largest number, round, multiply back."""
    top = fmt_info(name)["max"]
    amax = float(x.abs().max()) if amax is None else amax
    s = amax / top if amax > 0 else 1.0
    return round_to(x / s, name) * s


def mx_round(x, block=32, name="FP8 E4M3"):
    """Block scaling in the MX style: along the last axis, each block of `block` numbers shares one
    power-of-two scale chosen from the block's largest |x| (2 ^ (floor(log2 amax) - emax of the format))."""
    shp = x.shape
    xb = x.reshape(-1, block)
    amax = xb.abs().max(-1, keepdim=True).values.clamp_min(1e-30)
    emax = {"FP8 E4M3": 8, "FP8 E5M2": 15, "FP4 E2M1": 2}[name]
    scale = torch.exp2(torch.floor(torch.log2(amax)) - emax)
    return (round_to(xb / scale, name) * scale).reshape(shp)


# ---- a Tiny whose products run in 16-bit ---------------------------------------------------------------
class MixedTiny(TorchTiny):
    """The tables (self.t) are 32-bit and receive 32-bit nudges.  Every product of two tables, x @ W, the
    scores Q K^T, the blend shares @ V and the output scores, is computed with both inputs rounded to
    `compute` (torch.bfloat16 or torch.float16) and its result returned as 32-bit.  Norms, GELU, softmax,
    the residual stream and the loss stay in 32-bit.  This is what mixed precision means in this book."""
    def __init__(self, tables, config, compute=torch.bfloat16):
        super().__init__(tables, config, dtype=torch.float32)
        self.compute = compute

    @classmethod
    def from_numpy(cls, m, compute=torch.bfloat16):
        base = TorchTiny.from_numpy(m, dtype=torch.float32)
        obj = cls.__new__(cls)
        obj.config, obj.dtype, obj.t, obj.compute = base.config, torch.float32, base.t, compute
        return obj

    def mm(self, a, b):
        return (a.to(self.compute) @ b.to(self.compute)).to(torch.float32)

    def _attention(self, x, l):
        h = self.config["n_heads"]
        n = x.shape[-2]
        p = f"h.{l}.att."
        def split(T):
            return T.reshape(*T.shape[:-1], h, T.shape[-1] // h).transpose(-3, -2)
        Q, K, V = (split(self.mm(x, self.t[p + k])) for k in ("W_q", "W_k", "W_v"))
        scores = self.mm(Q, K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
        shares = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        out = self.mm(shares, V).transpose(-3, -2)
        out = out.reshape(*out.shape[:-2], -1)
        return self.mm(out, self.t[p + "W_o"])

    def _ffn(self, x, l):
        p = f"h.{l}.ff."
        up = self.mm(x, self.t[p + "W_1"]) + self.t[p + "b_1"]
        return self.mm(ACT[self.config["activation"]](up), self.t[p + "W_2"]) + self.t[p + "b_2"]

    def scores_from_rows(self, x):
        for l in range(self.config["n_layers"]):
            x = x + self._attention(self._norm(x, f"h.{l}.ln_1"), l)
            x = x + self._ffn(self._norm(x, f"h.{l}.ln_2"), l)
        if self.config["final_norm"]:
            x = self._norm(x, "ln_f")
        W = self.t["E"].T if self.config["tied"] else self.t["W_out"]
        return self.mm(x, W)


class LossScaler:
    def __init__(self, scale=2.0 ** 16, interval=100, factor=2.0):
        self.scale, self.interval, self.factor, self.good, self.skipped, self.history = scale, interval, factor, 0, 0, []

    def unscale_ok(self, grads):
        """Divide every nudge by the scale; return False (skip the step, halve the scale) if any is not finite."""
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        if not finite:
            self.scale /= self.factor
            self.good, self.skipped = 0, self.skipped + 1
        else:
            for g in grads:
                g.div_(self.scale)
            self.good += 1
            if self.good % self.interval == 0:
                self.scale *= self.factor
        self.history.append(self.scale)
        return finite


class _Shadow:
    """A 32-bit view of BF16 tables for the optimiser: fresh 32-bit tensors each step, holding .grad."""
    def __init__(self, t16):
        self.t = {k: v.detach().to(torch.float32) for k, v in t16.items()}
        for k, v in t16.items():
            self.t[k].grad = v.grad.to(torch.float32)


def train_mixed(tt, batches, steps, opt, schedule=None, mode="fp32", scaler=None, stochastic=False, gen=None,
                on_step=None, stats=None):
    """Returns [(step, loss)] (the loss as computed by the run, unscaled).  `tt` is a TorchTiny (fp32 or
    bf16 tables) or a MixedTiny.  For mode "weights16", `stats` (a list) receives, per step, the fraction of
    numbers whose 32-bit step was not zero but whose stored BF16 value did not change (lost updates)."""
    from tiny.optim import constant
    schedule = schedule or constant()
    history = []
    for s in range(steps):
        for v in tt.t.values():
            v.grad = None
        L = tt.loss(batches(s))
        Lf = float(L.detach().float())
        history.append((s, Lf))
        if not math.isfinite(Lf):
            break
        (L * scaler.scale if scaler else L).backward()
        if mode == "weights16":
            sh = _Shadow(tt.t)
            before = {k: v.clone() for k, v in sh.t.items()}
            opt.step(sh.t, schedule(s))
            moved = lost = 0
            with torch.no_grad():
                for k, v16 in tt.t.items():
                    new = round_bf16_stochastic(sh.t[k], gen) if stochastic else sh.t[k].to(torch.bfloat16)
                    stepped = sh.t[k] != before[k]
                    moved += int(stepped.sum())
                    lost += int((stepped & (new == v16.detach())).sum())
                    v16.copy_(new)
            if stats is not None:
                stats.append(lost / max(1, moved))
        else:
            if scaler and not scaler.unscale_ok([v.grad for v in tt.t.values()]):
                if on_step:
                    on_step(s, Lf)
                continue
            opt.step(tt.t, schedule(s))
        if on_step:
            on_step(s, Lf)
    return history


from tiny.optim import Adam as _Adam


class Adam16(_Adam):
    """Adam whose running averages m and v are stored in a 16-bit format (BF16 by default) between steps;
    the arithmetic of each step is 32-bit.  Halves Adam's state."""
    def __init__(self, *a, state_dtype=torch.bfloat16, **k):
        super().__init__(*a, **k)
        self.sdt = state_dtype

    def update(self, name, g):
        m = self.m.get(name)
        if m is None:
            self.m[name] = torch.zeros_like(g, dtype=self.sdt)
            self.v[name] = torch.zeros_like(g, dtype=self.sdt)
        m, v = self.m[name].to(torch.float32), self.v[name].to(torch.float32)
        m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        self.m[name], self.v[name] = m.to(self.sdt), v.to(self.sdt)
        m, v = self.m[name].to(torch.float32), self.v[name].to(torch.float32)
        mh, vh = (m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)) if self.bc else (m, v)
        return mh / (vh.sqrt() + self.eps)
