"""tiny/quant.py (Chapter 33): integer quantization of weights and activations, by rounding and by GPTQ.

  quantize(w, bits, group=None, axis=0, sym=True)
        round-to-nearest.  w is [in, out] (GPT-2's Conv1D layout).  group=None: one scale per output column over the
        whole input dimension ("per channel"); group='tensor': one scale for the whole table; group=g: one scale per
        output column per g input rows.  sym=False adds a zero point (asymmetric).  Returns the dequantized table.
  storage_bits(shape, bits, group, sym)   bits to store a table: bits x numbers + 16 per scale (+ bits per zero point)
  quant_act(x, bits=8, mode='tensor'|'token', keep=None)
        fake-quantize activations [T, d]: one scale per call ('tensor') or per row ('token'); keep = channel indices
        left in float (LLM.int8()'s outlier split)
  gptq(w, H, bits, group, damp=0.01)
        GPTQ (Frantar et al., 2022): quantize one input row at a time and spread its rounding error over the rows not
        yet quantized, weighted by the inverse of H = X^T X from calibration inputs.  w [in, out].  Returns dequantized.
  LINEARS                                  the four maps per GPT-2 layer
  forward_q(m, ids, W=None, act=None)      GPT2 forward with replaced tables W (dict, missing names = m.W) and an
                                           optional activation function act(x, name) applied to every linear map's input
  collect_inputs(m, windows)               calibration: X^T X for every linear map's input over the windows, and the
                                           largest |value| per input channel"""
import math
import torch

LINEARS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


def _qmax(bits, sym):
    return (2 ** (bits - 1) - 1) if sym else (2 ** bits - 1)


def _rtn(w, bits, sym, dim):
    if sym:
        s = w.abs().amax(dim, keepdim=True).clamp_min(1e-12) / _qmax(bits, True)
        return (w / s).round().clamp(-_qmax(bits, True) - 1, _qmax(bits, True)) * s
    lo, hi = w.amin(dim, keepdim=True), w.amax(dim, keepdim=True)
    s = (hi - lo).clamp_min(1e-12) / _qmax(bits, False); z = (-lo / s).round()
    return ((w / s).round() + z).clamp(0, _qmax(bits, False)).sub(z) * s


def quantize(w, bits, group=None, sym=True):
    if group == "tensor":
        return _rtn(w.reshape(1, -1), bits, sym, 1).reshape(w.shape)
    if group is None:
        return _rtn(w, bits, sym, 0)
    i, o = w.shape
    return _rtn(w.reshape(i // group, group, o), bits, sym, 1).reshape(i, o)


def storage_bits(shape, bits, group=None, sym=True):
    i, o = shape
    scales = 1 if group == "tensor" else (o if group is None else o * (i // group))
    return bits * i * o + scales * (16 + (0 if sym else bits))


def quant_act(x, bits=8, mode="tensor", keep=None):
    if keep is not None and len(keep):
        y = quant_act(x, bits, mode); y[:, keep] = x[:, keep]; return y
    q = _qmax(bits, True)
    s = (x.abs().amax() if mode == "tensor" else x.abs().amax(-1, keepdim=True)).clamp_min(1e-12) / q
    return (x / s).round().clamp(-q - 1, q) * s


def gptq(w, H, bits, group=128, damp=0.01, block=128):
    W = w.t().clone().double()                      # [out, in]: quantize input columns left to right
    H = H.double().clone(); n = H.shape[0]
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(n, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W); q = _qmax(bits, True)
    for b in range(0, n, block):                    # lazy batches: exact, the rest of W is updated once per block
        e = min(b + block, n); W1 = W[:, b:e].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[b:e, b:e]
        for i in range(e - b):
            if (b + i) % group == 0:
                s = (W1[:, i:i + group] if i + group <= e - b else W[:, b + i:b + i + group]).abs().amax(1).clamp_min(1e-12) / q
            wi = W1[:, i]
            qi = (wi / s).round().clamp(-q - 1, q) * s
            Q[:, b + i] = qi
            err = (wi - qi) / Hi[i, i]
            W1[:, i + 1:] -= err[:, None] * Hi[i, i + 1:][None, :]
            E1[:, i] = err
        W[:, e:] -= E1 @ Hinv[b:e, e:]
    return Q.t().float().contiguous()


def forward_q(m, ids, W=None, act=None):
    W0 = m.W; Wd = dict(W0); Wd.update(W or {})
    H, d = m.H, m.d; T = len(ids); hd = d // H
    A = act or (lambda x, name: x)
    x = Wd["wte.weight"][torch.as_tensor(ids)] + Wd["wpe.weight"][:T]
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    for l in range(m.L):
        p = f"h.{l}."
        h = A(m._ln(x, p + "ln_1"), p + "attn.c_attn")
        q, k, v = (h @ Wd[p + "attn.c_attn.weight"] + Wd[p + "attn.c_attn.bias"]).split(d, -1)
        q, k, v = (z.view(T, H, hd).transpose(0, 1) for z in (q, k, v))
        s = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~mask, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
        x = x + A(o, p + "attn.c_proj") @ Wd[p + "attn.c_proj.weight"] + Wd[p + "attn.c_proj.bias"]
        f = m._gelu(A(m._ln(x, p + "ln_2"), p + "mlp.c_fc") @ Wd[p + "mlp.c_fc.weight"] + Wd[p + "mlp.c_fc.bias"])
        x = x + A(f, p + "mlp.c_proj") @ Wd[p + "mlp.c_proj.weight"] + Wd[p + "mlp.c_proj.bias"]
    return m._ln(x, "ln_f") @ Wd["wte.weight"].T


def collect_inputs(m, windows):
    XtX, amax = {}, {}
    def rec(x, name):
        xd = x.double()
        XtX[name] = XtX.get(name, 0) + xd.t() @ xd
        a = x.abs().amax(0); amax[name] = torch.maximum(amax[name], a) if name in amax else a
        return x
    with torch.no_grad():
        for w in windows:
            forward_q(m, w, act=rec)
    return XtX, amax
