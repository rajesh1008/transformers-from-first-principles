"""Figures for Chapter 35, Context Windows and Their Limits.  Numbers are read from 04-code/ch35/outputs/ (run
04-code/ch35/RUN.txt first).  Run from the project root: python3 03-figures/src/ch35_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import torch
setup()
OUT = os.path.join(ROOT, "03-figures", "ch35"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch35", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)

def bands(d, prefix):
    out = []
    for k, v in d.items():
        if k.startswith(prefix) and "-" in k[len(prefix):]:
            a, b = k[len(prefix):].split("-"); out.append(((int(a) + int(b)) / 2, v, int(a), int(b)))
    return sorted(out)

# 01 Tiny past its trained window: one pass of 512 against a sliding window of 128
d = J("tiny_past_edge")
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for pre, col, lab, mk in (("full ", VERM, "one pass of 512 tokens", "o"), ("slide ", BLUE, "sliding window of 128", "s")):
    B = bands(d, pre)
    ax.plot([m for m, *_ in B], [v for _, v, *_ in B], marker=mk, color=col, label=lab)
ax.axvline(128, color=NAVY, ls="--", lw=1)
ax.annotate("position 128: the first row\nTiny never trained", (128, 4.95), xytext=(12, 0), textcoords="offset points", color=NAVY, fontsize=10)
ax.set_xlabel("position of the predicted token in the window (band midpoint)"); ax.set_ylabel("mean loss, nats per token")
ax.set_title("Tiny on Through the Looking-Glass, 12 windows of 512 tokens", fontsize=12)
ax.legend(frameon=False, loc="lower right"); save(fig, p("fig35-01-tiny-edge.png"))

# 02 GPT-2: loss by position inside its 1,024-token window
d = J("gpt2_edge"); B = bands(d, "b ")
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot([max(m, 0.5) for m, *_ in B], [v for _, v, *_ in B], marker="o", color=BLUE)
ax.set_xscale("log", base=2)
for m, v, a, b in B: ax.annotate(f"{v:.2f}", (max(m, 0.5), v), xytext=(4, 6), textcoords="offset points", fontsize=9, color=SLATE)
ax.set_xlabel("position of the predicted token (band midpoint, log scale)"); ax.set_ylabel("mean loss, nats per token")
ax.set_title("GPT-2 small, 51 windows of 1,024 tokens: most of the gain is in the first 128", fontsize=12)
save(fig, p("fig35-02-gpt2-positions.png"))

# 03 SmolLM2 at 16,384 tokens: plain angles against interpolation and raised bases (per-token losses, 512-token bins)
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for mode, col, lab in (("plain", VERM, "plain angles (base 100,000)"), ("pi2", AMBER, "positions / 2"), ("pi4", PLUM, "positions / 4"),
                       ("ntk4", BLUE, "base x 4"), ("ntk16", TEAL, "base x 16")):
    L = torch.load(os.path.join(O, f"smol_long_{mode}_16384.pt")).float()
    n = (len(L) // 512) * 512; v = L[:n].view(-1, 512).mean(1)
    ax.plot([512 * i + 256 for i in range(len(v))], v.tolist(), color=col, label=lab, lw=1.8)
ax.axvline(8192, color=NAVY, ls="--", lw=1)
ax.annotate("8,192: length in the config", (8192, 6.5), xytext=(6, 0), textcoords="offset points", color=NAVY, fontsize=10)
ax.set_xlabel("position in the input (512-token bins)"); ax.set_ylabel("mean loss, nats per token")
ax.set_title("SmolLM2-135M on 16,384 tokens of Through the Looking-Glass", fontsize=12)
ax.legend(frameon=False, loc="upper left", fontsize=9); save(fig, p("fig35-03-smol-16k.png"))

# 04 pass key by depth: log-probability per answer token
fig, ax = plt.subplots(figsize=(7.6, 5.0))
depths = [0.0, 0.25, 0.5, 0.75, 1.0]
for name, col, lab, mk in (("needle_gpt2_256", GREY, "GPT-2, 256 tokens", "o"), ("needle_gpt2_1000", SLATE, "GPT-2, 1,000 tokens", "s"),
                           ("needle_smol_512", TEAL, "SmolLM2, 512 tokens", "^"), ("needle_smol_2048", BLUE, "SmolLM2, 2,048 tokens", "v"),
                           ("needle_smol_4096", NAVY, "SmolLM2, 4,096 tokens", "D")):
    c = J(name)["cells"]
    ax.plot(depths, [c[f"depth {x}"]["logp"] for x in depths], marker=mk, color=col, label=lab)
ax.set_xlabel("depth of the key in the haystack (0 = start, 1 = just before the question)")
ax.set_ylabel("log-probability per answer token (0 is certain)")
ax.set_title("Pass key at five depths; no-key baselines are -9.6 (GPT-2) and -3.4 (SmolLM2)", fontsize=11)
ax.legend(frameon=False, loc="lower left", fontsize=9); save(fig, p("fig35-04-needle.png"))

# 05 cost per token against length
d = J("cost_curve")
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for key, col, lab, mk in (("tiny", TEAL, "Tiny (width 128), one pass", "o"), ("smol", BLUE, "SmolLM2-135M, one pass, blocked attention", "s")):
    xs = sorted(int(k) for k in d[key]); ys = [d[key][str(x)] / x * 1000 for x in xs]
    base = ys[0]; ax.plot(xs, [y / base for y in ys], marker=mk, color=col, label=lab)
    ax.annotate(f"{ys[-1]/base:.1f}x", (xs[-1], ys[-1] / base), xytext=(4, 4), textcoords="offset points", fontsize=10, color=col)
ax.set_xscale("log", base=2)
ax.set_xlabel("tokens in one forward pass (log scale)"); ax.set_ylabel("time per token, relative to 128 tokens")
ax.set_title("Each token costs more when there are more of them", fontsize=12)
ax.legend(frameon=False, loc="upper left"); save(fig, p("fig35-05-cost.png"))
print("ok", sorted(os.listdir(OUT)))
