"""Figures for Chapter 48, the capstone.  Signature 10, the capstone dashboard (fig48-01), from 04-code/ch48/outputs/main.jsonl
and the evaluation JSON of the last checkpoint; fig48-02, the learning-rate sweeps at two sizes, from outputs/s10m_*.jsonl and
outputs/s2m_*.jsonl (held-out loss on The Time Machine at the last evaluation of each 400-step run).
Run from the project root: python3 03-figures/src/ch48_figs.py"""
import os, sys, json, glob, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt, numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch48"); os.makedirs(OUT, exist_ok=True); p = lambda n: os.path.join(OUT, n)
O = os.path.join(ROOT, "04-code", "ch48", "outputs")
def rows(f): return [json.loads(l) for l in open(f)][1:]
M = rows(os.path.join(O, "main.jsonl")); tr = [r for r in M if "loss" in r]; ev = [r for r in M if "eval" in r]
last = max(r["step"] for r in tr)
evs = sorted(glob.glob(os.path.join(O, "eval_main_step*.json")), key=lambda f: int(f.split("step")[-1].split(".")[0]))
E = json.load(open(evs[-1])) if evs else None
# 01 Signature 10: the capstone dashboard
fig = plt.figure(figsize=(11, 8.2)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.25)
a = fig.add_subplot(gs[0, 0]); a.plot([r["step"] for r in tr], [r["loss"] for r in tr], color=BLUE, lw=1.6, label="training batch")
a.plot([r["step"] for r in ev], [r["eval"]["heldout"]["loss"] for r in ev], "o-", color=AMBER, ms=4, lw=1.6, label="held out (The Time Machine)")
a.set_title("Loss, nats per token"); a.set_xlabel("step"); a.legend(fontsize=9, frameon=False)
b = fig.add_subplot(gs[0, 1]); b.plot([r["step"] for r in tr], [r["lr"] for r in tr], color=PLUM, lw=1.8)
b.set_title("Learning rate (warm-up, constant, decay)"); b.set_xlabel("step"); b.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
c = fig.add_subplot(gs[1, 0]); c.plot([r["step"] for r in tr], [r["gnorm"] for r in tr], color=VERM, lw=1.4)
c.axhline(1.0, color=GREY, ls="--", lw=1); c.text(last, 1.03, "clip at 1.0", ha="right", va="bottom", fontsize=9, color=GREY)
c.set_title("Gradient norm before clipping"); c.set_xlabel("step")
d = fig.add_subplot(gs[1, 1]); tok = [r["step"] * 4096 / 1e6 for r in ev]
d.plot(tok, [r["eval"]["heldout"]["bpc"] for r in ev], "o-", color=AMBER, ms=4, label="The Time Machine")
d.plot(tok, [r["eval"]["glass"]["bpc"] for r in ev], "s-", color=TEAL, ms=4, label="Through the Looking-Glass")
lo = min(min(r["eval"]["heldout"]["bpc"] for r in ev), 2.0745) - 0.15; d.set_ylim(lo, None)
d.axhline(2.0745, color=NAVY, ls=":", lw=1.2); d.text(0.3, 2.0745 - 0.06, "Tiny on Looking-Glass, 2.0745 (Ch25)", ha="left", va="top", fontsize=9, color=NAVY)
d.set_title("Held-out bits per character against tokens seen"); d.set_xlabel("millions of tokens seen"); d.legend(fontsize=9, frameon=False)
s = fig.add_subplot(gs[2, :]); s.axis("off")
if E:
    t = E["samples"][0]["text"].replace("\n", " ")[:330]
    s.text(0.0, 1.0, f"Sample at step {E['step']}, seed 1, temperature 0.8, from 'the cat sat on the mat':", fontsize=10, color=NAVY, weight="bold", va="top", transform=s.transAxes)
    s.text(0.0, 0.78, "\n".join(textwrap.wrap(t, 130)), fontsize=9, color=SLATE, va="top", family="DejaVu Sans Mono", transform=s.transAxes)
fig.suptitle(f"The capstone dashboard: 10,050,560 learned numbers, steps 0 to {last}", fontsize=14, color=NAVY, weight="bold")
save(fig, p("fig48-01-dashboard.png"))
# 02 the sweeps
fig, ax = plt.subplots(figsize=(8.5, 5))
for pre, lab, col, mk in (("s10m", "10,050,560 numbers", BLUE, "o"), ("s2m", "1,872,640 numbers", AMBER, "s")):
    pts = []
    for f in glob.glob(os.path.join(O, pre + "_*.jsonl")):
        R = rows(f); lr = float(os.path.basename(f)[len(pre) + 1:-6]); e = [r for r in R if "eval" in r]
        if any(r.get("diverged") for r in R): pts.append((lr, None)); continue
        if e and e[-1]["step"] == 400: pts.append((lr, e[-1]["eval"]["heldout"]["loss"]))
    pts.sort(); ok = [(x, y) for x, y in pts if y is not None]
    ax.plot([x for x, _ in ok], [y for _, y in ok], mk + "-", color=col, lw=1.8, ms=7, label=lab)
    for x, y in ok: ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=col)
    for x, y in pts:
        if y is None: ax.axvline(x, color=CORAL, ls="--", lw=1); ax.text(x, ax.get_ylim()[1], "diverged", color=CORAL, fontsize=9, ha="center", va="top")
ax.set_xscale("log"); ax.set_xlabel("peak learning rate"); ax.set_ylabel("held-out loss after 400 steps (nats per token)")
ax.legend(frameon=False); ax.set_title("Learning-rate sweeps at two sizes, 400 steps each, same data order", fontsize=12, color=NAVY)
save(fig, p("fig48-02-sweep.png"))
print("ok", last, evs[-1] if evs else None)
