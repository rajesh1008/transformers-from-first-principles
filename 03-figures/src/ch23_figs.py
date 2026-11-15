"""Figures for Chapter 23, Scaling Laws in Miniature.  Every measured number is read from 04-code/ch23/outputs/*.json;
run the chapter's scripts first (04-code/ch23/RUN.txt).  fig23-02 draws a made-up power law (no measurement; labelled
in the figure).  Run from the project root: python3 03-figures/src/ch23_figs.py"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code")); sys.path.insert(0, os.path.join(ROOT, "04-code", "ch23"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mt
from tiny import scaling as S
def plainy(ax):
    ax.yaxis.set_major_formatter(mt.FormatStrFormatter('%.1f')); ax.yaxis.set_minor_formatter(mt.FormatStrFormatter('%.1f'))

setup()
OUT = os.path.join(ROOT, "03-figures", "ch23"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch23", "outputs")
A = json.load(open(os.path.join(CH, "analyse.json")))
SZ = json.load(open(os.path.join(CH, "sizes.json")))
R = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob(os.path.join(CH, "train_*.json"))}
p = lambda n: os.path.join(OUT, n)
FIVE = ["s1", "s2", "s3", "s4", "s5"]
Ns = {s: SZ[s]["non_embedding"] for s in SZ}

# 01 the sizes: non-embedding and embedding numbers
fig, ax = plt.subplots(figsize=(8.6, 4.2))
names = list(SZ)
ne = [SZ[s]["non_embedding"] / 1e3 for s in names]; em = [SZ[s]["embedding"] / 1e3 for s in names]
lab = [f"{s}\nwidth {SZ[s]['width']}" for s in names]
ax.bar(lab, ne, color=BLUE, label="non-embedding (layers and norms)")
ax.bar(lab, em, bottom=ne, color=AMBER, label="token and position tables")
for i, s in enumerate(names):
    ax.text(i, ne[i] + em[i] + 40, f"{SZ[s]['non_embedding']:,}", ha="center", fontsize=9, color=NAVY)
ax.set_ylabel("learned numbers (thousands)"); ax.legend(loc="upper left")
ax.set_title("Six four-layer Tinys: the non-embedding count roughly doubles each step")
save(fig, p("fig23-01-sizes.png"))

# 02 a power law on ordinary and log-log axes (made-up numbers)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
x = np.logspace(4, 7, 200)
for E, col, lb in ((0.0, BLUE, "L = 12 N^-0.25"), (4.0, VERM, "L = 4 + 12 N^-0.25")):
    a1.plot(x, E + 12 * x ** -0.25, color=col, label=lb); a2.loglog(x, E + 12 * x ** -0.25, color=col, label=lb)
a1.set_xlabel("N"); a1.set_ylabel("L"); a1.set_title("ordinary axes")
a2.set_xlabel("N (log scale)"); a2.set_title("log-log axes"); a2.legend(loc="center right", fontsize=9)
fig.suptitle("A made-up power law: a straight line on log-log axes until a floor bends it flat", y=1.02)
save(fig, p("fig23-02-powerlaw.png"))

# 03 five loss curves (order 1865, lr 1e-3)
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for s, col in zip(FIVE + ["s6"], SERIES):
    r = R.get(f"{s}_384")
    if not r: continue
    c = np.array([(q[0] + 1, q[2]) for q in r["curve"]])
    ax.plot(c[:, 0], c[:, 1], "o-", color=col, ms=4, label=f"{s} ({Ns[s]:,})")
ax.set_xlabel("step (8 windows of 129 tokens each)"); ax.set_ylabel("held-out loss (Carroll)")
ax.set_ylim(4.7, 7.8); ax.legend(title="size (non-embedding)", fontsize=9)
ax.set_title("Same data, same schedule: larger Tinys end lower, in the same order all the way")
save(fig, p("fig23-03-curves.png"))

# 04 the five-point fit with the sixth predicted
fig, ax = plt.subplots(figsize=(8.6, 4.6))
sw = A["lr1e-3"]; rows = sw["rows"]
xx = np.array([Ns[s] for s in FIVE], float)
for s in FIVE:
    v = rows[s]["carroll"]
    ax.plot([Ns[s]] * 3, v, "o", color=GREY, ms=4, alpha=0.7)
    ax.plot(Ns[s], np.mean(v), "o", color=BLUE, ms=8)
xg = np.logspace(np.log10(4e4), np.log10(2.5e6), 100)
ax.plot(xg, S.predict(sw["fit_line"], xg), color=BLUE, lw=1.6, label=f"straight line, alpha {sw['fit_line']['alpha']:.3f}")
ax.plot(xg, S.predict(sw["fit_floor"], xg), color=VERM, lw=1.6, ls="--", label=f"with a floor, alpha {sw['fit_floor']['alpha']:.3f}")
if "s6" in rows:
    ax.plot([Ns["s6"]] * 3, rows["s6"]["carroll"], "o", color=GREY, ms=4, alpha=0.7)
    ax.plot(Ns["s6"], rows["s6"]["mean"], "D", color=TEAL, ms=9, label="s6, held back, measured")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("non-embedding learned numbers N (log scale)"); ax.set_ylabel("held-out loss (log scale)")
plainy(ax); ax.legend(fontsize=9); ax.set_title("The five-point fit: means of three reading orders (grey: each order)")
save(fig, p("fig23-04-fit.png"))

# 05 how alpha moves
fig, ax = plt.subplots(figsize=(8.6, 4.0))
labels, vals, cols = [], [], []
for k in ("3", "4", "5"):
    labels.append(f"{k} points\n(line)"); vals.append(sw["by_points"][k]["line"]["alpha"]); cols.append(BLUE)
for sd in ("1865", "7", "11"):
    labels.append(f"order {sd}\n(line)"); vals.append(sw["by_order"][sd]["line"]["alpha"]); cols.append(AMBER)
labels.append("total count\n(line)"); vals.append(sw["fit_total"]["line"]["alpha"]); cols.append(PLUM)
if "lr3e-3" in A and "fit_line" in A["lr3e-3"]:
    labels.append("peak lr 3e-3\n(line)"); vals.append(A["lr3e-3"]["fit_line"]["alpha"]); cols.append(TEAL)
ax.bar(labels, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("fitted exponent alpha"); ax.tick_params(axis="x", labelsize=8)
ax.set_title("The exponent from the same five sizes, fitted seven ways")
save(fig, p("fig23-05-alpha.png"))

# 06 learning rate against size
fig, ax = plt.subplots(figsize=(8.6, 4.2))
lrs = ["3e-4", "1e-3", "3e-3", "6e-3", "1e-2"]
for s, col in zip(("s1", "s3", "s5"), (BLUE, VERM, TEAL)):
    v = A["lr"][s]
    pts = [(float(l), q) for l, q in zip(lrs, v) if q is not None]
    ax.plot([a for a, _ in pts], [b for _, b in pts], "o-", color=col, label=f"{s} ({Ns[s]:,})")
ax.set_xscale("log"); ax.set_xlabel("peak learning rate (log scale)"); ax.set_ylabel("held-out loss (Carroll), 384 steps")
ax.legend(); ax.set_title("At one pass, every size wanted a larger learning rate than 1e-3")
save(fig, p("fig23-06-lr.png"))

# 07 the data axis for s3
fig, ax = plt.subplots(figsize=(8.6, 4.2))
da = A["data_axis"]
D = np.array([d[1] for d in da], float); L = np.array([d[3] for d in da])
ax.loglog(D, L, "o", color=AMBER, ms=8)
for d in da:
    ax.annotate(f"{d[2]:.2g} pass{'es' if d[2] > 1.01 else ''}", (d[1], d[3]), textcoords="offset points", xytext=(6, 6), fontsize=9)
if "data_line" in A:
    dg = np.logspace(np.log10(D.min()), np.log10(D.max()), 50)
    ax.loglog(dg, S.predict(A["data_line"], dg), color=GREY, ls="--", lw=1.4, label="line through the points of one pass or less")
    ax.legend(fontsize=9)
plainy(ax); ax.set_xlabel("tokens read D (log scale)"); ax.set_ylabel("held-out loss (log scale)")
ax.set_title("One size read for longer: s3 (199,040) from a quarter pass to eight passes")
save(fig, p("fig23-07-data.png"))

# 08 isoFLOP curves
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for key, col, mk in (("192_1e-3", BLUE, "o"), ("768_1e-3", VERM, "o"), ("3072_1e-3", TEAL, "o"), ("768_3e-3", VERM, "s"), ("3072_3e-3", TEAL, "s")):
    if key not in A["isoflop"]: continue
    q = A["isoflop"][key]
    n = [t[1] for t in q["points"]]; l = [t[6] for t in q["points"]]
    ls = "-" if key.endswith("1e-3") else ":"
    ax.plot(n, l, mk + ls, color=col, label=f"C = {q['C']:.1e}, lr {key.split('_')[1]}")
ax.set_xscale("log"); ax.set_xlabel("non-embedding learned numbers N (log scale)"); ax.set_ylabel("held-out loss (Carroll)")
ax.legend(fontsize=8); ax.set_title("Three budgets, each spent five ways: the best size grows with the budget")
save(fig, p("fig23-08-isoflop.png"))

# 09 what 6ND leaves out
fig, ax = plt.subplots(figsize=(8.6, 4.2))
names = [s for s in FIVE + ["s6"] if s in A["time"]]
six = np.array([A["time"][s]["six"] for s in names]) / 1e9; out = np.array([A["time"][s]["out"] for s in names]) / 1e9
att = np.array([A["time"][s]["att"] for s in names]) / 1e9
tot = six + out + att
ax.bar(names, 100 * six / tot, color=BLUE, label="6N (layers)")
ax.bar(names, 100 * out / tot, bottom=100 * six / tot, color=AMBER, label="output table")
ax.bar(names, 100 * att / tot, bottom=100 * (six + out) / tot, color=PLUM, label="attention scores and blend")
ax.set_ylabel("share of one step's arithmetic (%)"); ax.legend(loc="lower right", fontsize=9)
ax.set_title("At small widths the output table costs as much as the layers that 6N counts")
save(fig, p("fig23-09-compute.png"))

# 10 the middle of a long run is not a short run
fig, ax = plt.subplots(figsize=(8.6, 4.2))
lg, sh = R["s3_3072"], R["s3_384"]
c = np.array([(q[0] + 1, q[2]) for q in lg["curve"][1:]]); ax.plot(c[:, 0], c[:, 1], "o-", color=BLUE, label="s3, one 3,072-step run")
c = np.array([(q[0] + 1, q[2]) for q in sh["curve"][1:]]); ax.plot(c[:, 0], c[:, 1], "s-", color=VERM, label="s3, a 384-step run with its own decay")
ax.set_xscale("log"); ax.set_xlabel("step (log scale)"); ax.set_ylabel("held-out loss (Carroll)"); ax.legend()
ax.set_title("A long run's middle is not a short run: they differ at step 384")
save(fig, p("fig23-10-midrun.png"))

# 11 too few steps: the point that falls off the line
fig, ax = plt.subplots(figsize=(8.6, 4.2))
for s in FIVE:
    ax.plot(Ns[s], rows[s]["mean"], "o", color=BLUE, ms=8)
ax.plot(xg, S.predict(sw["fit_line"], xg), color=BLUE, lw=1.4)
for st, mk in (("48", "v"), ("96", "^")):
    if st in A["short"]:
        ax.plot(Ns["s4"], A["short"][st], mk, color=CORAL, ms=10, label=f"s4 stopped at {st} steps")
ax.set_xscale("log"); ax.set_yscale("log"); plainy(ax); ax.legend(fontsize=9)
ax.set_xlabel("non-embedding learned numbers N (log scale)"); ax.set_ylabel("held-out loss (log scale)")
ax.set_title("Breakage: one size trained for too few steps falls off the line")
save(fig, p("fig23-11-short.png"))
print("figures written to", OUT)
