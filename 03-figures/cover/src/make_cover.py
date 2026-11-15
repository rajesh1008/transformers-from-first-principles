#!/usr/bin/env python3
"""Cover for Transformers from First Principles, drawn in code (no generative imagery).
Matches the author's FDE Handbook cover: cream ground, heavy navy grotesque title with the last line in the
accent colour, plain subtitle, a monospace-labelled architecture diagram with an amber signal path, a rubber
stamp, and the author's name bottom left. Renders at 2x and downsamples to 1600x2560."""
import random, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

S = 2
W, H = 1600, 2560
G = "/usr/share/fonts/truetype/sand-box/google/"
BG, NAVY, RED, AMBER, TEAL = (239,233,221), (27,34,50), (210,80,48), (228,162,26), (21,137,116)
LINE, BOXF, ZONEF, SUB = (88,99,119), (230,223,207), (229,229,217), (42,48,64)
OUT = Path(__file__).resolve().parents[1]

def font(path, size, var=None):
    f = ImageFont.truetype(G + path, int(size * S))
    if var: f.set_variation_by_name(var)
    return f
def s(*v): return [int(round(x * S)) for x in v]

TITLE = lambda z: font("Inter Tight/InterTight-VariableFont_wght.ttf", z, "Black")
INTER = lambda z, v="Regular": font("Inter/Inter-VariableFont_opsz,wght.ttf", z, v)
MONO = lambda z, w="SemiBold": font(f"IBM Plex Mono/IBMPlexMono-{w}.ttf", z)

img = Image.new("RGB", (W * S, H * S), BG)
d = ImageDraw.Draw(img)
X0 = 104

# ---- layout knobs (v2: bigger type, smaller diagram) -------------------
TITLE_LINES = [("Transformers", NAVY), ("from First", NAVY), ("Principles", RED)]   # never hyphenate the title
TITLE_W, TITLE_Y = 1412, 545
SUB_Z, SUB_GAP = 76, 142
AUTH_Z, AUTH_Y = 82, 2450
DK = 0.71                         # diagram scale
DCROP = (0, 1270, 1600, 2410)     # diagram + stamp region, in full-size coordinates
DRIGHT, DTOP = 1528, 1425         # where the scaled outer box's right edge / crop top land

main = img
img = Image.new("RGB", (W * S, H * S), BG)
d = ImageDraw.Draw(img)
# ---- diagram -----------------------------------------------------------
dy = 0
def Y(v): return v
def rrect(box, r, outline=LINE, fill=None, w=3):
    x1, y1, x2, y2 = box
    d.rounded_rectangle(s(x1, Y(y1), x2, Y(y2)), radius=r * S, outline=outline, fill=fill, width=int(w * S))
def label(x, y, t, f, col=LINE, anchor="ls", spacing=0):
    if not spacing:
        d.text((x * S, Y(y) * S), t, font=f, fill=col, anchor=anchor); return
    if anchor[0] == "m":
        tot = sum(d.textlength(c, font=f) for c in t) + spacing * S * (len(t) - 1); x = x - tot / S / 2
    for c in t:
        d.text((x * S, Y(y) * S), c, font=f, fill=col, anchor="l" + anchor[1]); x += d.textlength(c, font=f) / S + spacing
def line(pts, col=LINE, w=3):
    d.line([(px * S, Y(py) * S) for px, py in pts], fill=col, width=int(w * S), joint="curve")
def arrow(x, y, direction, col=LINE, size=13):
    if direction == "r": p = [(x, y), (x - size * 1.3, y - size * .75), (x - size * 1.3, y + size * .75)]
    else: p = [(x, y), (x + size * 1.3, y - size * .75), (x + size * 1.3, y + size * .75)]
    d.polygon([(a * S, Y(b) * S) for a, b in p], fill=col)
def node(x, y, r=8):
    d.ellipse(s(x - r, Y(y) - r, x + r, Y(y) + r), fill=BG, outline=LINE, width=int(2.5 * S))

zl = MONO(24, "SemiBold")
bl = MONO(27, "SemiBold")
OX1, OX2, OY1, OY2 = 243, 1528, 1300, 2295
IX1, IX2, IY1, IY2 = 423, 1478, 1395, 2225
TX1, TX2, TY1, TY2 = 770, 1428, 1490, 2155
rrect((OX1, OY1, OX2, OY2), 22, w=3.2)
rrect((IX1, IY1, IX2, IY2), 22, w=3.2)
rrect((TX1, TY1, TX2, TY2), 18, outline=TEAL, fill=ZONEF, w=3.6)
label(OX1 + 22, OY1 + 52, "DECODER-ONLY TRANSFORMER", zl, spacing=2.2)
label(IX1 + 22, IY1 + 52, "TRANSFORMER BLOCK", zl, spacing=2.2)
label(TX1 + 22, TY1 + 52, "MULTI-HEAD ATTENTION", zl, col=TEAL, spacing=2.2)

LY = 1830
AX1, AX2, AY1, AY2 = 832, 1092, 1764, 1896
# amber signal path
d.rectangle(s(0, Y(LY) - 5, AX1 - 18, Y(LY) + 5), fill=AMBER)
d.polygon([((AX1 - 2) * S, Y(LY) * S), ((AX1 - 30) * S, (Y(LY) - 17) * S), ((AX1 - 30) * S, (Y(LY) + 17) * S)], fill=AMBER)
for cx, t, col in [(OX1, "TOKENS", BG), (IX1, "EMBED", BG), (TX1, "NORM", None)]:
    if col: d.rectangle(s(cx - 6, Y(LY) - 84, cx + 6, Y(LY) + 36), fill=col)
    else:
        d.rectangle(s(cx - 6, Y(LY) - 84, cx, Y(LY) + 36), fill=BG); d.rectangle(s(cx, Y(LY) - 84, cx + 6, Y(LY) + 36), fill=ZONEF)
    d.rounded_rectangle(s(cx - 33, Y(LY) - 21, cx + 33, Y(LY) + 21), radius=4 * S, outline=AMBER, width=int(5 * S))
    label(cx, LY - 44, t, zl, anchor="ms", spacing=1.5)
# attention box
d.rounded_rectangle(s(AX1, Y(AY1), AX2, Y(AY2)), radius=14 * S, fill=AMBER)
def heat(x, y, cell, pattern, col=NAVY, bgc=None, alpha_cols=None):
    n = 4
    for i in range(n):
        for j in range(n):
            if j > i: continue
            v = pattern(i, j)
            if v <= 0: continue
            c = tuple(int(BG[k] + (col[k] - BG[k]) * v) for k in range(3)) if bgc is None else tuple(int(bgc[k] + (col[k] - bgc[k]) * v) for k in range(3))
            d.rectangle(s(x + j * cell, Y(y) + i * cell, x + (j + 1) * cell - 3, Y(y) + (i + 1) * cell - 3), fill=c)
heat(AX1 + 26, (AY1 + AY2) / 2 - 31, 16, lambda i, j: [1, .55, .8, .7][(i + j) % 4], col=NAVY, bgc=AMBER)
label(AX1 + 104, (AY1 + AY2) / 2 + 11, "ATTENTION", bl, col=NAVY)
# right: heads
RX = 1120
BX1, BX2 = 1150, 1412
heads = [(1640, "PREV TOKEN", lambda i, j: 1 if i - j == 1 else (.3 if i == j == 0 else 0)),
         (1830, "INDUCTION", lambda i, j: 1 if i - j == 2 else (.3 if i == j else 0)),
         (2020, "UNIFORM", lambda i, j: 1.0 / (i + 1) * 1.6 if j <= i else 0)]
line([(AX2, LY), (RX, LY)])
line([(RX, 1640), (RX, 2020)])
for cy, t, pat in heads:
    line([(RX, cy), (BX1 - 16, cy)])
    arrow(BX1 - 2, cy, "r")
    rrect((BX1, cy - 40, BX2, cy + 40), 8, fill=BOXF, w=2.8)
    heat(BX1 + 18, cy - 26, 13.5, lambda i, j, p=pat: min(1, p(i, j)), col=TEAL, bgc=BOXF)
    label(BX1 + 80, cy + 10, t, bl, col=NAVY)
# left: MLP and next token
cxm = (AX1 + AX2) / 2
for cy, t, box, icon in [(1640, "MLP", (478, 718), "chip"), (2040, "NEXT TOKEN", (440, 722), "bars")]:
    yy = AY1 if cy < LY else AY2
    line([(cxm, yy), (cxm, cy), (TX1, cy)])
    node(TX1, cy)
    line([(TX1 - 8, cy), (box[1] + 16, cy)])
    arrow(box[1] + 2, cy, "l")
    rrect((box[0], cy - 40, box[1], cy + 40), 8, fill=BOXF, w=2.8)
    ix = box[0] + 22
    if icon == "chip":
        d.rounded_rectangle(s(ix + 6, Y(cy) - 17, ix + 40, Y(cy) + 17), radius=3 * S, outline=NAVY, width=int(3 * S))
        d.rectangle(s(ix + 15, Y(cy) - 8, ix + 31, Y(cy) + 8), fill=NAVY)
        for k in range(4):
            o = -12 + k * 8
            line([(ix + 1, cy + o), (ix + 6, cy + o)], NAVY, 2.4); line([(ix + 40, cy + o), (ix + 45, cy + o)], NAVY, 2.4)
            line([(ix + 23 + o, cy - 22), (ix + 23 + o, cy - 17)], NAVY, 2.4); line([(ix + 23 + o, cy + 17), (ix + 23 + o, cy + 22)], NAVY, 2.4)
    else:
        for k, hgt in enumerate([10, 32, 18, 6, 14]):
            d.rectangle(s(ix + k * 9, Y(cy) + 18 - hgt, ix + k * 9 + 6, Y(cy) + 18), fill=NAVY if k != 1 else TEAL)
        line([(ix - 2, cy + 20), (ix + 46, cy + 20)], NAVY, 2.4)
    label(ix + 62, cy + 10, t, bl, col=NAVY)

# ---- stamp -------------------------------------------------------------
SW, SH = 470, 225
st = Image.new("L", (SW * S, SH * S), 0)
sd = ImageDraw.Draw(st)
sd.rounded_rectangle(s(6, 6, SW - 6, SH - 6), radius=18 * S, outline=255, width=int(9 * S))
sd.rounded_rectangle(s(22, 22, SW - 22, SH - 22), radius=10 * S, outline=255, width=int(3.5 * S))
big = font("Anton/Anton-Regular.ttf", 112)
sd.text((SW / 2 * S, 128 * S), "TRAINED", font=big, fill=255, anchor="ms")
small = MONO(23, "SemiBold")
txt = "FROM SCRATCH"
sp = 7
tot = sum(sd.textlength(c, font=small) for c in txt) / S + sp * (len(txt) - 1)
x = SW / 2 - tot / 2
for c in txt:
    sd.text((x * S, 182 * S), c, font=small, fill=255, anchor="ls"); x += sd.textlength(c, font=small) / S + sp
sd.line(s(54, 174, SW / 2 - tot / 2 - 16, 174), fill=255, width=int(3 * S))
sd.line(s(SW / 2 + tot / 2 + 16, 174, SW - 54, 174), fill=255, width=int(3 * S))
# distress: speckle erosion
rnd = random.Random(1865)
noise = Image.effect_noise((SW * S, SH * S), 90).filter(ImageFilter.GaussianBlur(1.6 * S))
mask = noise.point(lambda v: 0 if v < 92 else 255)
for _ in range(260):
    cx, cy, r = rnd.uniform(0, SW * S), rnd.uniform(0, SH * S), rnd.uniform(1, 4.5) * S
    ImageDraw.Draw(mask).ellipse((cx - r, cy - r, cx + r, cy + r), fill=0)
st = ImageChops.multiply(st, mask)
st = st.point(lambda v: int(v * 0.9))
st = st.rotate(9, resample=Image.BICUBIC, expand=True)
layer = Image.new("RGB", st.size, (206, 70, 50))
px, py = int((1328 - st.size[0] / S / 2) * S), int((Y(2250) - st.size[1] / S / 2) * S)
img.paste(layer, (px, py), st)

# ---- composite the scaled diagram -------------------------------------
dimg = img
cx1, cy1, cx2, cy2 = DCROP
crop = dimg.crop((cx1 * S, cy1 * S, cx2 * S, cy2 * S))
cw, ch = int(crop.size[0] * DK), int(crop.size[1] * DK)
crop = crop.resize((cw, ch), Image.LANCZOS)
ox = int((DRIGHT - (1528 - cx1) * DK) * S)    # 1528 = outer box right edge
oy = int(DTOP * S)
main.paste(crop, (ox, oy))
ly = oy + (LY - cy1) * DK * S
if ox > 0:
    ImageDraw.Draw(main).rectangle((0, ly - 5 * DK * S, ox + 2, ly + 5 * DK * S), fill=AMBER)
img = main
d = ImageDraw.Draw(img)

# ---- title -------------------------------------------------------------
TRACK = -0.035
def tracked(x, y, txt, f, col, size):
    for i, c in enumerate(txt):
        d.text((x * S + d.textlength(txt[:i], font=f) + i * TRACK * size * S, y * S), c, font=f, fill=col, anchor="ls")
def tracked_len(txt, f, size): return d.textlength(txt, font=f) / S + (len(txt) - 1) * TRACK * size
lines = TITLE_LINES
target = TITLE_W
z = 100
while max(tracked_len(l, TITLE(z + 1), z + 1) for l, _ in lines) <= target: z += 1
tf = TITLE(z)
pitch = z * 0.99
y = TITLE_Y
for txt, col in lines:
    tracked(X0 - 6, y, txt, tf, col, z)
    y += pitch
base_last = y - pitch
# ---- subtitle ----------------------------------------------------------
sf = INTER(SUB_Z)
sy = base_last + SUB_GAP
d.text((X0 * S, sy * S), "Build Attention, Step by Step,", font=sf, fill=SUB, anchor="ls")
d.text((X0 * S, (sy + SUB_Z * 1.3) * S), "with Only High-School Algebra", font=sf, fill=SUB, anchor="ls")
TOP = sy + 84

# ---- author ------------------------------------------------------------
af = TITLE(AUTH_Z)
tracked(X0, AUTH_Y, "Rajesh Ranjan Mahapatra", af, NAVY, AUTH_Z)

out = img.resize((W, H), Image.LANCZOS)
out.save(OUT / "Transformers-from-First-Principles-Cover-1600x2560.jpg", quality=95, subsampling=0, optimize=True, dpi=(300, 300))
print("ok", z, base_last, TOP, "diagram y", oy / S, oy / S + ch / S)
