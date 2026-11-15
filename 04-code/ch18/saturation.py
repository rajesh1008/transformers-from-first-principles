"""Chapter 18, section 18.10: when the shares harden, the nudges through the softmax fade.  The@4's row of five
scores from the hand Tiny is multiplied by a sharpness factor (1 to 1,024), as a larger query table would do, and
the nudge arriving at its shares (dP, from backward_head.py) is held fixed.  For each factor: the spread of the
shares (entropy in nats), the largest share, and the size of the nudge that gets through to the scores.
(The hand Tiny's sat@2 row has two equal scores, which no sharpening separates, so the@4's row is used.)
64-bit; nothing random."""
import _common
import numpy as np
from tiny.attention import softmax, softmax_backward
from _common import load_json, save_json, row
rc = load_json("recap"); j = load_json("backward_head")
s = np.array(rc["S"][4]); dP = np.array(j["g"]["P"][4])
print(f"the@4's scores {row(s)}; nudge arriving at its shares {row(dP, 5)}")
print(f"{'factor':>6} {'spread':>8} {'top share':>10} {'size of dS':>11} {'largest slope':>14}")
rows = []
for a in (1, 4, 16, 64, 256, 1024):
    p = softmax(a * s); dS = softmax_backward(p, dP)
    H = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    J = np.diag(p) - np.outer(p, p)
    r = dict(factor=a, spread=H, top=float(p.max()), dS=float(np.linalg.norm(dS)), slope=float(np.abs(J).max()))
    rows.append(r)
    print(f"{a:>6} {H:>8.4f} {r['top']:>10.4f} {r['dS']:>11.2e} {r['slope']:>14.2e}")
print(f"most even possible spread for five seats: ln 5 = {np.log(5):.4f}")
print(f"from factor 1 to 1,024 the nudge reaching the scores shrinks {rows[0]["dS"]/rows[-1]["dS"]:,.0f}-fold")
save_json("saturation", rows)
