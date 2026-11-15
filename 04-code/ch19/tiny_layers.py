"""Chapter 19, section 19.6: watching Tiny's activations layer by layer under five starts.
For each start: one forward and backward pass (64-bit) on eight windows of 129 tokens of the novel (seed 1865),
reporting the spread of the stream entering each layer, what each branch adds, the nudges, attention shares at
layer 0, and the loss.  Then the running sentence's rows at layer 1 and layer 6 under each start."""
import _common
import numpy as np
from tiny.model import Tiny
from tiny.init import initialise, spreads
from _common import novel_ids, sentence_ids, tok, save_json, SEED, probe, rms
T = tok(); ids = sentence_ids(T)
nov = np.array(novel_ids(T))
starts = np.random.default_rng(SEED).integers(0, len(nov) - 129, size=8)
B = np.stack([nov[a:a + 129] for a in starts])
STARTS = [("gpt2", 1.0, "0.02 (Tiny's default)"), ("scaled", 1.0, "0.02, W_o and W_2 / sqrt(24)"),
          ("fan_in", 1.0, "1/sqrt(fan-in)"), ("fan_in_scaled", 1.0, "1/sqrt(fan-in), residual / sqrt(24)"),
          ("gpt2", 10.0, "0.2 (ten times too large)")]
out = {}
print(f"Eight windows of 129 tokens; uniform attention entropy at seat t is ln(t+1), average over seats 1-127 "
      f"{np.mean([np.log(t + 1) for t in range(1, 128)]):.3f}; ln 2048 = {np.log(2048):.4f}")
for scheme, scale, desc in STARTS:
    m = initialise(Tiny(), scheme, scale)
    sp = spreads(m)
    P = probe(m, B)
    s = P["stream"]
    print(f"\n{desc}   [W_q {sp['att.W_q']:.4f}, W_o {sp['att.W_o']:.4f}, W_1 {sp['ff.W_1']:.4f}, W_2 {sp['ff.W_2']:.4f}]")
    print(f"  loss {P['loss']:.4f}; spread of the final scores {P['scores']:.3f}")
    print("  stream entering layers 0, 1, 2, 6, 12(out): " + ", ".join(f"{s[k]:.4f}" for k in (0, 1, 2, 6, 12)))
    print("  attention adds at layers 0, 5, 11:          " + ", ".join(f"{P['att'][k]:.4g}" for k in (0, 5, 11)))
    print("  feed-forward adds at layers 0, 5, 11:       " + ", ".join(f"{P['ff'][k]:.4g}" for k in (0, 5, 11)))
    print(f"  layer-0 attention: average entropy {P['entropy'][0]:.3f}, average largest share {P['share_max'][0]:.3f}")
    print("  nudge on the stream at layers 0, 6, 12:     " + ", ".join(f"{P['nudge'][k]:.3g}" for k in (0, 6, 12)))
    print("  nudge spread on W_o at layers 0, 11:        " + ", ".join(f"{P['grad'][k]['att.W_o']:.3g}" for k in (0, 11))
          + f";  on W_1 at layers 0, 11: " + ", ".join(f"{P['grad'][k]['ff.W_1']:.3g}" for k in (0, 11)))
    Ps = probe(m, ids)
    r0, r1, r6 = Ps["rows"][0][0], Ps["rows"][1][0], Ps["rows"][6][0]
    seat = 1                                   # 'cat'
    ch1 = rms(r1[seat] - r0[seat]) / rms(r0[seat]); ch6 = rms(r6[seat] - r0[seat]) / rms(r0[seat])
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    print(f"  running sentence, seat 1 ('cat'): spread {rms(r0[seat]):.4f} entering layer 0, {rms(r1[seat]):.4f} at layer 1,"
          f" {rms(r6[seat]):.4f} at layer 6; changed by {ch1:.3f} and {ch6:.3f} of its size;"
          f" first three numbers at layer 6 {', '.join(f'{v:.4f}' for v in r6[seat][:3])}")
    print(f"  the two 'the' rows (seats 0 and 4) at layer 6: cosine {cos(r6[0], r6[4]):.3f} (entering layer 0: {cos(r0[0], r0[4]):.3f})")
    out[f"{scheme}x{scale:g}"] = dict(desc=desc, spreads=sp, loss=P["loss"], scores=P["scores"], stream=s, att=P["att"],
                                      ff=P["ff"], nudge=P["nudge"], entropy=P["entropy"], share_max=P["share_max"],
                                      grad=P["grad"], sentence=dict(l0=float(rms(r0[seat])), l1=float(rms(r1[seat])),
                                      l6=float(rms(r6[seat])), ch1=float(ch1), ch6=float(ch6)))
save_json("tiny_layers", out)
