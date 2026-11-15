"""Lab 40: checks for Chapter 40.  Reads the saved outputs of probe.py, facts.py, facts2.py, heads.py, occl.py and re-runs the cheap exact checks.
No model call; the key file is read only to check that the key appears in no file under 04-code/ch40.  CPU, one thread, 32-bit, seed 1865."""
from lib import *
import numpy as np, glob
torch.set_num_threads(1)
ok = 0; n = 0
def check(msg, cond):
    global ok, n
    n += 1; ok += bool(cond); print(("[pass] " if cond else "[FAIL] ") + msg)
tt = load(CH24); ids = torch.tensor([[5, 77, 300, 12, 900, 14, 15, 1600, 42, 7]])
d = float((forward(tt, ids)["logits"] - tt.scores(ids)).abs().max()); print(f"  recording forward pass against tiny/torch_tiny.py: largest score difference {d:.1e}")
check("the recording forward pass equals the book's forward pass", d == 0.0)
sav = {k: tt.t[f"h.0.att.W_o"].clone() for k in ["o"]}
a = forward(tt, ids, ablate={(0, h) for h in range(4)})["logits"]
tt.t["h.0.att.W_o"] = torch.zeros_like(tt.t["h.0.att.W_o"]); b = tt.scores(ids); tt.t["h.0.att.W_o"] = sav["o"]
d = float((a - b).abs().max()); print(f"  zeroing all four heads before W_o against zeroing W_o: largest difference {d:.1e}")
check("zeroing all four heads before the output matrix is the same as removing the layer's attention", d < 1e-4)
one = forward(tt, ids, ablate={(0, 1)})["logits"]; nv = forward(tt, ids, naive={(0, 1)})["logits"]
d = float((one - nv).abs().max()); print(f"  head 0.1 zeroed before W_o against a channel slice zeroed after W_o: largest score difference {d:.3f}")
check("zeroing a channel slice after W_o is not the same as removing the head", d > 0.01)
x = forward(tt, ids)["resid"][5][0, 4]; same = forward(tt, ids, patch=(5, 4, x))["logits"]
d = float((same - forward(tt, ids)["logits"]).abs().max()); print(f"  patching a row with itself: largest difference {d:.1e}")
check("a patch with the row's own value changes nothing", d == 0.0)
R = load_json("probe")
print(f"  SPACE trained layer 0 {R['trained']['SPACE'][0]:.3f}, its control {R['trained']['CTL_SPACE'][0]:.3f}; untrained layer 12 {R['untrained']['SPACE'][12]:.3f}, its control {R['untrained']['CTL_SPACE'][12]:.3f}")
check("the untrained model's word-start probe is almost as good on random labels (selectivity under 0.05 at every layer)", max(a - b for a, b in zip(R["untrained"]["SPACE"], R["untrained"]["CTL_SPACE"])) < 0.05)
check("the trained model's word-start probe is selective (at least 0.2 above its control at every layer)", min(a - b for a, b in zip(R["trained"]["SPACE"], R["trained"]["CTL_SPACE"])) > 0.2)
F = load_json("facts")
print(f"  verb probe, base at the site: best layer {max(F['probe_site_base']):.2f}; fine-tuned: layer 11 {F['probe_site_fine-tuned'][11]:.2f}, layer 12 {F['probe_site_fine-tuned'][12]:.2f}; majority verb 0.233")
check("a verb probe on the base model never beats the majority verb by more than a fact's worth (0.067), and on the fine-tuned model reaches 0.9", max(F["probe_site_base"]) <= 0.233 + 0.067 and F["probe_site_fine-tuned"][11] >= 0.9)
check("the fine-tuned model puts the taught verb first for all 30 animals at the site", len(F["known"]) == 30)
check("head attention and ablation drop are not rank-correlated (|rho| under 0.3) over the 48 heads", abs(F["rho"]) < 0.3)
F2 = load_json("facts2"); k = F2["khead"]
print(f"  4 heads by attention: drop {k['4']['by attention'][0]:.2f}; 4 by ablation {k['4']['by ablation'][0]:.2f}; 4 random {k['4']['random (mean of 5)'][0]:.2f}")
check("four heads chosen by attention lower log p(verb) like random heads do, four chosen by ablation lower it more than ten times as much", k["4"]["by ablation"][0] > 10 * k["4"]["by attention"][0] and abs(k["4"]["by attention"][0] - k["4"]["random (mean of 5)"][0]) < 0.5)
check("patching the site's row at layer 9 makes the donor's verb the prediction in all 200 pairs", F2["patch"][9][0] == 1.0)
key = ""
for line in open(os.path.join(HERE, "..", "..", ".env")):
    if "=" in line and "ANTHROPIC" in line.split("=")[0].upper(): key = line.split("=", 1)[1].strip().strip('"').strip("'")
found = [f for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f) and key and key.encode() in open(f, "rb").read()]
check("the API key was read from .env and its string appears in no file under 04-code/ch40", key != "" and not found)
print(f"\n{ok} of {n} checkpoints pass")
