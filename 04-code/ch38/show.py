"""Chapter 38: print a checkpoint's greedy answers to the first few test prompts of each kind (argument: checkpoint file
in outputs/ckpt/).  CPU, one thread, 32-bit."""
from _common import *
import sys, torch
torch.set_num_threads(1)
from tiny.eval import load
T = tok(); EOT = T.specials["<|endoftext|>"]
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); sd = torch.load(os.path.join(HERE, "outputs", "ckpt", sys.argv[1]))
for k in tt.t: tt.t[k] = sd[k]
def answer(prompt, n=24):
    ids = T.encode(f"User: {prompt}\nTiny:"); out = []
    with torch.no_grad():
        for _ in range(n):
            t = int(tt.scores(ids[-128:])[-1].argmax())
            if t == EOT: break
            out.append(t); ids = ids + [t]
    return T.decode(out).strip()
D = load_json("instructions"); seen = {}
for e in D["test"]:
    if seen.get(e["kind"], 0) >= 4: continue
    seen[e["kind"]] = seen.get(e["kind"], 0) + 1
    print(f"{e['kind']:13s} {e['prompt']!r}\n{'':13s} -> {answer(e['prompt'])!r}   (wanted {e['response']!r})", flush=True)
for p in ("What is the capital of France?", "Who wrote Sylvie and Bruno?", "What did the cat do on the mat?"):
    print(f"{'other':13s} {p!r} -> {answer(p)!r}", flush=True)
