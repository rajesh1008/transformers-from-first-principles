"""Chapter 48, the portfolio record: fingerprint each saved checkpoint of the main run with Chapter 26's tiny.seed.fingerprint
(SHA-256 over every table in sorted name order, first 16 hex digits) and record the run card.  -> outputs/fp48.json"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import torch
from tiny.seed import fingerprint, run_card
R = dict(run_card=run_card(), checkpoints={})
for f in sorted(glob.glob(os.path.join(HERE, "ckpt", "main_step*.pt")), key=lambda f: int(f.split("step")[-1][:-3])):
    ck = torch.load(f, map_location="cpu"); R["checkpoints"][os.path.basename(f)] = dict(step=ck["step"], fingerprint=fingerprint(ck["state"]), bytes=os.path.getsize(f))
    print(os.path.basename(f), ck["step"], R["checkpoints"][os.path.basename(f)]["fingerprint"], os.path.getsize(f))
print(R["run_card"])
json.dump(R, open(os.path.join(HERE, "outputs", "fp48.json"), "w"), indent=1)
