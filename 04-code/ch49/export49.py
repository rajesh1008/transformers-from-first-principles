"""Chapter 49, 49.1: from checkpoint to artefact.  Usage: python3 export49.py [../ch48/ckpt/main_step2000.pt]
Writes artefact/capstone-fp32.safetensors, artefact/capstone-int8.safetensors (every 2-D table as int8 with one 32-bit scale per
output row, as Chapter 48's eval48.py), artefact/tokenizer.json, artefact/config.json and artefact/MANIFEST.json (SHA-256 of every
file, source fingerprint, software).  Then reloads each artefact and compares its logits with the checkpoint's on the running
sentence.  -> outputs/export49.json.  artefact/ is not in the code zip (the script re-creates it)."""
import os, sys, json, shutil, struct, pickletools, io
from core49 import *
from safetensors.torch import save_file
from tiny.seed import fingerprint, run_card
from tokenizers import Tokenizer
CK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "ch48", "ckpt", "main_step2000.pt")
TOKJ = os.path.join(HERE, "..", "ch48", "corpus", "tok8192.json")
ck = torch.load(CK, map_location="cpu"); m0 = Capstone(**ck["cfg"]); m0.load_state_dict(ck["state"]); m0.eval()
os.makedirs(ART, exist_ok=True); R = dict(source=os.path.basename(CK), step=ck["step"], source_bytes=os.path.getsize(CK), source_fingerprint=fingerprint(ck["state"]))
# What is in a .pt file: a zip holding a pickle program plus raw tensor storages.
import zipfile
z = zipfile.ZipFile(CK); names = z.namelist(); pk = [n for n in names if n.endswith("data.pkl")][0]
ops = {}
for op, arg, pos in pickletools.genops(io.BytesIO(z.read(pk))): ops[op.name] = ops.get(op.name, 0) + 1
R["pt"] = dict(entries=len(names), pickle_bytes=len(z.read(pk)), pickle_opcodes=sum(ops.values()), global_imports=ops.get("GLOBAL", 0) + ops.get("STACK_GLOBAL", 0), reduce_calls=ops.get("REDUCE", 0))
class _Calls:  # a harmless object whose pickle asks the loader to call os.getcwd: code, not data
    def __reduce__(self): return (os.getcwd, ())
_p = os.path.join(HERE, "outputs", "guard49.pt"); torch.save(dict(x=_Calls()), _p)
try: torch.load(_p); R["pt_guard"] = dict(default_load="ran the call")
except Exception as e: R["pt_guard"] = dict(default_load="refused", error=type(e).__name__, torch=torch.__version__)
os.remove(_p)
st = {n: p.detach().clone().contiguous() for n, p in m0.state_dict().items()}
save_file(st, os.path.join(ART, "capstone-fp32.safetensors"), metadata=dict(format="pt", step=str(ck["step"])))
q8 = {}
for n, p in st.items():
    if p.dim() == 2:
        s = p.abs().amax(1, keepdim=True) / 127.0; s[s == 0] = 1.0
        q8[n + ".q"] = torch.clamp(torch.round(p / s), -127, 127).to(torch.int8).contiguous(); q8[n + ".scale"] = s.contiguous()
    else: q8[n] = p
save_file(q8, os.path.join(ART, "capstone-int8.safetensors"), metadata=dict(format="pt", step=str(ck["step"]), quant="int8 per output row, symmetric"))
shutil.copy(TOKJ, os.path.join(ART, "tokenizer.json"))
cfg = dict(model=ck["cfg"], tokenizer="tokenizer.json", context=ck["cfg"]["ctx"], name="capstone-48", step=ck["step"])
json.dump(cfg, open(os.path.join(ART, "config.json"), "w"), indent=1)
files = ["capstone-fp32.safetensors", "capstone-int8.safetensors", "tokenizer.json", "config.json"]
man = dict(files={f: dict(bytes=os.path.getsize(os.path.join(ART, f)), sha256=sha256(os.path.join(ART, f))) for f in files},
           source=R["source"], source_fingerprint=R["source_fingerprint"], run_card=run_card())
json.dump(man, open(os.path.join(ART, "MANIFEST.json"), "w"), indent=1); R["manifest"] = man["files"]
# What is in a safetensors file: 8 bytes of header length, a JSON header, then raw bytes.
with open(os.path.join(ART, "capstone-fp32.safetensors"), "rb") as f:
    hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
R["safetensors"] = dict(header_bytes=hl, tensors=len([k for k in hdr if k != "__metadata__"]), example={k: hdr[k] for k in ("E.weight", "lnf.bias")})
tk = Tokenizer.from_file(TOKJ); ids = tk.encode("the cat sat on the mat").ids
with torch.no_grad(): ref = m0(torch.tensor([ids]))[0]
for kind in ("fp32", "int8"):
    m, _ = load_model(kind)
    with torch.no_grad(): out = m(torch.tensor([ids]))[0]
    R[kind] = dict(max_logit_diff=float((out - ref).abs().max()), same_top_token=bool((out.argmax(-1) == ref.argmax(-1)).all()),
                   top_next=tk.decode([int(out[-1].argmax())]), bytes=man["files"][f"capstone-{kind}.safetensors"]["bytes"])
for k in ("pt", "pt_guard", "safetensors", "fp32", "int8"): print(k, R[k])
print("manifest:", {f: (v["bytes"], v["sha256"][:16]) for f, v in R["manifest"].items()})
json.dump(R, open(os.path.join(HERE, "outputs", "export49.json"), "w"), indent=1)
