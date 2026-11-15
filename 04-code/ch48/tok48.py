"""Chapter 48, the capstone tokenizer: byte-level BPE with 8,192 tokens trained on corpus/train.txt with the Hugging Face
`tokenizers` library (0.23.3), the same algorithm as Chapter 3's tokenizer at a size Chapter 3's pure-Python trainer would take
hours to reach.  Encodes the training text, the held-out Time Machine and Through the Looking-Glass (Chapter 22's held-out text,
for comparison with Tiny in bits per character) to uint16 arrays in corpus/.  Writes corpus/tok8192.json and outputs/tok48.json."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "corpus")
sys.path.insert(0, HERE); from _common import glass_text
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
t0 = time.time()
tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
tk.train([os.path.join(C, "train.txt")], trainers.BpeTrainer(vocab_size=8192, min_frequency=2, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[]))
tk.save(os.path.join(C, "tok8192.json")); dt = time.time() - t0
R = dict(vocab=tk.get_vocab_size(), train_seconds=round(dt, 1))
texts = dict(train=open(os.path.join(C, "train.txt")).read(), heldout=open(os.path.join(C, "heldout_timemachine.txt")).read(), glass=glass_text())
for k, t in texts.items():
    ids = np.array(tk.encode(t).ids, dtype=np.uint16); ids.tofile(os.path.join(C, f"{k}.u16"))
    assert tk.decode(ids.tolist()) == t, k
    R[k] = dict(chars=len(t), bytes=len(t.encode()), tokens=int(len(ids)), bytes_per_token=round(len(t.encode()) / len(ids), 3))
    print(f"{k:8s} {len(t):>11,} characters  {len(ids):>10,} tokens  {len(t.encode())/len(ids):.3f} bytes per token  (round trip exact)")
s = "the cat sat on the mat"; e = tk.encode(s)
R["sentence"] = dict(ids=e.ids, pieces=[tk.decode([i]) for i in e.ids]); print(f"{s!r} -> {e.ids} {R['sentence']['pieces']}")
print(f"vocabulary {tk.get_vocab_size():,}; trained in about {dt:.0f} s")
json.dump(R, open(os.path.join(HERE, "outputs", "tok48.json"), "w"), indent=1)
