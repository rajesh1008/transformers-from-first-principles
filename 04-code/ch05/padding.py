"""Where This Bites: a padded batch that forgets to shift its position numbers.  Run: python3 padding.py"""
import _common
from _common import prompt
import numpy as np
import tiktoken
import _gpt2
from tiny.embed import cosine

enc = tiktoken.get_encoding("gpt2")
E = _gpt2.tensor("wte.weight"); P = _gpt2.tensor("wpe.weight")
ids = enc.encode(prompt)
long_ids = enc.encode("The cat sat on the mat and")
pad = len(long_ids) - len(ids)
print(f"prompt {prompt!r} -> {len(ids)} tokens {ids}; batched with a {len(long_ids)}-token prompt, so {pad} pad tokens go in front")
right = [E[t] + P[k] for k, t in enumerate(ids)]
wrong = [E[t] + P[k + pad] for k, t in enumerate(ids)]
for k, t in enumerate(ids):
    print(f"  {enc.decode([t])!r:>9}: position {k} if shifted correctly, {k + pad} if not; "
          f"cosine of the two input rows {cosine(right[k], wrong[k]):.3f}")
print(f"the first token is hit hardest: position 0's row is {np.linalg.norm(P[0]):.2f} long, position {pad}'s is {np.linalg.norm(P[pad]):.2f}")
