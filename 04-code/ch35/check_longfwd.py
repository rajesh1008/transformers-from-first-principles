"""Check longfwd.token_losses against tiny/load.py's Llama.forward on 700 tokens (blocks of 512 cross a boundary)."""
from _common import *
import torch
torch.set_num_threads(1)
from tiny.load import Llama
from transformers import AutoTokenizer
from longfwd import token_losses
tk = AutoTokenizer.from_pretrained(CKPT("smollm2")); m = Llama.from_dir(CKPT("smollm2"))
ids = tk.encode(glass_text())[:700]
a = token_losses(m, ids)
lp = torch.log_softmax(m.scores(ids).double(), -1); b = -lp[torch.arange(699), torch.tensor(ids[1:])]
print(f"largest difference in per-token loss, blocked against tiny/load.py: {float((a.double() - b).abs().max()):.2e}")
