from lib import *
tt = load(CH24); ids = torch.tensor([[5, 77, 300, 12, 900, 14, 15, 1600]])
d = (forward(tt, ids)["logits"] - tt.scores(ids)).abs().max(); print("max difference from tiny/torch_tiny.py", float(d))
