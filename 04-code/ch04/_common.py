import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SENTENCE = "the cat sat on the mat"
prompt = "The rabbit looked at"
HERE = os.path.dirname(os.path.abspath(__file__))
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")
