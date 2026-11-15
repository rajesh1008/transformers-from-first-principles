import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ch04"))   # for _gpt2, the small GPT-2 reader from Chapter 4
SENTENCE = "the cat sat on the mat"
SCRAMBLED = "the mat sat on the cat"
prompt = "The rabbit looked at"
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")
