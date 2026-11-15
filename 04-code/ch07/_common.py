import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ch04"))   # for _gpt2, the small GPT-2 reader from Chapter 4
SENTENCE = "the cat sat on the mat"
SCRAMBLED = "the mat sat on the cat"
prompt = "The rabbit looked at"
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")

# Timings are measured once on the build machine and recorded, so that the numbers quoted in the text
# stay the numbers in the output boxes.  Run any timing script with --measure to time YOUR machine.
import json
def recorded(name, measure):
    """Return measure()'s result (a dict of plain numbers), measuring only if asked or never measured."""
    path = os.path.join(HERE, "outputs", name + ".json")
    if "--measure" in sys.argv or not os.path.exists(path):
        result = measure()
        result["measured"] = __import__("datetime").date.today().isoformat()
        json.dump(result, open(path, "w"), indent=1)
    return json.load(open(path))
