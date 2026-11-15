"""Chapter 25: numerical answers to the Interlude's Check Yourself and Exercises 1 to 7 (for Appendix A).
Pure arithmetic; no model is loaded.  Writes outputs/solutions.txt."""
import math, os
HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def say(s=""):
    L.append(s); print(s)
ps = [0.5, 0.1, 0.02]
s = [-math.log(p) for p in ps]; avg = sum(s) / 3; gm = math.prod(ps) ** (1 / 3)
say("Check yourself")
say(f"1. surprises {', '.join(f'{x:.4f}' for x in s)}; average {avg:.4f}; perplexity {math.exp(avg):.4f}; geometric mean {gm:.4f}; product {math.exp(avg) * gm:.6f}")
say(f"2. factor e^0.10 = {math.exp(0.10):.4f}, whatever the two losses are (only their difference enters)")
say(f"3. 3.0 / ln 2 x 0.3 = {3.0 / math.log(2) * 0.3:.4f} bits per character")
say("Exercises")
ps = [0.25, 0.05, 0.4, 0.01]; s = [-math.log(p) for p in ps]; avg = sum(s) / 4
say(f"1. loss {avg:.4f} nats = {avg / math.log(2):.4f} bits; perplexity {math.exp(avg):.4f}; share on p = 0.01: {s[3] / sum(s):.3f}")
d = math.log(64.2) - math.log(42.7)
say(f"2. change in loss ln(64.2 / 42.7) = {d:.4f} nats = {d / math.log(2):.4f} bits per token")
a = 4.1 / math.log(2) * 0.30; b = 3.4 / math.log(2) * 0.38
say(f"3. bits per character: first {a:.4f}, second {b:.4f}; the {'first' if a < b else 'second'} predicts the text better")
se = 0.4 / math.sqrt(50)
say(f"4. clustered standard error 0.4 / sqrt(50) = {se:.4f}; halving it needs four times the windows, 200")
say(f"5. confidence 0.8, accuracy {12 / 20:.2f}: gap {0.8 - 12 / 20:.2f}, overconfident")
z = (255 - 436 / 2) / (math.sqrt(436) / 2)
say(f"6. (255 - 218) / (sqrt(436) / 2) = {z:.2f} standard deviations")
say(f"7. sum scoring: right answer log p = 2 ln 0.1 = {2 * math.log(0.1):.4f} beats distractors' 3 ln 0.1 = {3 * math.log(0.1):.4f}, accuracy 1.00; "
    f"mean scoring: every choice ln 0.1 per token, a four-way tie, so accuracy 0.25 if ties are broken at random")
open(os.path.join(HERE, "outputs", "solutions.txt"), "w").write("\n".join(L) + "\n")
