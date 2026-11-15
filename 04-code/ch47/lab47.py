"""Lab 47: checks on the whiteboard exercise and its rubric, from outputs/whiteboard47.json and outputs/rubric47.json
(run whiteboard47.py and rubric47.py first).  No model calls.  python3 lab47.py > outputs/lab47.txt"""
import os, json, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); O = lambda n: json.load(open(os.path.join(HERE, "outputs", n)))
W, R = O("whiteboard47.json"), O("rubric47.json"); S = R["scores"]; ok = []
def check(name, c): ok.append(bool(c)); print(f"check {name}: {'pass' if c else 'FAIL'}")
check("1, the hand count equals the tables (2,766,592)", W["hand"] == W["actual"] == 2766592)
check("2, one layer's attention by hand matches the model within 1e-6", W["diff"] < 1e-6)
check("3, the shares for the last token sum to 1", abs(sum(W["shares"]) - 1) < 1e-6)
check("4, the 7x7 table keeps only the lower triangle (28 shares)", W["nonzero"] == 28)
check("5, cache arithmetic 2 x 12 x 128 x 4 = 12,288", W["cache"] == 12288)
check("6, the rubric has five anchors, 0 to 4", sorted(int(k) for k in R["anchors"]) == [0, 1, 2, 3, 4])
check("7, breakage visible: the keyword rubric scores the jargon answer B above the best plain answer A", S["B"]["keyword"] > S["A"]["keyword"])
m = lambda k: st.mean(x for x in S[k]["anchored"] if x is not None)
check("8, the anchored grader puts both level-4 answers above the jargon answer", min(m("A"), m("F")) > m("B"))
print(f"{sum(ok)} of {len(ok)} checks pass")
