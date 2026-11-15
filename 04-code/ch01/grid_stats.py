"""How empty is the pair-count grid?  Run from 04-code/ch01:  python3 grid_stats.py"""
import sys, os
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tiny.corpus import load_text, words

tokens = words(load_text())
pairs = defaultdict(Counter)
for a, b in zip(tokens, tokens[1:]): pairs[a][b] += 1
V = len(set(tokens))
seen = sum(len(c) for c in pairs.values())
corner = ["the", "said", "alice", "queen", "mock", "turtle", "march", "hare", "white", "rabbit"]
filled = sum(1 for a in corner for b in corner if pairs[a][b])
print(f"vocabulary: {V:,} words -> grid of {V*V:,} cells")
print(f"different pairs ever seen: {seen:,} ({seen/(V*V):.3%} of the grid, about 1 cell in {V*V//seen})")
print(f"the ten-word corner in Figure 1.5: {filled} of 100 cells seen")
print(f"said->the {pairs['said']['the']}, said->alice {pairs['said']['alice']}, mock->turtle {pairs['mock']['turtle']}, march->hare {pairs['march']['hare']}")
