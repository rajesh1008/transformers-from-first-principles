"""Two ways to pick the next word: always take the top one, or roll a weighted die.
Run from 04-code/ch01:  python3 greedy.py"""
import random
from bigram import pairs, next_word_shares

def greedy(start, n=12):
    out = [start]
    for _ in range(n):
        out.append(pairs[out[-1]].most_common(1)[0][0])   # always the single most likely word
    return " ".join(out)

def sampled(start, seed, n=12):
    rng = random.Random(seed)                            # one seed per line, printed below
    out = [start]
    for _ in range(n):
        shares = next_word_shares(out[-1])
        out.append(rng.choices(list(shares), weights=list(shares.values()))[0])
    return " ".join(out)

if __name__ == "__main__":
    print("greedy, from 'the':   ", greedy("the"))
    print("greedy, from 'rabbit':", greedy("rabbit", n=15))
    for seed in (1, 2, 3):
        print(f"sampled, seed {seed}:     ", sampled("the", seed))
