"""Chapter 44: how many characters each format asks the model to write, on the held-out pairs (mean, min, max), and the
supervised characters seen in training (batch 128 x steps x mean answer length).  Pure arithmetic on add.text; no model."""
import os, importlib, numpy as np
for dig, runs in ((2, [("direct", 2000), ("pad", 2000)]), (4, [("direct", 1500), ("pad", 1500), ("direct", 4500)])):
    for fmt, steps in runs:
        os.environ.update(DIGITS=str(dig), FORMAT=fmt); import add; importlib.reload(add)
        tr, te = add.pairs(); L = np.array([len(add.text(a, b)[1]) for a, b in te]); Ltr = np.mean([len(add.text(a, b)[1]) for a, b in tr])
        print(f"{dig} digits {fmt:6s} {steps:5d} steps: written after '=' mean {L.mean():.2f} (min {L.min()}, max {L.max()}); "
              f"sequence slots per example {add.T}; supervised characters in training {128 * steps * Ltr / 1e6:.2f} million")
