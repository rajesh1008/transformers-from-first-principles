#!/bin/bash
# Chapter 43 third batch: how accuracy falls with fewer training images, Tiny-ViT against the three baselines (same steps).
cd "$(dirname "$0")"
for f in 0.1 0.25 0.5; do
  for s in 1865 2; do NAME=vit_f${f}_s$s FRAC=$f SEED=$s python3 vit.py > outputs/vit_f${f}_s$s.txt 2>&1 & done
done
for f in 0.1 0.25 0.5; do FRAC=$f python3 baseline.py > outputs/baseline_f$f.txt 2>&1 & done
wait
