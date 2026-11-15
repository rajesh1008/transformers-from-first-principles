#!/bin/bash
# Chapter 43: Tiny-ViT runs (each well under two minutes on one CPU thread), eight at a time.
cd "$(dirname "$0")"
r() { NAME=$1 "${@:2}" python3 vit.py > outputs/$1.txt 2>&1; }
export -f r
(
r vit                                  &
r vit_s2        env SEED=2             &
r vit_s3        env SEED=3             &
r vit_nopos     env POS=none           &
r vit_nopos_s2  env POS=none SEED=2    &
r vit_rowcol    env POS=rowcol         &
r vit_sincos    env POS=sincos2d       &
r vit_mean      env POOL=mean          &
wait
r vit_p1        env P=1                &
r vit_p4        env P=4                &
r vit_shuf      env SHUF=1             &
r vit_fixperm   env FIXPERM=1          &
r vit_rowcol_s2 env POS=rowcol SEED=2  &
r vit_sincos_s2 env POS=sincos2d SEED=2 &
r vit_mean_s2   env POOL=mean SEED=2   &
r vit_l1        env L=1                &
wait
)
for s in 1865 2 3; do SEED=$s python3 baseline.py > outputs/baseline_s$s.txt 2>&1 & done; wait
