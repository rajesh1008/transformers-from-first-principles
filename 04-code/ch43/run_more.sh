#!/bin/bash
# Chapter 43 second batch: no warm-up, longer training, a narrower ViT, the attention/position analysis, and the
# image-to-name model trained from scratch and on top of the frozen Tiny-ViT.
cd "$(dirname "$0")"
NAME=vit_warm0   WARM=0     python3 vit.py > outputs/vit_warm0.txt 2>&1 &
NAME=vit_e100    EPOCHS=100 python3 vit.py > outputs/vit_e100.txt 2>&1 &
NAME=vit_d32     D=32       python3 vit.py > outputs/vit_d32.txt 2>&1 &
NAME=vit_p1_e100 P=1 EPOCHS=100 python3 vit.py > outputs/vit_p1_e100.txt 2>&1 &
NAME=vit python3 look.py > outputs/look_vit.txt 2>&1 &
NAME=vit_sincos python3 look.py > outputs/look_vit_sincos.txt 2>&1 &
MODE=scratch python3 mm.py > outputs/mm_scratch.txt 2>&1 &
MODE=frozen  python3 mm.py > outputs/mm_frozen.txt 2>&1 &
wait
MODE=scratch SEED=2 python3 mm.py > outputs/mm_scratch_s2.txt 2>&1 &
MODE=frozen  SEED=2 python3 mm.py > outputs/mm_frozen_s2.txt 2>&1 &
NAME=vit_p4 python3 look.py > outputs/look_vit_p4.txt 2>&1 &
wait
