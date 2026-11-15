#!/bin/bash
# Chapter 42 training runs (session 32): eight runs in parallel, one thread each, 2,000 steps.
cd "$(dirname "$0")"
S=2000
NAME=dense            KIND=dense HID=512                 STEPS=$S python3 moe.py > outputs/dense.txt 2>&1 &
NAME=dense_wide       KIND=dense HID=2048                STEPS=$S python3 moe.py > outputs/dense_wide.txt 2>&1 &
NAME=moe4k1           KIND=moe E=4 K=1 HID=512 AUX=0.01  STEPS=$S python3 moe.py > outputs/moe4k1.txt 2>&1 &
NAME=moe4k1_noaux     KIND=moe E=4 K=1 HID=512 AUX=0     STEPS=$S python3 moe.py > outputs/moe4k1_noaux.txt 2>&1 &
NAME=moe8k2_fine      KIND=moe E=8 K=2 HID=256 AUX=0.01  STEPS=$S python3 moe.py > outputs/moe8k2_fine.txt 2>&1 &
NAME=moe8k1           KIND=moe E=8 K=1 HID=512 AUX=0.01  STEPS=$S python3 moe.py > outputs/moe8k1.txt 2>&1 &
NAME=dense_s2         KIND=dense HID=512 SEED=2          STEPS=$S python3 moe.py > outputs/dense_s2.txt 2>&1 &
NAME=moe4k1_s2        KIND=moe E=4 K=1 HID=512 AUX=0.01 SEED=2 STEPS=$S python3 moe.py > outputs/moe4k1_s2.txt 2>&1 &
wait
