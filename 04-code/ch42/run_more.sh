#!/bin/bash
# Chapter 42 session 32 second batch: bias balancing, a shared expert, and second seeds.  One thread each, 2,000 steps.
cd "$(dirname "$0")"
S=2000
NAME=moe4k1_bias      KIND=moe E=4 K=1 HID=512 AUX=0 BIAS=0.001            STEPS=$S python3 moe.py > outputs/moe4k1_bias.txt 2>&1 &
NAME=moe7k1_shared    KIND=moe E=7 K=1 HID=256 AUX=0.01 SHARED=256         STEPS=$S python3 moe.py > outputs/moe7k1_shared.txt 2>&1 &
NAME=moe8k2_fine_s2   KIND=moe E=8 K=2 HID=256 AUX=0.01 SEED=2             STEPS=$S python3 moe.py > outputs/moe8k2_fine_s2.txt 2>&1 &
NAME=moe4k1_noaux_s2  KIND=moe E=4 K=1 HID=512 AUX=0 SEED=2                STEPS=$S python3 moe.py > outputs/moe4k1_noaux_s2.txt 2>&1 &
NAME=dense_wide_s2    KIND=dense HID=2048 SEED=2                           STEPS=$S python3 moe.py > outputs/dense_wide_s2.txt 2>&1 &
wait
