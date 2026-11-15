#!/bin/bash
# Chapter 42 tick 32 third batch: a second seed for bias balancing, and a shared expert with finer routing (exercise 9).
cd "$(dirname "$0")"
S=2000
NAME=moe4k1_bias_s2       KIND=moe E=4 K=1 HID=512 AUX=0 BIAS=0.001 SEED=2      STEPS=$S python3 moe.py > outputs/moe4k1_bias_s2.txt 2>&1 &
NAME=moe12k3_shared128    KIND=moe E=12 K=3 HID=128 AUX=0.01 SHARED=128         STEPS=$S python3 moe.py > outputs/moe12k3_shared128.txt 2>&1 &
wait
for n in moe4k1_bias_s2 moe12k3_shared128; do NAME=$n python3 analyse.py > outputs/analyse_$n.txt; done
