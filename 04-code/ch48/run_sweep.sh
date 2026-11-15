#!/bin/sh
# Chapter 48 learning-rate sweeps (tick 35): 400 steps each, warm-up 40, WSD decay over the last fifth; one thread per process.
for lr in 3e-4 1e-3 3e-3 1e-2; do NAME=s10m_$lr LR=$lr STEPS=400 WARMUP=40 python3 -u train48.py > logs/s10m_$lr.txt 2>&1 & done
for lr in 1e-3 3e-3 1e-2 3e-2; do NAME=s2m_$lr D=128 LAYERS=4 HEADS=4 LR=$lr STEPS=400 WARMUP=40 python3 -u train48.py > logs/s2m_$lr.txt 2>&1; done
wait
