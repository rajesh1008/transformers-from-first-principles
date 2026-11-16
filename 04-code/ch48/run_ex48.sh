#!/bin/sh
# Exercises 48.8, 48.11 and 48.12 (fix-all session, 16 November 2026).  CPU, one thread per process, seed 1865 unless stated.
# 48.8: a third size, six layers of width 192 (six heads of 32), at the capstone's four sweep rates, 400 steps each.
for lr in 3e-4 1e-3 3e-3 1e-2; do NAME=s6l_$lr D=192 LAYERS=6 HEADS=6 LR=$lr STEPS=400 WARMUP=40 nice python3 -u train48.py > logs/s6l_$lr.txt 2>&1 & done
# 48.11: decay branch from the main run's step-1,000 checkpoint.
nice python3 -u ex48_decay.py > logs/ex48_decay.txt 2>&1 &
# 48.12: the main run's schedule with a second seed, first 500 steps only.
NAME=seed1866 SEED=1866 LR=1e-3 STEPS=2000 STOP=500 nice python3 -u train48.py > logs/seed1866.txt 2>&1 &
wait
