#!/bin/bash
# Chapter 42 session 33 fourth batch: second seeds of the two shared-expert runs, a router learning-rate sweep (with and
# without the balance loss), and capacity applied during training rather than only at evaluation.
cd "$(dirname "$0")"
S=2000
NAME=moe12k3_shared128_s2 KIND=moe E=12 K=3 HID=128 AUX=0.01 SHARED=128 SEED=2      STEPS=$S python3 moe.py > outputs/moe12k3_shared128_s2.txt 2>&1 &
NAME=moe7k1_shared_s2     KIND=moe E=7 K=1 HID=256 AUX=0.01 SHARED=256 SEED=2      STEPS=$S python3 moe.py > outputs/moe7k1_shared_s2.txt 2>&1 &
NAME=moe4k1_rlr10         KIND=moe E=4 K=1 HID=512 AUX=0.01 RLR=10                 STEPS=$S python3 moe.py > outputs/moe4k1_rlr10.txt 2>&1 &
NAME=moe4k1_noaux_rlr10   KIND=moe E=4 K=1 HID=512 AUX=0 RLR=10                    STEPS=$S python3 moe.py > outputs/moe4k1_noaux_rlr10.txt 2>&1 &
NAME=moe4k1_noaux_rlr01   KIND=moe E=4 K=1 HID=512 AUX=0 RLR=0.1                   STEPS=$S python3 moe.py > outputs/moe4k1_noaux_rlr01.txt 2>&1 &
NAME=moe4k1_cap10         KIND=moe E=4 K=1 HID=512 AUX=0.01 CAPTRAIN=1.0           STEPS=$S python3 moe.py > outputs/moe4k1_cap10.txt 2>&1 &
NAME=moe4k1_cap125        KIND=moe E=4 K=1 HID=512 AUX=0.01 CAPTRAIN=1.25          STEPS=$S python3 moe.py > outputs/moe4k1_cap125.txt 2>&1 &
NAME=moe4k1_noaux_cap125  KIND=moe E=4 K=1 HID=512 AUX=0 CAPTRAIN=1.25             STEPS=$S python3 moe.py > outputs/moe4k1_noaux_cap125.txt 2>&1 &
wait
for n in moe12k3_shared128_s2 moe7k1_shared_s2 moe4k1_rlr10 moe4k1_noaux_rlr10 moe4k1_noaux_rlr01 moe4k1_cap10 moe4k1_cap125 moe4k1_noaux_cap125; do NAME=$n python3 analyse.py > outputs/analyse_$n.txt 2>&1; done
