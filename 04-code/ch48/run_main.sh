#!/bin/sh
# Chapter 48 main capstone run (session 35, started about 18:33 IST): 10,050,560 learned numbers, 2,000 steps of 4,096 tokens.
NAME=main LR=1e-3 STEPS=2000 SAVE=500,1000,1500 python3 -u train48.py > logs/main.txt 2>&1
