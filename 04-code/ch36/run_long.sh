#!/bin/bash
# session 28: longer continuation runs for 36.3 (three processes side by side; wall times are not speed measurements)
cd "$(dirname "$0")"
STEPS=4000 LR=1e-3 PLEN=32 EVERY=250 TAG=_long32 python3 -u continue_repeat.py repeat > outputs/continue_repeat_long32.txt 2> outputs/continue_repeat_long32.err &
STEPS=4000 LR=1e-3 PLEN=64 EVERY=250 TAG=_long64 python3 -u continue_repeat.py repeat > outputs/continue_repeat_long64.txt 2> outputs/continue_repeat_long64.err &
STEPS=4000 LR=1e-3 EVERY=250 TAG=_long python3 -u continue_repeat.py plain > outputs/continue_plain_long.txt 2> outputs/continue_plain_long.err &
wait
