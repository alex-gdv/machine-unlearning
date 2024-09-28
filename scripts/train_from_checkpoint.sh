#!/bin/bash

nohup python3 train.py --experiment $1 --checkpoint_epoch $2 > logs/train_$1-checkpoint_$2.out 2>&1 &