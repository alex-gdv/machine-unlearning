#!/bin/bash

nohup python3 train.py --experiment $1 > logs/train_$1.out 2>&1 &