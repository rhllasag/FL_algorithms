#!/usr/bin/env bash
python3  -u main.py --dataset=$1 --optimizer='fedavg_mse'  \
            --learning_rate=0.01 --num_rounds=$2 --clients_per_round=$3 \
            --eval_every=1 --batch_size=2 \
            --num_epochs=5 \
            --model='stacked_lstm' \
            --drop_percent=$4 \
