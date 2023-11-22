#!/bin/bash

python3 main.py \
    --model_name pmc-llama-13b-awq \
    --dataset_name medmcqa \
    --data_path data/train/medmcqa_stratified_10percent.csv \
    --split train \
    --batch_size 8 \
    --n 2 \
    --best_of 5 \
    --temperature 0.9 \
    --frequency_penalty 1.95 \
    --top_p 1.0 \
    --max_tokens 512 \
    --use_beam_search False