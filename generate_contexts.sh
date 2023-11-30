#!/bin/bash

python3 generate_contexts.py \
    --model_name pmc-llama-13b-awq \
    --dataset_name medmcqa \
    --batch_size 8 \
    --n 2 \
    --temperature 0.9 \
    --frequency_penalty 1.95 \
    --top_p 1.0 \
    --max_tokens 512 \
    --use_beam_search False \
    --data_path_train data/train/medmcqa_stratified_10-20percent.csv \
    --train_set \
    --no_options