#!/bin/bash

python3 generate_contexts.py \
    --model_name pmc-llama-13b-awq \
    --dataset_name medqa \
    --batch_size 8 \
    --n 2 \
    --temperature 0.9 \
    --frequency_penalty 1.95 \
    --top_p 1.0 \
    --max_tokens 512 \
    --use_beam_search False \
    --train_set \
    --validation_set \
    --test_set \
    --data_path_train data/medqa/train/phrases_no_exclude_train.jsonl \
    --data_path_test data/medqa/test/phrases_no_exclude_test.jsonl \
    --data_path_validation data/medqa/dev/phrases_no_exclude_dev.jsonl \
    --out_name medqa_pmc_llama_3n \
    --no_options
    #--max_samples_train 40 \
    #--max_samples_validation 16 \
    #--max_samples_test 8