#!/bin/bash

python3 generate_contexts.py \
    --model_name disi-unibo-nlp/pmc-llama-13b-awq \
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
    --data_path_train data/medqa/train/train.jsonl \
    --data_path_test data/medqa/test/test.jsonl \
    --data_path_validation data/medqa/dev/dev.jsonl \
    --out_name medqa_pmc_llama_2n_no_options \
    --no_options