python3 benchmark.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --dataset_name mmlu \
    --n_shots 2 \
    --batch_size 8 \
    --max_context_window 4000 \
    --no_contexts \
    --mmlu_test_path ../data/mmlu/mmlu_complete.csv \
    #--mmlu_ctxs_path ../data/fid/mmlu/MMLU_complete_fid_5n.json \