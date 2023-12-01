python3 fid/train_reader.py \
        --train_data data/fid/TRAIN_FID_4_CTXS_pmc_llama_20percent_medmcqa.json \
        --eval_data data/fid/DEV_FID_4_CTXS_pmc-llama-13b-awq_medmcqa.json \
        --model_size base \
        --per_gpu_batch_size 6 \
        --n_context 4 \
        --name results_4ctxs_pmc_llama_base_20percent_medmcqa \
        --accumulation_steps 2 \
        --total_steps 24180 \
        --eval_freq 6045 \
        --warmup_steps 3022
# train_pmc-llama-13b-awq_medmcqa_stratified_10percent.json
# 36553 pmc llama
# 36554 biomedgpt
# 36542 combined
# 36268 pmc 4ctxs