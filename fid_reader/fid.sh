python3 train.py \
        --train_data data/fid/TRAIN_FID_4_CTXS_pmc_llama_20percent_medmcqa.json \
        --eval_data DEV_FID_4_CTXS_pmc-llama-13b-awq_medmcqa.json \
        --model_size base \
        --per_gpu_batch_size 2 \
        --n_context 4 \
        --name results_4ctxs_pmc_llama_BASE_20percent_medmcqa_512maxlength \
        --accumulation_steps 4 \
        --total_steps 72536 \
        --eval_freq 18134 \
        --text_maxlength 512 \
        --warmup_steps 3022
# train_pmc-llama-13b-awq_medmcqa_stratified_10percent.json
# 36553 pmc llama
# 36554 biomedgpt
# 36542 combined
# 36268 pmc 4ctxs