<h1 align="center"> <img src="figures/hamlet.png" alt="hamlet icon" width="30">  MedGENIE </h1>
<h2 align="center"> To Generate or to Retrieve? <br>  On the Effectiveness of Artificial Contexts for Medical Open-Domain Question Answering  </h2>

Official source code of **MedGENIE**, the first generate-then-read framework for multiple-choice question answering in medicine. This method generates relevant information through domain-specific models before answering questions, outperforming traditional retrieval-based approaches. Tested on MedQA-USMLE, MedMCQA, and MMLU datasets within a 24GB VRAM limit, **MedGENIE** sets new benchmarks, proving that generated contexts can significantly enhance accuracy in medical question answering. 

## Generate Context 
Briefly explanation of how to generate contexts, using [`generate_contexts.py`](./context-generation/generate_contexts.py):

* **Model parameters configuration**
```bash
cd context_generation
python3 generate_contexts.py \
    --model_name pmc-llama-13b-awq \
    --batch_size 8 \
    --temperature 0.9 \
    --frequency_penalty 1.95 \
    --top_p 1.0 \
    --max_tokens 512 \
    --use_beam_search False \
```

* **Dataset information**
```bash
    --dataset_name medqa \
    --train_set \
    --validation_set \
    --test_set \
```

* **Number of contexts**
```bash
    --n 2 \
```

* **NOT to include options in the question** (by default, the options are included)
```bash
    --no_options \
```

## Reader
### Input file format
After the context generation is necessary to concatenate and convert all contexts into a single input file for the readers. <br/> For conversion use [`preprocess.py`](./utils/preprocess.py) as follow:
```bash
cd utils
python3 preprocess.py \
    --dataset_name medqa \
    --test_set \
    --data_path_test path_to_test_set \
    --contexts_w_ops path_to_generated_contexts_w_ops \
    --contexts_no_ops path_to_generated_contexts_no_ops \
```

### 1. Fusion-in-Decoder (FiD)
#### Train
The first step in utilizing FiD as a reader is to train the model:
```bash
cd fid_reader
python3 train.py \
    --train_data train_data.json \
    --eval_data eval_data.json \
    --model_size base \
    --per_gpu_batch_size 2 \
    --accumulation_steps 4 \
    --total_steps number_of_total_steps \
```
* **Contexts information**
```bash
    --n_context 5 \
    --text_maxlength 512 \
```
#### Test
Then, it is possible to evaluate the trained model:
```bash
cd fid_reader
python3 test.py \
    --model_path checkpoint_dir/my_experiment/my_model_dir/checkpoint/best_dev \
    --eval_data eval_data.json \
    --per_gpu_batch_size 2 \
    --n_context 5 \
```

### 2. In-Context-Learning zero-shot
```bash
cd icl_reader
python3 benchmark.py \
    --model_name HuggingFaceH4/zephyr-7b-beta \
    --dataset_name medqa \
    --n_options 4 \
    --batch_size 8 \
    --max_context_window 4000 \
```
* It is possible to specify **whether to use the contexts or not** (by default, contexts are used).
```bash
    --no_contexts \
```


## Main accuracy results
| Model | Ground (Source) | Learning | Params | MedQA | MedMCQA | MMLU | AVG (&darr;) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MEDISTRON | Ø | Fine-tuned | 7B | 52.0 | 59.2 | 55.6 | 55.6 |
| PMC-LLaMA | Ø | Fine-tuned | 7B | 49.2 | 51.4 | 59.7 | 53.4 |
| LLaMA-2 | Ø | Fine-tuned | 7B | 49.6 | 54.4 | 56.3 | 53.4 |
| Zephyr-β | Ø | 2-shot | 7B | 49.3 | 43.4 | 60.7 | 51.1 |
| Mistral-Instruct | Ø | 3-shot | 7B | 41.1 | 40.2 | 55.8 | 45.7 |
| LLaMA-2-chat | Ø | 2-shot | 7B | 36.9 | 35.0 | 49.3 | 40.4 |
| Codex | Ø | 0-shot | 175B | 52.5 | 50.9 | - | - |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **MedGENIE-Zephyr-β** | G (PMC-LLaMA) | 2-shot | 7B | 59.7 <span style="color:green">(+10.4)</span> | 51.0 <span style="color:green">(+7.6)</span> | 66.1 <span style="color:green">(+5.4)</span> | 58.9 <span style="color:green">(+7.8)</span> |
| **MedGENIE-FID-Flan-T5** | G (PMC-LLaMA) | Fine-tuned | 250M | 53.1 | 52.1 | 59.9 | 55.0 |
| Zephyr-β | R (MedWiki) | 2-shot | 7B | 50.5 | 47.0 | 66.9 | 54.8 |
| VOD | R (MedWiki) | Fine-tuned | 220M | 45.8 | 58.3 | 56.8 | 53.6 |
| **MedGENIE-LLaMA-2-chat** | G (PMC-LLaMA) | 2-shot | 7B | 52.6 <span style="color:green">(+15.7)</span> | 44.8 <span style="color:green">(+9.8)</span> | 58.8 <span style="color:green">(+9.5)</span> | 52.1 <span style="color:green">(+11.7)</span> |
| Mistral-Instruct | R (MedWiki) | 2-shot | 7B | 45.1 | 44.3 | 58.5 | 49.3 |
| LLaMA-2-chat | R (MedWiki) | 2-shot | 7B | 37.2 | 37.2 | 52.0 | 42.1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human (passing score) | | | | 60.0 | 50.0 | - | - |
| Human (expert score) | | | | 87.0 | 90.0 | 89.8 | 89.8 |








