# <img src="figures/hamlet.png" alt="hamlet icon" width="30">  MedGENIE

To Generate or to Retrieve? On the Effectiveness of Artificial Contexts for Medical Open-Domain Question Answering

## Generate Context 
Briefly explanation of how to generate contexts. Using [`generate_contexts.py`](./context-generation/generate_contexts.py).

**Model parameters configuration**
```bash
python3 generate_contexts.py \
    --model_name pmc-llama-13b-awq \
    --batch_size 8 \
    --temperature 0.9 \
    --frequency_penalty 1.95 \
    --top_p 1.0 \
    --max_tokens 512 \
    --use_beam_search False \
    --dataset_name medqa \
    --n 2 \
    --train_set \
    --validation_set \
    --test_set \
```


## Reader
### 1. Fusion-in-Decoder
### 2. In-Context-Learning zero-shot

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








