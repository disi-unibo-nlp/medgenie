import torch
import time
import json
import logging
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.input_args import get_parser
from src.utils import clean_generated_text, get_prompts_medmcqa
from datasets import load_dataset

parser = get_parser()
args = parser.parse_args_into_dataclasses()[0]

dataset_out_name = args.data_path.split("/")[2].split(".")[0] if args.data_path else args.dataset_name
logging.basicConfig(filename=f"{args.out_dir}/{args.model_name}_{dataset_out_name}.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the LLM
sampling_params = SamplingParams(
    n=args.n, 
    best_of=args.best_of, 
    temperature=args.temperature, 
    frequency_penalty=args.frequency_penalty, 
    top_p=args.top_p, 
    max_tokens=args.max_tokens, 
    use_beam_search=args.use_beam_search
)

llm = LLM(
    model=args.model_name,
    quantization='awq',
    dtype='half',
    gpu_memory_utilization=.95,
    max_model_len=2048
)

logger.info(f"Reading data...")
if args.data_path and "stratified" in args.data_path and args.data_path.endswith('.csv'):
    dataset = load_dataset('csv', data_files=args.data_path, split=args.split)
else:
    dataset = load_dataset(args.dataset_name, split=args.split)
logger.info(f"Done!")

batch_size = args.batch_size
filename = args.model_name + "-template.txt"

with open(f"prompt/{filename}") as f:
    prompt_template = f.read().strip()

out_json = {}
fails = {}
step = 0

if "medmcqa" in args.dataset_name and "pmc-llama" in args.model_name:
    
    max_samples = len(dataset) if args.max_samples < 1 else args.max_samples
    logger.info(f"Samples considered: {max_samples}")
    start_idx = args.start_train_sample_idx
    ids = dataset['id'][start_idx:max_samples]
    data = dataset[start_idx:max_samples]
    prompts = get_prompts_medmcqa(template=prompt_template, data=data)
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    ids_batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    for batch in tqdm(batches):
        ids_batch = ids_batches[step]
        step += 1
        # Generate output based on the input prompt and sampling parameters
        start_time = time.time()
        try:
            outputs = llm.generate(batch, sampling_params, use_tqdm=False)

            for i, out in enumerate(outputs):
                prompt = out.prompt
                question = prompt.split("### Question:")[3]
                question = question.replace("### Context:", "").strip()

                generated_text_1 = clean_generated_text(args, out.outputs[0].text)
                generated_text_2 = clean_generated_text(args, out.outputs[1].text)

                out_json[ids_batch[i]] = {
                    "question": question,
                    "generated_text_1": generated_text_1,
                    "generated_text_2": generated_text_2,
                }

                if not generated_text_1.strip() and not generated_text_2.strip():
                    fails[ids_batch[i]] = question

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"BATCH {step} ---> Elapsed time: {elapsed_time} seconds")

        except AssertionError as e:
            # Custom handling for the AssertionError
            fails[ids_batch[i]] = question
            logger.exception(f"An exception occurred. BATCH {step} skipped.")
            

        # save every 2 steps
        if step % args.saving_steps == 0:
            # Read existing data from the file if it exists
            logger.info(f"Saving generated contexts at step {step}...")
            try:
                with open(f'{args.out_dir}/contexts_{args.model_name}_{dataset_out_name}.json', 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = {}

            existing_data.update(out_json)
            with open(f'{args.out_dir}/contexts_{args.model_name}_{dataset_out_name}.json', 'w') as f:
                json.dump(existing_data, f, indent=4)  
            logger.info(f"Done!")

    logger.info(f"Saving failure questions...")
    with open(f'{args.out_dir}/fails_{args.model_name}_{dataset_out_name}.json', 'w') as f:
        json.dump(fails, f, indent=4)  
    logger.info(f"Done!")