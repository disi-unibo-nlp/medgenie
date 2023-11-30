import torch
import time
import json
import logging
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.input_args import get_parser
from src.utils import clean_generated_text, get_prompts_medmcqa, get_split_info, get_dataset_splits

def main(args, logger):

    splits = [split for split, flag in [("train", args.train_set), ("validation", args.validation_set), ("test", args.test_set)] if flag]
    logger.info(f"Splits processed: {splits}")

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

    datasets = get_dataset_splits(args)
    batch_size = args.batch_size
    filename_template = args.model_name + "-template-no-options.txt" if args.no_options else args.model_name + "-template.txt"

    logger.info(f"Reading template...")
    with open(f"prompt/{filename_template}") as f:
        prompt_template = f.read().strip()
    logger.info(f"Done!")

    last_index_saved = {}
    for split in splits:
        last_index_saved[split] = 0
        out_json = {}
        fails = {}
        step = 0
        logger.info(f"Split: {split}")
        dataset, max_samples, start_idx, data_path, filename = get_split_info(datasets, split, args)
        data = dataset[start_idx:max_samples]
        logger.info(f"Start index: {start_idx}\nMax samples index: {max_samples}\nSamples considered: {max_samples-start_idx}")

        if "medmcqa" in args.dataset_name and args.model_name in ["pmc-llama-13b-awq","BioMedGPT-LM-7B-awq"]:
            
            ids = dataset['id'][start_idx:max_samples]
            prompts = get_prompts_medmcqa(template=prompt_template, data=data, no_options=args.no_options)
            logger.info(f"First prompt example:\n{prompts[0]}")
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
                    
                        if "pmc-llama" in args.model_name.lower():
                            question = prompt.split("### Question:")[3]
                        if "biomedgpt" in args.model_name.lower():
                            question = prompt.split("### Question:")[1]

                        question = question.replace("### Context:", "").strip()
                        generated_text_1 = clean_generated_text(args, out.outputs[0].text)
                        generated_text_2 = clean_generated_text(args, out.outputs[1].text)
                    
                        if not generated_text_1.strip() and not generated_text_2.strip():
                            fails[ids_batch[i]] = question
                        else:
                            out_json[ids_batch[i]] = {
                                "question": question,
                                "generated_text_1": generated_text_1,
                                "generated_text_2": generated_text_2,
                            }

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    logger.info(f"BATCH {step} ---> Elapsed time: {elapsed_time} seconds")

                except AssertionError as e:
                    # Custom handling for the AssertionError
                    logger.error(f"An exception occurred. BATCH {step} skipped.")
                    break 

                # save every 2 steps
                if step % args.saving_steps == 0:
                    # Read existing data from the file if it exists
                    logger.info(f"Saving generated contexts at step {step}...")
                    try:
                        with open(f'{args.out_dir}/{split}/contexts_{args.model_name}_{filename}.json', 'r') as f:
                            existing_data = json.load(f)
                    except FileNotFoundError:
                        existing_data = {}

                    existing_data.update(out_json)
                    with open(f'{args.out_dir}/{split}/contexts_{args.model_name}_{filename}.json', 'w') as f:
                        json.dump(existing_data, f, indent=4)  
                    logger.info(f"Done!")
                    last_index_saved[split] += args.saving_steps * batch_size
                    logger.info(f"Last index saved: {last_index_saved[split]}")
                    

            logger.info(f"Saving failure questions...")
            with open(f'{args.out_dir}/{split}/fails_{args.model_name}_{filename}.json', 'w') as f:
                json.dump(fails, f, indent=4)  
            logger.info(f"Done!")
            logger.info(f"{split.upper()} SET FULL PROCESSED!")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args_into_dataclasses()[0]

    log_file_path= f"{args.out_dir}/{args.model_name}_{args.dataset_name}.log"

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=log_file_path,
                        filemode='w')

    logger = logging.getLogger(__name__)

    main(args, logger)