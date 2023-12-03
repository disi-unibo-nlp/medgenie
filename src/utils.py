from datasets import load_dataset
import os
import logging
import json

def setup_logger(args, split):
    logger = logging.getLogger(f"{__name__}.{split}")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

    file_handler = logging.FileHandler(f"{args.out_dir}/{split}/{args.model_name}_{args.dataset_name}.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def get_loggers(args, splits):
    loggers = {split: setup_logger(args, split) for split in splits}
    return loggers

def apply_template_medmcqa(prompt_template, question, opa=None, opb=None, opc=None, opd=None):
    if not(opa and opb and opc and opd):
        prompt = prompt_template + f"""\n\n### Question:
{question}

### Context:
"""	
    else:
        prompt = prompt_template + f"""\n\n### Question:
{question}
- {opa}
- {opb}
- {opc}
- {opd}

### Context:
"""	

    return prompt

def apply_template_medqa(prompt_template, question, options):
    if options:
        prompt = prompt_template + f"\n\n### Question:\n{question}\n- {options['A']}\n- {options['B']}\n- {options['C']}\n- {options['D']}"	
        if 'E' in options:
            prompt = prompt + f"\n- {options['E']}"
        prompt = prompt + f"\n\n### Context:"
        
    else:
        prompt = prompt_template + f"\n\n### Question:\n{question}\n\n### Context:"	

    return prompt

def clean_generated_text(args, generated_text):
    if "pmc-llama" in args.model_name.lower() or "biomedgpt" in args.model_name.lower():
        for delimiter in ["###Answer", "### Answer", "###Answers", "### Answers", "answer:", "answer is", "answers:"]:
            if delimiter.lower() in generated_text.lower():
                generated_text = generated_text.split(delimiter, 1)[0]
                break
            
        # Additional check for "### Context:"
        generated_text = generated_text.replace("### Context:", "").replace("###Context:", "")
        
    return generated_text.strip()


def get_prompts_medmcqa(template, data, no_options=False):
   

    questions = data['question']
    if no_options:
        prompts = [apply_template_medmcqa(template, question) for question in questions]
    else:
        opas = data['opa']
        opbs = data['opb']
        opcs = data['opc']
        opds = data['opd']
        prompts = [apply_template_medmcqa(template, question, opa, opb, opc, opd) for question, opa, opb, opc, opd in zip(questions, opas, opbs, opcs, opds)]

    return prompts

def get_prompts_medqa(template, data, no_options=False):
    prompts = []
    for item in data:
        question = item['question']
        options = [] if no_options else item['options']
        prompts.append(apply_template_medqa(template, question, options))
        
    return prompts

def get_dataset_splits(args):
    train_dataset, val_dataset, test_dataset = [], [], []
    if args.train_set:
        if args.dataset_name == "medmcqa":
            if args.data_path_train and "stratified" in args.data_path_train and args.data_path_train.endswith('.csv'):
                train_dataset = load_dataset('csv', data_files=args.data_path_train, split="train")
            else:
                train_dataset = load_dataset(args.dataset_name, split="train")
        if args.dataset_name == "medqa":
            with open(args.data_path_train, 'r') as f:
                jsonl_content = f.read()
                train_dataset = [json.loads(jline) for jline in jsonl_content.splitlines()]

    if args.validation_set:
        if args.dataset_name == "medmcqa":
            if args.data_path_validation and "stratified" in args.data_path_validation and args.data_path_validation.endswith('.csv'):
                val_dataset = load_dataset('csv', data_files=args.data_path_validation, split="validation")
            else:
                val_dataset = load_dataset(args.dataset_name, split="validation")
        if args.dataset_name == "medqa":
            with open(args.data_path_validation, 'r') as f:
                jsonl_content = f.read()
                val_dataset = [json.loads(jline) for jline in jsonl_content.splitlines()]

    if args.test_set:
        if args.dataset_name == "medmcqa":
            if args.data_path_test and "stratified" in args.data_path_test and args.data_path_test.endswith('.csv'):
                test_dataset = load_dataset('csv', data_files=args.data_path_test, split="test")
            else:
                test_dataset = load_dataset(args.dataset_name, split="test")
        if args.dataset_name == "medqa":
            with open(args.data_path_test, 'r') as f:
                jsonl_content = f.read()
                test_dataset = [json.loads(jline) for jline in jsonl_content.splitlines()]
    
    return train_dataset, val_dataset, test_dataset

def get_split_info(datasets, split, args):
    train_dataset, val_dataset, test_dataset = datasets
    
    dataset = (
        train_dataset if split == "train" 
        else val_dataset if split == "validation" 
        else test_dataset
    )

    max_samples = (
        args.max_samples_train if split == "train" 
        else args.max_samples_validation if split == "validation" 
        else args.max_samples_test
    )

    start_idx = (
        args.start_train_sample_idx if split == "train" 
        else args.start_validation_sample_idx if split == "validation" 
        else args.start_test_sample_idx
    )

    data_path = (
        args.data_path_train if split == "train"
        else args.data_path_validation if split == "validation" 
        else args.data_path_test
    )

    max_samples = len(dataset) if max_samples < 1 else max_samples
    #filename = os.path.basename(data_path).split('.')[0] if data_path else args.dataset_name

    return dataset, max_samples, start_idx


