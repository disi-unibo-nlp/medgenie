import torch
import json
import os
import logging
import os
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from prompts import get_template, append_question

# Load variables from the .env file
load_dotenv()

# Access the hugging face key
hf_key = os.getenv('HF_KEY')

# Define and parse arguments.
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "model's HF directory or local path"})
    dataset_name: Optional[str] = field(default="medqa", metadata={"help": "the dataset name", "choices":["medmcqa", "medqa", "mmlu"]})
    train_set: Optional[bool] = field(default=False, metadata={"help": "train set split is consider for context generation"})
    validation_set: Optional[bool] = field(default=False, metadata={"help": "validation set split is consider for context generation"})
    test_set: Optional[bool] = field(default=False, metadata={"help": "test set split is consider for context generation"})
    n_options: Optional[int] = field(default=4, metadata={"help": "Number of choices per question."})
    templates_dir: Optional[str] =  field(default="./templates", metadata={"help": "prompt templates directory"})
    out_preds_dir: Optional[str] =  field(default="./predictions", metadata={"help": "outputs directory"})
    out_example_prompt_dir: Optional[str] =  field(default="./examples", metadata={"help": "outputs directory for storing example prompts"})
    out_name: Optional[str] =  field(default=None, metadata={"help": "output filename"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    batch_size: Optional[int] = field(default=8, metadata={"help": "Maximum number of data to process per batch."})
    no_contexts: Optional[bool] = field(default=False, metadata={"help": "whether include or not generated contexts in the prompt"})
    max_context_window: Optional[int] = field(default=4000, metadata={"help": "Maximum context window for the input prompt."})
    max_model_len: Optional[int] = field(default=4096, metadata={"help": "Maximum context window specified by default for the selected model"})
    n_contexts: Optional[int] = field(default=5, metadata={"help": "Number of contexts given as input within the prompt."})
    test_set_path: Optional[str] =  field(default=None, metadata={"help": "input path for test data."})
    n_shots: Optional[int] = field(default=2, metadata={"help": "The number of shot used in the prompt."})


def get_accuracy(true_labels, predictions):
    """
    Calculate accuracy given the true labels and predicted labels.

    Parameters:
    - true_labels: List or array of true labels.
    - predictions: List or array of predicted labels.

    Returns:
    - Accuracy as a percentage.
    """
    if len(true_labels) != len(predictions):
        raise ValueError("Length of true_labels and predictions must be the same.")
    correct_predictions = 0
    #correct_predictions = sum(1 for true, pred in zip(true_labels, predictions) if pred in true)
    total_samples = len(true_labels)
    
    for true, pred in zip(true_labels, predictions):
        if true.lower() == pred.lower():
            correct_predictions += 1
    
    accuracy = (correct_predictions / total_samples) * 100

    return accuracy

def get_contexts(input_data):
    ctxs_final = []
    for item in input_data:
        ctxs = [ctx['text'] for ctx in item['ctxs']] 
        ctxs_final.append(ctxs)
        
    return ctxs_final

if __name__ == "__main__":
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="out.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    login(token=hf_key)
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if "pmc-llama" in args.model_name.lower():
        args.no_contexts = True

    if "llama-3" in args.model_name.lower() or "llama3" in args.model_name.lower():
        terminators = [
            tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    sampling_params = SamplingParams(
        n=1, 
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=50, 
        use_beam_search=False,
        stop_token_ids = terminators if "llama-3" in args.model_name.lower() or "llama3" in args.model_name.lower() else None,
    )
    
    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in args.model_name.lower() else "auto",
        quantization="awq" if "awq" in args.model_name.lower() else None,
        max_model_len=2048 if "pmc-llama" in args.model_name.lower() else args.max_model_len
    )

    logger.info(f"Dataset: {args.dataset_name}")
    if args.test_set_path:
        test_set = load_dataset('csv' if args.test_set_path.endswith('.csv') else 'json', data_files=args.test_set_path)
    else:
        if args.dataset_name == "medqa":
            if args.n_options == 4: 
                test_set = load_dataset('disi-unibo-nlp/medqa-MedGENIE', split="train" if args.train_set else "validation" if args.validation_set else "test")
            else:
                test_set = load_dataset('disi-unibo-nlp/medqa-5-opt-MedGENIE', split="train" if args.train_set else "validation" if args.validation_set else "test")

        if args.dataset_name == "medmcqa":
             test_set = load_dataset('disi-unibo-nlp/medmcqa-MedGENIE', split="train" if args.train_set else "validation" if args.validation_set else "test")
        
        if args.dataset_name == "mmlu":
             test_set = load_dataset('disi-unibo-nlp/mmlu-medical-MedGENIE', split="train" if args.train_set else "validation" if args.validation_set else "test")

    MAX_INDEX = len(test_set) if args.max_samples == -1 else args.max_samples
    test_set = test_set.select(range(0,MAX_INDEX))
    logger.info(f"Number of samples: {len(test_set)}")
    
    template = get_template(args)
    contexts = [[] for _ in range(len(test_set))] if args.no_contexts else get_contexts(test_set) 
    prompts = [template + append_question(args, template, item, tokenizer, contexts=contexts[i]) for i, item in enumerate(test_set)]
    batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    
    prompt_dir = args.out_example_prompt_dir
    out_model_name = f"{args.model_name.split('/')[len(args.model_name.split('/'))-1]}"
    out_dataset_name = args.dataset_name
    logger.info(out_dataset_name)

    example_path = f"{prompt_dir}/{out_model_name}/{out_dataset_name}"
    if args.out_name:
        prompt_filename = f"{example_path}/{args.out_name}.txt"
    else:
        prompt_filename = f"{example_path}/{out_dataset_name}_{args.n_options}opt_no_ctxs.txt" if args.no_contexts else f"{example_path}/{out_dataset_name}_{args.n_options}opt.txt"
  
    os.makedirs(example_path, exist_ok=True)
    with open(prompt_filename, 'w') as f:
        f.write(prompts[0])

    json_file = {
        "accuracy": 0,
        "predictions": [],
        "outputs": []
    }

    id = 0
    for batch in tqdm(batches):
        outputs = llm.generate(batch, sampling_params, use_tqdm=False)

        for out in outputs:
            prompt = out.prompt
            answer = out.outputs[0].text
            
            question = prompt.split("Question:")[args.n_shots+1].strip() if "llama-3" in args.model_name.lower() or "llama3" in args.model_name.lower() else prompt.split("### Question:")[args.n_shots+1].strip()
            if "zephyr" in args.model_name.lower():
                question = question.split("<|assistant|>")[0].strip()
            elif "llama-2" in args.model_name.lower():
                question = question.split("[/INST]")[0].strip()
            elif "llama-3" in args.model_name.lower() or "llama3" in args.model_name.lower():
                question = question.split("<|eot_id|>")[0].strip()
                answer = answer.replace("<|eot_id|>", "").strip()
                if "(" not in answer:
                    answer = "(" + answer.strip()[0] + ")"
            
            if answer.strip():
                if "pmc-llama" in args.model_name.lower():
                    pred = answer.strip()[0] 
                    answer = answer.split('\n')[0]
                else: 
                    start = answer.find('(')
                    end =  answer.find(')')
                    pred = answer[start+1:end]

            json_file['outputs'].append({
                "question": question.strip(),
                "gold": f"{test_set[id]['target']}" ,
                "answer": answer.strip()
            })

            json_file['predictions'].append(pred)
            id += 1
    
    true_labels = [item['target'] for item in test_set]
    
    if args.dataset_name == 'mmlu':
        last_index = 0
        json_file['accuracy'] = {}
        accuracy = get_accuracy(true_labels, json_file['predictions'])
        json_file['accuracy']['total'] = accuracy
        subjects = list(set([item['subject'] for item in test_set]))
        for subject in subjects:
            subset = [item for item in test_set if item['subject'] == subject]
            true_labels_subset = true_labels[last_index:last_index+len(subset)]
            accuracy = get_accuracy(true_labels_subset, json_file['predictions'][last_index:last_index+len(true_labels_subset)])
            last_index += len(true_labels_subset)
            json_file['accuracy'][subject] = accuracy
        
    else:
        accuracy = get_accuracy(true_labels, json_file['predictions'])
        json_file['accuracy'] = accuracy

    logger.info(f"Accuracy: {json_file['accuracy']}")

    pred_path = f"{args.out_preds_dir}/{out_model_name}/{out_dataset_name}"
    os.makedirs(pred_path, exist_ok=True)
    if args.out_name:
        out_filename = f"{pred_path}/{args.out_name}.json"
    else:
        out_filename = f"{pred_path}/{out_dataset_name}_{args.n_options}opt_no_ctxs.json" if args.no_contexts else f"{pred_path}/{out_dataset_name}_{args.n_options}opt.json" 
    with open(out_filename, 'w') as f:
        json.dump(json_file, f, indent=4)