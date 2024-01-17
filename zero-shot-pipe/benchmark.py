import torch
import time
import json
import os
import logging
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from prompts import get_template_medqa, get_template_medmcqa, get_template_no_ctxs, append_question

# Load variables from the .env file
load_dotenv()

# Access the hugging face key
hf_key = os.getenv('HF_KEY')

# Define and parse arguments.
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="medqa", metadata={"help": "the dataset name", "choices":["medmcqa", "medqa", "mmlu"]})
    mmlu_ctxs_path: Optional[str] = field(default=None, metadata={"help": "contexts path for specific mmlu subset"})
    mmlu_test_path: Optional[str] = field(default=None, metadata={"help": "test data path for specific mmlu subset"})
    out_preds_dir: Optional[str] =  field(default="./predictions", metadata={"help": "outputs directory"})
    out_example_prompt_dir: Optional[str] =  field(default="./examples", metadata={"help": "outputs directory for storing example prompts"})
    out_name: Optional[str] =  field(default=None, metadata={"help": "output filename"})
    ctxs_shot_dir: Optional[str] =  field(default="./shots", metadata={"help": "Directory where to find contexts for shot examples"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    batch_size: Optional[int] = field(default=8, metadata={"help": "Maximum number of data to process per batch."})
    no_contexts: Optional[bool] = field(default=False, metadata={"help": "whether include or not generated contexts in the prompt"})
    n_shots: Optional[int] = field(default=2, metadata={"help": "Number of shot demonstartions within the prompt."})
    max_context_window: Optional[int] = field(default=4000, metadata={"help": "Maximum context window for the input prompt."})

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
        start = pred.find('(')
        end =  pred.find(')')
        def_choice = pred[start+1:end]
        if len(def_choice) == 1 and def_choice.lower() == true[2].lower():
            correct_predictions += 1
    
    accuracy = (correct_predictions / total_samples) * 100

    return accuracy

def get_contexts(args):
    if args.dataset_name == "medqa":
        with open('../data/fid/medqa/TEST_FID_medqa_4op_DEF_ABCD.json') as f:
            input_data = json.load(f)
    if args.dataset_name == "medmcqa":
        with open('../data/fid/medmcqa/DEV_FID_medmcqa_5n.json') as f:
            input_data = json.load(f)
    if args.dataset_name == "mmlu":
        with open(args.mmlu_ctxs_path) as f:
            input_data = json.load(f)

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
    
    id2lbl = {
        0: "A",
        1: "B",
        2: "C",
        3: "D"
    }

    sampling_params = SamplingParams(
        n=1, 
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=100, 
        use_beam_search=False
    )

    llm = LLM(
        model='meta-llama/Llama-2-7b-chat-hf',
        gpu_memory_utilization=.95,
        max_model_len=4096
    )

    logger.info(f"Dataset: {args.dataset_name}")
    if args.dataset_name == "medqa":

        with open('../data/medqa/train/phrases_no_exclude_train.jsonl', 'r') as f:
            jsonl_content = f.read()
            train_set = [json.loads(jline) for jline in jsonl_content.splitlines()]

        with open('../data/medqa/test/phrases_no_exclude_test.jsonl', 'r') as f:
            jsonl_content = f.read()
            test_set = [json.loads(jline) for jline in jsonl_content.splitlines()]

    if args.dataset_name == "medmcqa":
        train_set = load_dataset('medmcqa', split="train")
        test_set = load_dataset('medmcqa', split="validation")
    
    if args.dataset_name == "mmlu":
        train_set = load_dataset('medmcqa', split="train")
        if "complete" in args.mmlu_test_path:
            df = pd.read_csv(args.mmlu_test_path)
        else:
            df = pd.read_csv(args.mmlu_test_path, names=["question", "opa", "opb", "opc", "opd", "cop"])
        
        label2id = {"A": 0, "B": 1, "C": 2, "D": 3}
        df['cop'] = df['cop'].map(lambda x: label2id[x])
        df.to_csv("mmlu_formatted.csv", index=False)
        test_set = load_dataset("csv", data_files={"test": "mmlu_formatted.csv"})['test']

    MAX_INDEX = len(test_set) if args.max_samples == -1 else args.max_samples
   
    sub_data = test_set[:MAX_INDEX]
    if args.dataset_name in ["medmcqa", "mmlu"]: sub_data = Dataset.from_dict(test_set[:MAX_INDEX])
    logger.info(f"Number of samples: {len(sub_data)}")
    shots = [train_set[i] for i in range(args.n_shots)]
    with open(f'{args.ctxs_shot_dir}/ctxs_{args.dataset_name}.txt') as f:
        shots_context = f.readlines()
        assert len(shots_context) >= args.n_shots
        shots_context = shots_context[:args.n_shots]
        for i, shot in enumerate(shots):
            shot['context'] = shots_context[i]
    
    if args.no_contexts:
        template = get_template_no_ctxs()
    elif args.dataset_name in ["medmcqa", "mmlu"]:
        template = get_template_medmcqa(shots)
    elif args.dataset_name == "medqa":
        template = get_template_medqa(shots)
    
    contexts = [[] for _ in range(len(sub_data))] if args.no_contexts else get_contexts(args) 
    prompts = [template + append_question(args, template, item, tokenizer, contexts=contexts[i]) for i, item in enumerate(sub_data)]
    batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    
    prompt_dir = args.out_example_prompt_dir
    os.makedirs(prompt_dir, exist_ok=True)
    out_model_name = f"{args.model_name.split('/')[len(args.model_name.split('/'))-1]}"
    out_dataset_name = args.mmlu_test_path.split('/')[len(args.mmlu_test_path.split('/'))-1].replace("_test", "").replace(".csv", "") if args.dataset_name == "mmlu" else args.dataset_name
    logger.info(out_dataset_name)
    prompt_filename = prompt_dir+f"/{out_model_name}_{out_dataset_name}_no_ctxs.txt" if args.no_contexts else prompt_dir+f"/{out_model_name}_{out_dataset_name}.txt"
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
            question = prompt.split("### Question:")[1]
            question = question.split("[/INST]")[0].strip()

            #if out.outputs[0].text.find(".") >= 0:
            answer = out.outputs[0].text
            #else:
            #    answer = out.outputs[0].text.split('\n')[0]
        
            if args.dataset_name =="medqa":
                json_file['outputs'].append({
                    "question": question.strip(),
                    "gold": f"The answer is ({test_set[id]['answer_idx']}) {test_set[id]['answer']}" ,
                    "answer" : answer.strip()
                })

            if args.dataset_name in ["medmcqa", "mmlu"]:
                json_file['outputs'].append({
                    "question": question.strip(),
                    "gold": f"The answer is ({id2lbl[test_set[id]['cop']]}) {[test_set[id]['opa'], test_set[id]['opb'], test_set[id]['opc'], test_set[id]['opd']][test_set[id]['cop']]}." ,
                    "answer" : answer.strip()
                })
            json_file['predictions'].append(answer.strip())
            id += 1
        

    if args.dataset_name == "medqa":
        true_labels = [(f"The answer is ({item['answer_idx']}) {item['answer'].strip()}", 
                        f"The answer is {item['answer'].strip()}",
                        item['answer_idx']) for item in sub_data]
    if args.dataset_name in ["medmcqa", "mmlu"]:
        true_labels = [(f"{id2lbl[item['cop']]}. {[item['opa'], item['opb'], item['opc'], item['opd']][item['cop']]}", 
                        [item['opa'], item['opb'], item['opc'], item['opd']][item['cop']], 
                        id2lbl[item['cop']]) for item in sub_data]

    accuracy = get_accuracy(true_labels, json_file['predictions'])
    json_file['accuracy'] = accuracy

    logger.info(f"Accuracy: {json_file['accuracy']}")

    os.makedirs(args.out_preds_dir, exist_ok=True)
    if args.out_name:
        out_filename = args.out_preds_dir+f"/{args.out_name}"
    else:
        out_filename = args.out_preds_dir+f"/{out_model_name}_{out_dataset_name}_no_ctxs.json" if args.no_contexts else args.out_preds_dir+f"/{out_model_name}_{out_dataset_name}.json" 
    with open(out_filename, 'w') as f:
        json.dump(json_file, f, indent=4)