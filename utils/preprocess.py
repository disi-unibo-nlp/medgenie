import sys
import json
import os
import logging
from transformers import HfArgumentParser
from typing import Optional
from dataclasses import dataclass, field

sys.path.append('../')
from context_generation.src.utils import get_dataset_splits, get_split_info

# Define and parse arguments.
@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = field(default="medqa", metadata={"help": "The dataset name.", "choices":["medmcqa", "medqa", "mmlu"]})
    data_path_train: Optional[str] = field(default=None, metadata={"help": "The train dataset file path if you want to load data locally."})
    data_path_validation: Optional[str] = field(default=None, metadata={"help": "The validation dataset file path if you want to load data locally."})
    data_path_test: Optional[str] = field(default=None, metadata={"help": "The test dataset file path if you want to load data locally."})
    train_set: Optional[bool] = field(default=False, metadata={"help": "Train set split is consider for context generation."})
    validation_set: Optional[bool] = field(default=False, metadata={"help": "Validation set split is consider for context generation."})
    test_set: Optional[bool] = field(default=False, metadata={"help": "Test set split is consider for context generation."})
    n_options: Optional[int] = field(default=4, metadata={"help": "Number of choices per question."})
    contexts_w_ops: str = field(default=None, metadata={"help": "Path of contexts generated with options."})
    contexts_no_ops: str = field(default=None, metadata={"help": "Path of contexts generated without options."})
    n_context: Optional[int] = field(default=5, metadata={"help": "Number of total contexts used."})
    out_dir: str = field(default="./out" , metadata={"help": "Path where to save the fid formatted file."})
    max_samples_train: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    max_samples_validation: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in validation set. Default is -1 to process all data."})
    max_samples_test: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in test set. Default is -1 to process all data."})
    start_train_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first train sample to consider"})
    start_validation_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first validation sample to consider"})
    start_test_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first test sample to consider"})

def get_question_and_answer(args, dataset, id_question):
    if args.dataset_name == "medmcqa" or  args.dataset_name == "mmlu":
        opa = dataset['opa'][id_question]
        opb = dataset['opb'][id_question]
        opc = dataset['opc'][id_question]
        opd = dataset['opd'][id_question]
        question = f'{dataset["question"][id_question]}\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}'
        answer = str(dataset['cop'][id_question]).replace("0", "A").replace("1", "B").replace("2", "C").replace("3", "D")
    
    if args.dataset_name == "medqa":
        question_data = dataset[id_question]
        opa = question_data["options"]["A"]
        opb = question_data["options"]["B"]
        opc = question_data["options"]["C"]
        opd = question_data["options"]["D"]
        question = f'{question_data["question"]}\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}'
        answer = question_data["answer_idx"]
        
        if args.n_options==5:
            ope = question_data["options"]["E"]
            question+=f'\nE. {ope}'

    return question, answer



def concat_and_convert(args, logger, contexts_w_ops, contexts_no_ops, dataset, dataset_size):
    contexts_fid_format = []
    for id_question in range(dataset_size):
        
        n_context_w_ops = []
        n_context_no_ops = []
        
        if str(id_question) in contexts_w_ops:
            n_context_w_ops = contexts_w_ops[str(id_question)]["contexts"]
        
        if str(id_question) in contexts_no_ops:
            n_context_no_ops = contexts_no_ops[str(id_question)]["contexts"]
        
        ctxs = n_context_w_ops + n_context_no_ops

        if len(ctxs) < args.n_context:
            logger.info(f"Question {id_question}: {args.n_context-len(ctxs)} context(s) are missing.")
            ctxs_empty = ["" for _ in range(args.n_context-len(ctxs))]
            ctxs = ctxs + ctxs_empty
        elif len(ctxs) > args.n_context:
            logger.error(f"Question {id_question}: {len(ctxs)} total context(s) instead of {args.n_context}.")

        question_w_ops, answer = get_question_and_answer(args, dataset, id_question)
        
        contexts_fid_format.append({
            "id": id_question,
            "question": question_w_ops,
            "target": answer,
            "answers": [answer],
            "ctxs": [{"text":cont} for cont in ctxs]

        })
    
    return contexts_fid_format



if __name__ == "__main__":
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="out.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.contexts_w_ops:
        with open(args.contexts_w_ops, 'r') as file:
            contexts_w_ops = json.load(file)
    else:
        contexts_w_ops = {}

    if args.contexts_no_ops:
        with open(args.contexts_no_ops, 'r') as file:
            contexts_no_ops = json.load(file)
    else:
        contexts_no_ops = {}
    
    splits = [split for split, flag in [("train", args.train_set), ("validation", args.validation_set), ("test", args.test_set)] if flag]
    datasets = get_dataset_splits(args)
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"N. options: {args.n_options}")
    for split in splits:
        logger.info(f"Split: {split}")
        dataset, max_samples, start_idx = get_split_info(datasets, split, args) 
        data = dataset[start_idx:max_samples]
        data_size = max_samples-start_idx
        contexts_fid_format = concat_and_convert(args, logger, contexts_w_ops, contexts_no_ops, data, data_size)

        os.makedirs(args.out_dir, exist_ok=True)
        file_name = f"{args.out_dir}/FID_{split}_{args.dataset_name}_{args.n_options}op.json"
        with(open(file_name, "w")) as f:
            json.dump(contexts_fid_format, f, indent=4)
        logger.info(f"Saved: {file_name}")
        

