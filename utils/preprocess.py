import sys
import json
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
    path_fid_input: str = field(default="./" , metadata={"help": "Path where to save the fid formatted file."})
    max_samples_train: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    max_samples_validation: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in validation set. Default is -1 to process all data."})
    max_samples_test: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in test set. Default is -1 to process all data."})
    start_train_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first train sample to consider"})
    start_validation_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first validation sample to consider"})
    start_test_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first test sample to consider"})

def get_question_and_answer(args, question_data):
    if args.dataset_name == "medmcqa" or  args.dataset_name == "mmlu":
        opa = question_data['opa']
        opb = question_data['opb']
        opc = question_data['opc']
        opd = question_data['opd']
        question = f'{question_data["question"]}\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}'
        answer = question_data['cop']
    
    if args.dataset_name == "medqa":
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



def concat_and_convert(args, contexts_w_ops, contexts_no_ops, dataset):
    contexts_fid_format = []
    for id_question in range(len(dataset)):
        
        if str(id_question) in contexts_w_ops and str(id_question) in contexts_no_ops:
            n_context_w_ops = contexts_w_ops[str(id_question)]["contexts"]
            n_context_no_ops = contexts_no_ops[str(id_question)]["contexts"]
            ctxs = n_context_w_ops + n_context_no_ops
        else:
            ctxs = ["" for _ in range(args.n_context)]
        
        assert(len(ctxs) == args.n_context)
        
        question_w_ops, answer = get_question_and_answer(args, dataset[id_question])
        
        contexts_fid_format.append({
            "id": id_question,
            "question": question_w_ops,
            "target": answer,
            "answers": [answer],
            "ctxs": [{"text":cont} for cont in ctxs]

        })
    
    return contexts_fid_format



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    with open(args.contexts_w_ops, 'r') as file:
        contexts_w_ops = json.load(file)

    with open(args.contexts_no_ops, 'r') as file:
        contexts_no_ops = json.load(file)
    
    splits = [split for split, flag in [("train", args.train_set), ("validation", args.validation_set), ("test", args.test_set)] if flag]
    datasets = get_dataset_splits(args)

    for split in splits:
        dataset, max_samples, start_idx = get_split_info(datasets, split, args) 
        data = dataset[start_idx:max_samples]
        contexts_fid_format = concat_and_convert(args, contexts_w_ops, contexts_no_ops, data)

        with(open(f"{args.path_fid_input}/FID_{split}_{args.dataset_name}_{args.n_options}op.json", "w")) as f:
            json.dump(contexts_fid_format, f, indent=4)

