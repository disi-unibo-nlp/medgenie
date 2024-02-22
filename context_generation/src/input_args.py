from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

# Define and parse arguments.
@dataclass
class ScriptArguments:
   
    model_name: Optional[str] = field(default="pmc-llama-13b-awq", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="medmcqa", metadata={"help": "the dataset name", "choices":["medmcqa", "medqa", "mmlu"]})
    data_path_train: Optional[str] = field(default=None, metadata={"help": "the train dataset file path if you want to load data locally"})
    data_path_validation: Optional[str] = field(default=None, metadata={"help": "the validation dataset file path if you want to load data locally"})
    data_path_test: Optional[str] = field(default=None, metadata={"help": "the test dataset file path if you want to load data locally"})
    train_set: Optional[bool] = field(default=False, metadata={"help": "train set split is consider for context generation"})
    validation_set: Optional[bool] = field(default=False, metadata={"help": "validation set split is consider for context generation"})
    test_set: Optional[bool] = field(default=False, metadata={"help": "test set split is consider for context generation"})
    out_dir: Optional[str] =  field(default="./out", metadata={"help": "outputs directory"})
    out_name: Optional[str] =  field(default="out_contexts.json", metadata={"help": "output filename"})
    prompt_dir: Optional[str] =  field(default="./prompt", metadata={"help": "prompt templates directory"})
    cache_dir: Optional[str] =  field(default="/home/llms", metadata={"help": "cache directory"})
    max_samples_train: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    max_samples_validation: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in validation set. Default is -1 to process all data."})
    max_samples_test: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in test set. Default is -1 to process all data."})
    start_train_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first train sample to consider"})
    start_validation_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first validation sample to consider"})
    start_test_sample_idx: Optional[int] = field(default=0, metadata={"help": "Start index of first test sample to consider"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "Number of prompts per batch to process during inference"})
    saving_steps: Optional[int] = field(default=2, metadata={"help": "interval for saving model outputs"})
    no_options: Optional[bool] = field(default=False, metadata={"help": "if set to True, question options will not be included in the prompt for context generation."})


    # Sampling Parameters
    n: Optional[int] = field(default=2, metadata={"help": "Number of output sequences to return for the given prompt."})
    best_of: Optional[int] = field(default=None, metadata={"help": "Number of output sequences that are generated from the prompt. From these best_of sequences, the top n sequences are returned. best_of must be greater than or equal to n. This is treated as the beam width when use_beam_search is True. By default, best_of is set to n."})
    temperature: Optional[float] = field(default=0.9, metadata={"help": "Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling."})
    frequency_penalty: Optional[float] = field(default=0, metadata={"help": "Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": " Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens."})
    max_tokens: Optional[int] = field(default=512, metadata={"help": "maximum number of tokens to generate"})
    use_beam_search: Optional[bool] = field(default=False, metadata={"help": "Whether to use beam search instead of sampling."})

def get_parser():
    return HfArgumentParser(ScriptArguments)