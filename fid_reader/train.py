# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import os
import json
import torch
import transformers
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
from src.slurm import init_distributed_mode, init_signal_handler
from src.util import init_logger, average_main, save, weighted_average, set_optim
from src.evaluation import ems
from src.model import FiDT5
from src.data import Collator, Dataset, load_data


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    progress_bar = tqdm(total=opt.total_steps, desc="Training")
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
                return_dict=False 
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em, answers = evaluate(model, eval_dataset, tokenizer, collator, opt)
                os.makedirs('./out', exist_ok=True)
                with open(f'out/preds_step_{step}.json', 'w') as f:
                    json.dump(answers, f, indent=4)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    answers = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                answers.append(ans)
                gold = dataset.get_example(idx[k])['answers']
                score = ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch, answers

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    init_distributed_mode(opt)
    init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = "google/flan-t5-" + opt.model_size
    model_class = FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    
    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = Dataset(eval_examples, opt.n_context)
    
    # use golbal rank and world size to split the eval set on multiple gpus
    t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    model = FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    model = model.to(opt.local_rank)
    optimizer, scheduler = set_optim(opt, model)
    step, best_dev_em = 0, 0.0
    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )