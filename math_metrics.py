import os, json
import click 
import numpy as np
from tqdm import tqdm
from typing import Sequence
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset

from networks.lladou_v0 import LLaDOUModelLM, sample
from dataloaders.collate_fn_math import collate_fn_math, extract_answer_gsm8k, collate_fn_gsm8k
from evaluate.grader import math_equal
from evaluate.parser import extract_answer



def judge_answer_MATH(answers: Sequence[str], responses: Sequence[str], counts):
    ext_ans = [extract_answer(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]

    # stat acc
    counts[1] += len(ext_ans)
    for ans, res in zip(ext_ans, ext_res):
        if math_equal(ans, res, timeout=True):
            counts[0] += 1
    
    return counts

def judge_answer_GSM8K(answers: Sequence[str], responses: Sequence[str], counts):
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]

    # stat acc
    counts[1] += len(ext_ans)
    for ans, res in zip(ext_ans, ext_res):
        if math_equal(ans, res):
            counts[0] += 1
    
    return counts


@click.command()
@click.option("--ckpt_path", type=str, default="")
@click.option('--local_data_path', type=str, default="datasets/gsm8k")
@click.option('--batch_size', type=int, default=1)
@click.option('--num_workers', type=int, default=1)
@click.option('--steps', type=int, default=256)
@click.option('--gen_length', type=int, default=256)
@click.option('--block_length', type=int, default=8)
@click.option('--task', type=str, default="gsm8k")
@click.option('--seed', type=int, default=113)
@click.option('--no_sample', type=bool, default=True)
def main(
    ckpt_path, 
    local_data_path, 
    batch_size, 
    num_workers, 
    steps, 
    gen_length, 
    block_length, 
    no_sample,
    seed,
    **kwargs,
):
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    torch.manual_seed(seed)
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.pad_token_id = 126081
    model = LLaDOUModelLM.from_pretrained(
        pretrained_model_name_or_path=ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False).to(device)
    # load data 
    if 'MATH' in local_data_path:
        ds = load_dataset(local_data_path, split='test').with_format('torch')
        task = 'MATH500' if '500' in local_data_path else 'MATH'
    elif 'gsm8k' in local_data_path:
        ds = load_dataset(local_data_path, split='test', data_dir='main').with_format('torch')
        task = 'gsm8k'
    else:
        raise ValueError(f"Invalid data path: {local_data_path}")
    sampler = DistributedSampler(ds, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=False)
    # collate_fn = {
    #     'MATH': collate_fn_math,
    #     'gsm8k': collate_fn_gsm8k,
    # }
    collate_fn = collate_fn_math if 'MATH' in local_data_path else collate_fn_gsm8k
    dl = DataLoader(
        ds, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, sampler=sampler
    )
    pbar = tqdm(dl, disable=dist.get_rank() != 0)
    counts = torch.tensor([0, 0], device=device) # correct, total

    for ix, batch in enumerate(pbar):
        answers = batch['answers']

        inputs = sample(
            model,
            batch,
            tokenizer,
            device=device,
            inference=no_sample,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,)
        responses = tokenizer.batch_decode(inputs['trajectory_outputs'][-1], skip_special_tokens=True)
        if 'MATH' in local_data_path:
            counts = judge_answer_MATH(answers, responses, counts)
        elif 'gsm8k' in local_data_path:
            counts = judge_answer_GSM8K(answers, responses, counts)

        if dist.get_rank() == 0:
            counts_list = [counts.clone() for _ in range(dist.get_world_size())]
        else:
            counts_list = None 
        # gather acc

        torch.distributed.gather(counts, counts_list, dst=0)
        if dist.get_rank() == 0:
            counts_list = torch.stack(counts_list, dim=0).sum(dim=0)
            acc = counts_list[0] / counts_list[1]
            pbar.set_description(f"acc: {acc.item() * 100:.2f}%")
    if dist.get_rank() == 0:
        print(counts_list)
        print("Final Acc: ", acc)

if __name__ == "__main__":
    main()