import os 
import torch 
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch_utils.distributed import get_rank, get_world_size
from dataloaders.sampler import InfiniteSampler
from evaluate.grader import math_equal
from evaluate.parser import extract_answer, parse_ground_truth



def collate_fn_math(batch,):
    problems = []
    answers = []
    levels = []
    instruct = r"(Please put the final answer in \boxed{} tag, i.e. $\boxed{answer here}$)"
    for item in batch:
        problems.append(item['problem'] + instruct)
        answers.append(item['solution'])
        levels.append(item['level'])
    
    return {
        "problems": problems, 
        "answers": answers,
        "levels": levels,
    }
    

def collate_fn_gsm8k(batch,):
    problems = []
    answers = []
    for item in batch:
        problems.append(item['question'])
        answers.append(item['answer'])

    return {
        "problems": problems, 
        "answers": answers
    }


def try_get_level(level: str, default: int = 5):
    try:
        return int(level.split()[-1])
    except:
        return default


def reward_MATH(
    batch, responses, num_generations, device
):
    answers = batch['answers'] * num_generations
    # answer rewards
    ext_ans = [extract_answer(ans) for ans in answers]
    ext_res = [parse_ground_truth(res)[1] for res in responses]
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if math_equal(ans, res, timeout=True):
            rewards[i] += 1.0
        else:
            rewards[i] -= 1.0

    return rewards


def load_math_dataset_and_reward(
    local_path: str,
    batch_size: int,
    split: str = 'train', 
    num_workers: int = 8,
    max_level: int = None,
    only_level: int = None,
    max_rows: int = 1e8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 112,
):
    ds = load_dataset(local_path, split=split)
    # level <= 2: ~1344
    if max_level is not None:
        ds = ds.filter(lambda x: try_get_level(x['level'], 5) <= max_level)
    if only_level is not None:
        ds = ds.filter(lambda x: try_get_level(x['level'], 5) == only_level)
    ds = ds.select(range(min(len(ds), max_rows)))
    ds = ds.filter(lambda x: len(x.get('problem', [])) > 0 and len(x.get('problem', '')) < 1500)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(
            ds, rank=rank, num_replicas=num_replicas, 
        )
    else:
        sampler = InfiniteSampler(
            ds, rank=get_rank(), num_replicas=get_world_size(), 
        )
    
    dl = DataLoader(
        ds, collate_fn=collate_fn_math,
        batch_size=batch_size, sampler=sampler, 
        num_workers=num_workers, pin_memory=True, 
    )
    
    return dl, reward_MATH


def extract_answer_gsm8k(answer: str):
    # find the last part starting with '#### xxx'
    return answer.split('####')[-1].strip()


def reward_gsm8k(
    batch, responses, num_generations, device
):
    answers = batch['answers'] * num_generations
    # answer rewards
    ext_ans = [extract_answer_gsm8k(ans) for ans in answers]
    ext_res = [extract_answer(res) for res in responses]
    rewards = torch.zeros(len(answers), device=device)
    for i, (ans, res) in enumerate(zip(ext_ans, ext_res)):
        if math_equal(ans, res):
            rewards[i] += 1.0
        else:
            rewards[i] -= 1.0

    return rewards

def load_gsm8k_dataset_and_reward(
    local_path: str,
    batch_size: int,
    split: str = 'train', 
    num_workers: int = 8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 112, 
):
    ds = load_dataset(local_path, split=split, data_dir='main')
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(
            ds, rank=rank, num_replicas=num_replicas, 
        )
    else:
        sampler = InfiniteSampler(
            ds, rank=get_rank(), num_replicas=get_world_size(), 
        )

    dl = DataLoader(
        ds, collate_fn=collate_fn_gsm8k,
        batch_size=batch_size, sampler=sampler, 
        num_workers=num_workers, pin_memory=True, 
    )

    return dl, reward_gsm8k
