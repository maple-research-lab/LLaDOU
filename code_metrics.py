import click 
import numpy as np
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch 
from transformers import AutoTokenizer

from networks.lladou_v0 import LLaDOUModelLM, sample
from custom_humaneval.data import write_jsonl, read_problems
from custom_humaneval.evaluation import evaluate_functional_correctness



@click.command()
@click.option("--ckpt_path", type=str, default="")
@click.option('--output', type=str, default="EvalResult")
@click.option('--steps', type=int, default=256)
@click.option('--gen_length', type=int, default=256)
@click.option('--block_length', type=int, default=8)
@click.option('--task', type=str, default="HumanEval")
@click.option('--seed', type=int, default=113)
@click.option('--no_sample', type=bool, default=True)
def main(
    ckpt_path, 
    output,
    steps, 
    gen_length, 
    block_length, 
    task,
    no_sample,
    seed,
    **kwargs,
):
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank())
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
    if task=="HumanEval":    
        problems = read_problems("datasets/HumanEval.jsonl.gz")
        from evaluate.humaneval import format_HumanEval_prompt_zero_shot
        problems = format_HumanEval_prompt_zero_shot(problems)
        print(f"{task} number of problem: {len(problems)}")
        print("using format_HumanEval_prompt")
    elif task=="MBPP":
        MBPP_path = "datasets/mbpp.jsonl"
        # zero-shot if want to evaluate with the zero-shot prompt, use this
        from evaluate.mbpp import read_MBPP_test_examples
        problems = {key["task_id"]:key for key in list(
            read_MBPP_test_examples(MBPP_path)
        )}
        print("Read {} examples for evaluation over.".format(len(problems)))
        print("Using zero-shot format")

    # distributed
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    task_ids = sorted(problems.keys())
    while len(task_ids) % world_size != 0:
        task_ids.append("[PAD]")
    task_ids_for_this_rank = task_ids[rank::world_size]
    print(f"[Rank {rank}] Assigned {len(task_ids_for_this_rank)} problems")
    samples = []

    # output
    output = os.path.join(output, f"{task}-{os.environ.get('POSFIX', 'tmp')}-len{gen_length}-blk{block_length}-step{steps}")
    os.makedirs(output, exist_ok=True)
    print(f"Output to {output}")

    for task_id in tqdm(task_ids_for_this_rank, disable=rank != 0):
        if task_id == "[PAD]":
            prompt = "Please just write a program for padding the test cases."
        else:
            prompt = problems[task_id]['prompt'] # build_mbpp_instruction(problems[task_id])
        assert isinstance(prompt, str), "The prompt must be a string."
    
        batch = {
            "problems": [prompt],
        }

        inputs = sample(
            model,
            batch,
            tokenizer,
            device=device,
            inference=no_sample,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,)
        responses = tokenizer.batch_decode(inputs['trajectory_outputs'][-1][:, -gen_length:], skip_special_tokens=True)

        if task_id == "[PAD]":
            continue
        elif task=="HumanEval":
            samples.append({
                "task_id": task_id,
                "completion": responses[0],
            })
        elif task=="MBPP":
            samples.append({
                "task_id": task_id,
                "prompt": prompt,
                "completion": responses[0],
            })
        else:
            raise("Invalid task name!")

    write_jsonl(os.path.join(output, f"{task}_samples_rank{torch.distributed.get_rank()}.jsonl"), samples)

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    gathered_samples = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered_samples, samples)

    if torch.distributed.get_rank() == 0:
        saved_path = os.path.join(output, f"{task}_samples_merged.jsonl")
        merged_samples = []
        for sample_list in gathered_samples:
            merged_samples.extend(sample_list)

        write_jsonl(saved_path, merged_samples)

        if task=="MBPP":
            result = evaluate_functional_correctness(
                saved_path,
                n_workers=8,
                problem_file="datasets/mbpp_test.jsonl",
                is_mbpp= task=="MBPP",
            )
        elif task=="HumanEval":
            result = evaluate_functional_correctness(
                saved_path,
                n_workers=8,
                problem_file="datasets/HumanEval.jsonl.gz",
            )
        else:
            raise("Invalid task name!")
        print(f"{task}: \n{result}")

if __name__ == "__main__":
    main()