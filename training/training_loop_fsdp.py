# This file is modified by MAPLE research lab, based on the original code from https://github.com/NVlabs/edm

# Original code is licensed under the following license:

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import psutil
import numpy as np
import torch
import dnnlib
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch_utils import distributed as dist
from torch_utils import misc


#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    batch_size          = 512,
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    ref_network_kwargs  = {},       # Options for ref model and preconditioning.
    loss_kwargs        = {},
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    total_steps         = 200000,   # Training duration, measured in thousands of training images.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    step_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    grad_accumulation   = 1,
    lr_scheduler_kwargs = {},
    precision           = "fp16",
    resume_pt           = None,
    resume_state_dump   = None,
    resume_step         = 0,
    max_grad_norm       = 1000,
    val_ticks           = 5,
    skip_spike_grad     = 10e10,
    infer_kwargs        = {},
    tokenizer_kwargs    = {},
    activation_checkpointing = 'whole_layer',
    training_state_dir  = None,
    *args, **kwargs
):
    dist.print0(f"Useless parameters: \n {args}\n {kwargs}")
    opts = {
        "batch_size": batch_size,
        "data_loader_kwargs": data_loader_kwargs,
        "network_kwargs": network_kwargs,
        "loss_kwargs": loss_kwargs,
        "optimizer_kwargs": optimizer_kwargs,
        "seed": seed,
        "total_steps": total_steps,
        "loss_scaling": loss_scaling,
        "step_per_tick": step_per_tick,
        "snapshot_ticks": snapshot_ticks,
        "state_dump_ticks": state_dump_ticks,
        "grad_accumulation": grad_accumulation,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "precision": precision,
        "resume_pt": resume_pt,
        "resume_state_dump": resume_state_dump,
        "resume_step": resume_step,
        "max_grad_norm": max_grad_norm,
        "val_ticks": val_ticks,
        "skip_spike_grad": skip_spike_grad,
        "infer_kwargs": infer_kwargs,
        "activation_checkpointing": activation_checkpointing,
    }
    # Initialize.
    rank = dist.get_rank()
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    
    # Load dataset.
    dist.print0('Loading dataset...')
    dataloader_iterator, reward_fn = dnnlib.util.construct_class_by_name(**data_loader_kwargs)
    
    # Construct network.
    dist.print0('Constructing network...')
    model = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    # model.train().requires_grad_(True).to(device)
    model.eval().to(device)
    model_params = misc.count_parameters(model)
    
    model.model.set_activation_checkpointing(activation_checkpointing)

    # tokenizer
    tokenizer = dnnlib.util.construct_class_by_name(**tokenizer_kwargs)
    if 'gpt2' in tokenizer_kwargs.get('pretrained_model_name_or_path', ''):
        dist.print0("Adding <MASK> token to the tokenizer.")
        mask_token = "<MASK>"
        tokenizer.add_tokens([mask_token])
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        tokenizer.pad_token = tokenizer.eos_token
    elif 'llada' in tokenizer_kwargs.get('pretrained_model_name_or_path', '').lower():
        dist.print0("Setting pad_token_id to mask_token_id for LLaDA.")
        tokenizer.pad_token_id = 126336

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    optimizer = dnnlib.util.construct_class_by_name(
        params=[p for p in model.parameters() if p.requires_grad],
        **optimizer_kwargs
    )

    # Setup LR scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=dnnlib.util.construct_class_by_name(**lr_scheduler_kwargs))

    accelerator = dist.get_accelerator()
    assert accelerator is not None
    model, optimizer, dataloader_iterator, scheduler = accelerator.prepare(
       model, optimizer, dataloader_iterator, scheduler
    )

    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        dist.print0(f"Resume from {resume_state_dump}")
        accelerator.load_state(resume_state_dump)

    dataloader_iterator = iter(dataloader_iterator)
    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        print(f"Resume from step {resume_step}, skipping training data ...")
        for i in range(resume_step):
            next(dataloader_iterator)

    # Train.
    cur_tick = resume_step
    cur_nsamples = 0
    training_step = resume_step # 0 for default
    tick_start_step = training_step
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    
    dist.print0("parameters Required grad:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            dist.print0(name, p.shape)
    
    # tensorboard 
    if rank == 0:
        wandb.init(
            entity='maple',
            project="rl-discrete-diffusion",
            name=':'.join(run_dir.split('/')[-2:]),
            dir=run_dir,
            config=opts,
            mode='offline'
        )
        eval_dir = os.path.join(run_dir, 'evaluations')
        os.makedirs(eval_dir, exist_ok=True)
        text_table = wandb.Table(
            columns=['step', 'prompt', 'response'],
        )

    dist.print0(f'Training for {total_steps} steps in {precision_dtype}...')
    dist.print0(f"Model with Param: {model_params}")
    dist.print0()

    batch_total = batch_size * dist.get_world_size() * grad_accumulation * infer_kwargs['num_generations']

    while True:
        if rank == 0 and not os.path.exists(run_dir):
            raise SystemError(f'Run directory "{run_dir}" does not exist.')
        
        optimizer.zero_grad(set_to_none=True)

        all_loss_los_kwargs = []
        for round_idx in range(grad_accumulation):
        # generate data and score the completions
            with misc.ddp_sync(model, sync=round_idx == grad_accumulation - 1):
                model.eval()
                if rank == 0:
                    print("Start Sampling...")
                with torch.autocast(device_type="cuda", enabled=True, dtype=precision_dtype):
                    batch = next(dataloader_iterator)
                    inputs_chunks = []
                    for _ in range(0, infer_kwargs['repeat_times']):
                        inputs = dnnlib.util.call_func_by_name(
                            **infer_kwargs,
                            model=accelerator.unwrap_model(model),
                            tokenizer=tokenizer,
                            batch=batch,
                            reward_fn=reward_fn,
                            device=device,
                        )
                        inputs_chunks.append(inputs)
                        
                # gather inputs to get advantages and valid_samples

                rewards_list = [inputs['rewards'] for inputs in inputs_chunks]
                rewards = torch.cat(rewards_list, dim=0)

                rewards_mean = rewards.view(infer_kwargs['num_generations']*infer_kwargs['repeat_times'], -1).mean(dim=0).repeat(infer_kwargs['num_generations']*infer_kwargs['repeat_times'],)
                rewards_std = rewards.view(infer_kwargs['num_generations']*infer_kwargs['repeat_times'], -1).std(dim=0).repeat(infer_kwargs['num_generations']*infer_kwargs['repeat_times'],)
                advantages = (rewards - rewards_mean) / (rewards_std + 1e-4)

                valid_samples = (advantages != 0).sum()
                split_advantages = advantages.split(infer_kwargs['num_generations'], dim=0) 
                for chunk, adv in zip(inputs_chunks, split_advantages):
                    chunk["advantages"] = adv
                
                samples = inputs_chunks[0]['trajectory_outputs'][-1]
                accelerator.wait_for_everyone()

                if rank == 0:
                    print("Start Loss Calcing...")
                model.train()
                for inputs in inputs_chunks:
                    loss_log_kwargs = dnnlib.util.call_func_by_name(
                        **loss_kwargs, 
                        model=model, 
                        inputs=inputs, 
                        gain=loss_scaling, 
                        accelerator=accelerator, 
                        valid_samples=valid_samples,
                        steps=infer_kwargs['steps'],
                        gen_length=infer_kwargs['gen_length'],
                        block_length=infer_kwargs['block_length'],
                    )
                    all_loss_los_kwargs.append(loss_log_kwargs)

            # validation
            if cur_tick % val_ticks == 0 and rank == 0 and round_idx == 0:
                # save the inputs
                if "answers" in batch:
                    text_inputs = batch['problems'][0] + '\n\n' + batch['answers'][0]
                else:
                    text_inputs = batch['problems'][0] + '\n\n'
                text_responses = "\n***\n".join(
                    tokenizer.batch_decode(samples, skip_special_tokens=True)
                )

                text_table.add_data(str(training_step), text_inputs, text_responses)

                with open(os.path.join(eval_dir, f'evaluate_{training_step}.txt'), 'w') as f:
                    f.write(text_inputs + '\n' + '=' * 20 + '\n' + text_responses)
            # sync
            accelerator.wait_for_everyone()

            for key in list(inputs.keys()):
                del inputs[key]
            # torch.cuda.empty_cache()

        loss_log_kwargs = {
            k: sum([item[k] for item in all_loss_los_kwargs]) / len(all_loss_los_kwargs) for k in all_loss_los_kwargs[0]
        }

        # maintenance
        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

        _grad_norm = accelerator.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        grad_norm = model.get_global_grad_norm() if hasattr(model, "get_global_grad_norm") else _grad_norm
        # In some cases the grad norm may not return a float
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()
        
        scheduler.step(training_step)
        optimizer.step()

        if rank == 0:
            wandb.log({
                'lr': scheduler.get_lr()[0],
                'grad_norm': grad_norm,
                **loss_log_kwargs
            }, step=training_step)

        cur_nsamples += batch_total
        done = (training_step >= total_steps)
        training_step += 1
        # Perform maintenance tasks once per tick.
        if (not done) and (cur_tick != 0) and (training_step < tick_start_step + step_per_tick):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        
        fields = {
            'tick': cur_tick,
            'step': training_step,
            'time': dnnlib.util.format_time(tick_end_time - start_time),
            'sec-per-tick': f"{(tick_end_time - tick_start_time):<7.1f}",
            'sec-per-samples': f"{((tick_end_time - tick_start_time) / cur_nsamples):<7.2f}",
            'maintenance': f"{maintenance_time:<6.1f}",
            'resource/cpumem': f"{(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}",
            'resource/gpumem': f"{(torch.cuda.memory_allocated(device) / 2**30):<6.2f}",
            'resource/reserved': f"{(torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            'grad_norm': grad_norm,
            'lr': scheduler.get_lr()[0],
            **loss_log_kwargs,
        }

        torch.cuda.reset_peak_memory_stats()
        for key, value in fields.items():
            dist.print0(f"{key} {value}", end='\t')
        dist.print0()

        # Update logs.
        if rank == 0:
            # delete useless fields
            fields.pop('tick'); fields.pop('step'); fields.pop('time'); fields.pop('reward', None); fields.pop('gsm_reward', None); fields.pop('math_reward', None); fields.pop('loss', None)
            # convert string to float
            wandb.log({k: float(v) for k, v in fields.items()}, step=training_step)
        

        if cur_tick % snapshot_ticks == 0:
            state_dict = accelerator.get_state_dict(model)
            save_path = os.path.join(training_state_dir, f'training-state-{training_step:06d}')
            accelerator.save_state(save_path)

            if rank == 0:
                save_path = os.path.join(run_dir, f'ckpt-{training_step:06d}')
                accelerator.unwrap_model(model).save_pretrained(
                    save_path, state_dict=state_dict, safe_serialization=True
                )
        accelerator.wait_for_everyone()

        # Update state.
        cur_tick += 1
        cur_nsamples = 0
        tick_start_step = training_step
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
       
        if done:
            break
    if rank == 0:
        wandb.log({
            'text_response': text_table,
        })
        
    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
