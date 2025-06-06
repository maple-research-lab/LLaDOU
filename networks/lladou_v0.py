import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.modeling_llada import LLaDAModelLM, LLaDABlock, LLaDALlamaBlock
from networks.layers import TimestepEmbedder, AdaLayerNormContinuous


def reverse_cumsum(x, dim=-1):
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim=dim), [dim])
def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    elif method == 'max':
        return categorical_probs.argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")

class LLaDOUModelLM(LLaDAModelLM):
    def __init__(self, *args, **kwargs):
        # set newly assigned attributes if not already set
        if not hasattr(args[0], "num_head_layers"):
            setattr(args[0], "num_head_layers", kwargs.pop("num_head_layers", 1))
        if not hasattr(args[0], "use_mask_embeddings"):
            setattr(args[0], "use_mask_embeddings", kwargs.pop("use_mask_embeddings", True))
        if not hasattr(args[0], "use_block_embeddings"):
            setattr(args[0], "use_block_embeddings", kwargs.pop("use_block_embeddings", True))
        if not hasattr(args[0], "use_adaln"):
            setattr(args[0], "use_adaln", kwargs.pop("use_adaln", True))
        if len(kwargs):
            print("Warning, there are unused kwargs", kwargs)

        super().__init__(*args)
        self.mask_head = nn.ModuleList([
            LLaDABlock.build(i, self.model.config, self.model.alibi_cache) for i in range(self.config.num_head_layers)
        ])
        self.hidden_size = self.config.hidden_size
        if self.config.use_adaln:
            self.time_embedding = TimestepEmbedder(hidden_size=self.hidden_size)
            self.norm_out_1 = AdaLayerNormContinuous(self.hidden_size, self.hidden_size, eps=1e-4)
            self.norm_out_2 = AdaLayerNormContinuous(self.hidden_size, self.hidden_size, eps=1e-4)
        else:
            self.norm_out_2 = nn.LayerNorm(4096, eps=1e-4)
        self.mask_linear = nn.Linear(4096, 1)
        setattr(self.mask_linear, "is_last_linear", True)

        if self.config.use_mask_embeddings:
            self.mask_embedding = nn.Embedding(2, 4096)
        if self.config.use_block_embeddings:
            self.block_embedding = nn.Embedding(2, 4096)

        self.reset_dropout()

    def reset_dropout(self):
        for m in self.modules():
            # Only override for layers where behavior changes between train/eval
            if isinstance(m, (
                nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                nn.AlphaDropout,
            )):
                m.p = 0  # Force eval behavior

    def _init_weights(self, module):
        if isinstance(module, LLaDALlamaBlock) or isinstance(module, AdaLayerNormContinuous) or isinstance(module, TimestepEmbedder):
            module.reset_parameters()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.embedding_dim == 1:
                module.weight.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.trunc_normal_(module.weight, 0.02)
            nn.init.zeros_(module.bias)
        elif hasattr(module, "is_last_linear"):
            module.weight.data.normal_(0, 0.02)
            module.bias.data.zero_()

    def get_mask_prob(self, hidden_states, timestep, **kwargs):
        if self.config.use_adaln:
            temb = self.time_embedding(timestep).unsqueeze(1)
            if self.config.use_mask_embeddings:
                mask_feat = self.mask_embedding(kwargs["mask_index"].int())
                temb = temb + mask_feat
            if self.config.use_block_embeddings:
                block_feat = self.block_embedding(kwargs["current_block"].int())
                temb = temb + block_feat
            hidden_states = self.norm_out_1(hidden_states, temb)
        f = hidden_states
        for layer in self.mask_head:
            f, _ = layer(f)
        if self.config.use_adaln:
            f = self.norm_out_2(f, temb)
        else:
            f = self.norm_out_2(f)
        prob = self.mask_linear(f).squeeze(-1)
        prob = prob.float()
        return prob

    def forward(self, *args, **kwargs):
        pred_mask_prob = kwargs.pop("pred_mask_prob", False)
        if pred_mask_prob:
            return self.get_mask_prob(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)

@torch.no_grad()
def sample(model, batch, tokenizer, device, reward_fn=None, num_generations=1, repeat_times=1, temperature=1., steps=256,
    gen_length=256, block_length=8, mask_id=126336, eos_id=126081, inference=False):
    '''
    Args:
        model: Mask predictor.
        batch (or prompt): A dict that is collated, or a simple string as a propmt.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        mask_id: The toke id of [MASK] is 126336.
    '''
    if isinstance(batch, str):
        batch = {
            "problems": [batch],
        }
    if block_length is None:
        block_length = gen_length
    assert gen_length % block_length == 0
    steps_per_block = steps * block_length // gen_length

    prob_dtype = torch.float64
    problems = batch['problems']
    m = [[{"role": "user", "content": prompt}] for prompt in problems]
    prompts = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, )
    prompt = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    prompt_len = attention_mask.sum(dim=1)

    attention_mask = torch.cat([
        torch.ones((len(problems), gen_length), device=attention_mask.device, dtype=attention_mask.dtype),
        attention_mask,
    ], dim=1)
    attention_mask = attention_mask.repeat(num_generations, 1)

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()
    # set eos_id to the last position of the generated answer
    for i in range(x.shape[0]):
        x[i, prompt_len[i] + gen_length:] = eos_id

    x = x.repeat(num_generations, 1)
    prompt_len = prompt_len.repeat(num_generations)

    trajectory_inputs = []
    trajectory_outputs = []
    update_flags = []
    current_blocks = []
    sample_orders = []
    batch_size = x.shape[0]

    current_block = torch.zeros((x.shape[0], gen_length), device=x.device,  dtype=torch.bool)
    current_block[:, :block_length] = True
    for step in range(steps):
        # record model inputs
        trajectory_inputs.append(x.clone())
        current_blocks.append(current_block)
        
        mask_index = (x == mask_id)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(x, output_hidden_states=True, attention_mask=attention_mask)
        merge_hidden_states = outputs.hidden_states[-1] # + outputs.hidden_states[0]
        last_hidden_states = torch.stack([f[prompt_len[i]: prompt_len[i] + gen_length] for i, f in enumerate(merge_hidden_states)])

        logits = outputs.logits / temperature if temperature > 0. else outputs.logits
        p = F.softmax(logits.to(prob_dtype), dim=-1)
        pred_out = sample_categorical(p, 'hard' if not inference else 'max')
        pred_out = torch.where(mask_index, pred_out, x)

        timestep = torch.full(
            (last_hidden_states.shape[0],),
            float(step) / float(steps),
            device=last_hidden_states.device
        )

        mask_index = torch.stack([im[prompt_len[i]: prompt_len[i] + gen_length] for i, im in enumerate(mask_index)])
        remask_logits = model(last_hidden_states, pred_mask_prob=True, timestep=timestep, mask_index=mask_index, current_block=current_block)
        remask_logits = remask_logits.masked_fill(~mask_index, -torch.inf)
        remask_logits = remask_logits.masked_fill(~current_block, -torch.inf)
        remask_prob = remask_logits.softmax(-1)

        if inference:
            samples = remask_prob.topk(gen_length // steps).indices
        else:
            samples = torch.multinomial(remask_prob, num_samples=gen_length // steps, replacement=False)
        bs_idx = torch.arange(batch_size, dtype=samples.dtype).unsqueeze(1)
        update_flag = torch.zeros_like(remask_logits).bool()
        update_flag[bs_idx, samples] = True
        update_index = torch.zeros_like(x).bool()
        update_index[bs_idx, prompt_len.unsqueeze(1) + samples] = True
        sample_orders.append(samples)

        x0 = torch.where(update_index, pred_out, x)

        if step % steps_per_block == steps_per_block - 1:
            current_block = current_block.roll(block_length, 1)

        # record model outputs
        trajectory_outputs.append(x0.clone())
        update_flags.append(update_flag)
        x = x0

    responses = tokenizer.batch_decode(x0, skip_special_tokens=True)
    rewards = reward_fn(batch, responses, num_generations, device).float() if reward_fn is not None else torch.zeros(batch_size)

    output_dict = {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        'current_blocks': current_blocks,
        'update_flags': update_flags,
        'prompt_len': prompt_len,
        'rewards': rewards,
        'sample_orders': sample_orders,
        'attention_mask': attention_mask,
    }
    
    return output_dict


def logprob_loss(model, inputs, valid_samples, eps = .2, beta= 0.0, gain=1.0, temperature=1., accelerator=None, steps=256,
    gen_length=256, block_length=8, mask_id=126336):

    if block_length is None:
        block_length = gen_length
    assert gen_length % block_length == 0
    steps_per_block = steps * block_length // gen_length

    advantages = inputs['advantages']
    prompt_len = inputs['prompt_len']
    trajectory_inputs = inputs['trajectory_inputs']
    trajectory_outputs = inputs['trajectory_outputs']
    sample_orders = inputs['sample_orders']

    batch_size = advantages.shape[0]
    # B, L_{block}
    loss_to_log = 0.

    valid_samples = accelerator.gather(valid_samples)
    valid_samples = valid_samples.float().mean().item()

    coeffs_1 = []

    for step in range(steps):
        x = trajectory_inputs.pop(0)
        x0 = trajectory_outputs.pop(0)
        update_flag = inputs["update_flags"][step]
        mask_index = (x == mask_id)
        current_block = inputs["current_blocks"][step]
        attention_mask = inputs['attention_mask']

        timestep = torch.full(
            (batch_size,),
            float(step) / float(steps),
            device=x.device
        )
        kl_loss = torch.zeros(batch_size, device=x.device)

        mask_index = torch.stack([im[prompt_len[i]: prompt_len[i] + gen_length] for i, im in enumerate(mask_index)])
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(x, output_hidden_states=True, attention_mask=attention_mask)
            logits = outputs.logits / temperature if temperature > 0. else outputs.logits

            merge_hidden_states = outputs.hidden_states[-1]
            last_hidden_states = torch.stack([f[prompt_len[i]: prompt_len[i] + gen_length] for i, f in enumerate(merge_hidden_states)])
            remask_logits = model(last_hidden_states, pred_mask_prob=True, timestep=timestep, mask_index=mask_index, current_block=current_block)

        if step % steps_per_block != steps_per_block - 1:
            remask_prob = remask_logits.exp() # bs, gen_length
            # mask_index: full-sequence 中 为 mask token 的地方
            remask_candidates = mask_index & current_block

            remask_prob = torch.where(remask_candidates, remask_prob, 0)

            sample_order = sample_orders[step]
            # F-set sum
            sum_neg = (remask_prob * (~update_flag)).sum(-1, keepdim=True) # bs, 1
            # (bs, gen_length // steps)
            bs_idx = torch.arange(batch_size, dtype=sample_order.dtype).unsqueeze(1)
            select_remask = remask_prob[bs_idx, sample_order]
            # (bs, 1)  + (bs, gen_length // steps)
            fen_mu_log = (sum_neg + reverse_cumsum(select_remask, dim=-1)).log()
            log_probs = select_remask.log() - fen_mu_log
            log_probs = log_probs.sum(dim=-1)
        else:
            log_probs = torch.zeros(
                (batch_size,),
                device=x.device,
            )

        logit_logprob = F.log_softmax(logits.to(torch.float32), dim=-1).gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        logit_logprob = torch.stack([f[prompt_len[i]: prompt_len[i] + gen_length] for i, f in enumerate(logit_logprob)])
        # TODO: check
        # logit_prob = torch.where(mask_index & current_block, logit_prob, 1)
        logit_logprob = torch.where(update_flag, logit_logprob, 0)
        log_probs = log_probs + logit_logprob.sum(dim=-1)

        # per-step loss
        old_log_probs = log_probs.detach()
        coeff_1 = (log_probs - old_log_probs).exp()
        coeff_2 = torch.clamp(coeff_1, 1 - eps, 1 + eps)
        coeffs_1.append(coeff_1)

        reward_loss = -torch.min(
            coeff_1 * advantages,
            coeff_2 * advantages
        )

        loss = reward_loss + kl_loss

        # normalize loss by denoising steps & token length
        if accelerator is not None:
            accelerator.backward(loss.mul(gain / steps / (valid_samples + 1e-5)).sum())
        else:
            loss.mul(1. / steps / valid_samples).sum().backward()
        loss_to_log = loss_to_log + loss.mul(gain / steps / (valid_samples + 1e-5)).detach().mean().item()

    all_rewards = accelerator.gather(inputs['rewards'].detach())
    reward_mean = all_rewards.mean().item()

    length = ((x0 != mask_id).sum(-1).to(prompt_len.device).float() - prompt_len).mean().item()

    return {
        "reward": reward_mean,
        "length": length,
        "valid_samples": valid_samples,
    }


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path= "models/LLaDOU-Math-8B/")
    model = LLaDOUModelLM.from_pretrained("models/LLaDOU-Math-8B/", trust_remote_code= True, torch_dtype = torch.bfloat16).cuda()
