import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.modeling_llada import LLaDAModelLM, LLaDABlock
from networks.layers import TimestepEmbedder, AdaLayerNormContinuous

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
def sample(model, batch, tokenizer, device, temperature=1., steps=256,
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

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()
    # set eos_id to the last position of the generated answer
    for i in range(x.shape[0]):
        x[i, prompt_len[i] + gen_length:] = eos_id

    trajectory_inputs = []
    trajectory_outputs = []
    remask_prob = None
    batch_size = x.shape[0]

    current_block = torch.zeros((x.shape[0], gen_length), device=x.device,  dtype=torch.bool)
    current_block[:, :block_length] = True
    for step in range(steps):
        # record model inputs
        trajectory_inputs.append(x.clone())
        
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
        update_index = torch.zeros_like(x).bool()
        update_index[bs_idx, prompt_len.unsqueeze(1) + samples] = True

        x0 = torch.where(update_index, pred_out, x)

        if step % steps_per_block == steps_per_block - 1:
            current_block = current_block.roll(block_length, 1)

        # record model outputs
        trajectory_outputs.append(x0.clone())
        x = x0

    responses = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)

    output_dict = {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_outputs': trajectory_outputs,
        'responses': responses,
    }
    
    return output_dict


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path= "models/LLaDOU-Math-8B/")
    model = LLaDOUModelLM.from_pretrained("models/LLaDOU-Math-8B/", trust_remote_code= True, torch_dtype = torch.bfloat16).cuda()
