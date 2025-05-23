import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, hidden_size, cond_size, eps=1e-5):
        """
        hidden_size: the size of input hidden states
        cond_size: the size of conditioning vector (e.g., temb)
        """
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size

        # Standard LayerNorm without affine parameters
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)

        # MLP to project temb into scale and shift
        self.modulation = nn.Linear(cond_size, 2 * hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for better training stability
        nn.init.xavier_uniform_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, temb):
        """
        x: (B, L, D) or (B, D) hidden states
        temb: (B, cond_size)
        """
        x_norm = self.norm(x)

        scale_shift = self.modulation(temb)  # (B, 2 * D)
        scale, shift = scale_shift.chunk(2, dim=-1)  # (B, D), (B, D)

        # In case input is (B, L, D), we need to broadcast (B, D) → (B, 1, D)
        if x.ndim == 3 and scale.ndim == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        return x_norm * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def reset_parameters(self):
        # Xavier initialization for better training stability
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        # import pdb;pdb.set_trace()
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].to() * freqs[None]
        # freqs.max().item()
        # t.max().item()
        # import pdb;pdb.set_trace()
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # import pdb;pdb.set_trace()
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # import pdb;pdb.set_trace()
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        
        # import pdb;pdb.set_trace()
        # print("t_freq has NaN?", torch.isnan(t_freq).any().item())
        # fc1 = self.mlp[0](t_freq)
        # print("fc1 has NaN?", torch.isnan(fc1).any().item())
        # act = self.mlp[1](fc1)
        # print("activation has NaN?", torch.isnan(act).any().item())
        # fc2 = self.mlp[2](act)
        # print("final t_emb has NaN?", torch.isnan(fc2).any().item())

        t_emb = self.mlp(t_freq)
        return t_emb