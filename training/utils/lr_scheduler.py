import torch 
import numpy as np


def constantlr():
    return lambda step: 1


def warmup(t_warmup: int):
    def lr_adjust(step):
        if step < t_warmup:
            return step / t_warmup + 1e-8
        else:
            return 1 
    
    return lr_adjust

def cosine_with_warmup(
    step_warmup: int = 0,
    T_max: int = 550_000,
    eta_min: float = 1e-8
):
    def scheduler(step):
        if step < step_warmup:
            return step / step_warmup
        else:
            return max(.5 * (1 + np.cos((step - step_warmup) * torch.pi / T_max)), eta_min)
        
    return scheduler

def warmup_and_decay(t_warmup: int, t_decay: int):
    assert t_warmup <= t_decay
    def lr_adjust(step):
        if step < t_warmup:
            return step / t_warmup + 1e-8
        else:
            return 1 / np.sqrt(max(step / t_decay, 1))
    
    return lr_adjust