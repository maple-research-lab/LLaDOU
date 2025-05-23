<div align="center">

<h1>Large Language Diffusion with Ordered Unmasking (LLaDOU)</h1>
<p align="center">
<a href="https://arxiv.org/abs/2505.10446"><img src="https://img.shields.io/badge/arXiv-2505.10446-b31b1b.svg" alt="ArXiv"></a>
<a href="https://huggingface.co/maple-research-lab/LLaDOU-v0-Math"><img src="https://img.shields.io/badge/Huggingface-LLaDOU v0 Math-yellow" alt="Checkpoint"></a>
<a href="https://huggingface.co/maple-research-lab/LLaDOU-v0-Code"><img src="https://img.shields.io/badge/Huggingface-LLaDOU v0 Code-yellow" alt="Checkpoint"></a>
</p>

</div>

![Demo Generation GIF](assets/demo_generation.gif)

## Getting Started

### Inference

```python
import torch
from transformers import AutoTokenizer
from networks.lladou_v0 import LLaDOUModelLM, sample

tokenizer = AutoTokenizer.from_pretrained("models/LLaDOU-Math-8B")
tokenizer.pad_token_id = 126081
model = LLaDOUModelLM.from_pretrained(
    pretrained_model_name_or_path="models/LLaDOU-Math-8B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

### Evaluation

Prepare datasets as following:
```
```

- For evaluate on [GSM8K]() and [MATH](), please run [scripts/eval_math.sh](scripts/eval_math.sh).
- For evaluate on [MBPP]() and [HumanEval](), please run [scripts/eval_code.sh](scripts/eval_code.sh).

<div align="center"><strong>Evaluation Metrics</strong></div>
![Evaluation Metrics](assets/metrics.png)

## Citation
If this repository helps with your work, please consider giving a star ⭐ and citation 🦖:
```
@article{huang2025reinforcing,
  title={Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models},
  author={Zemin Huang and Zhiyang Chen and Zijun Wang and Tiancheng Li and Guo-Jun Qi},
  journal={arXiv preprint arXiv:2505.10446},
  year={2025}
}
```


