def format_HumanEval_prompt_zero_shot(problems):
    task_ids = list(problems.keys())
    for task_id in task_ids:
        problem = problems[task_id]
        function_name = problem['entry_point']
        prompt = problem['prompt'].rstrip()

        HumanEval_prompt = f"""You are an expert Python programmer. Your task is to complete the implementation of a function named `{function_name}`.

Here is the function to complete:
```python
{prompt}
```
"""
        problems[task_id]['prompt'] = HumanEval_prompt
    return problems

if __name__ == "__main__":
    pass