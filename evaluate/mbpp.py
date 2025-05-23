import json


def format_MBPP_prompt_zero_shot(problem, need_code=False):
    function_name = problem["test_list"][0].split(' ')[1].split('(')[0]
    unit_tests = "\n".join(["    " + item for item in problem["test_list"]])
    # function_declaration = [item for item in problem["code"].split("\r\n")
    #                         if function_name in item and item.startswith("def ")][0]
    import ast
    first_assert = problem["test_list"][0]
    num_args = 0
    try:
        tree = ast.parse(first_assert.strip())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == function_name:
                num_args = len(node.args)
                break
    except Exception as e:
        print("Warning: AST parsing failed, fallback to 2 args.")
        num_args = 2  

    param_names = ", ".join([f"input_param_{i+1}" for i in range(num_args)])
    function_declaration = f"def {function_name}({param_names}):"
    return f"""You are an expert Python programmer. Your task is to complete the implementation of a function named `{function_name}`.

** TARGET FUNCTION **
{problem["text"]}

** UNIT TESTS **
Your code should pass unit tests like:
{unit_tests}

Here is the function to complete:
```python
{function_declaration}
    \"\"\"{problem["text"]}
    \"\"\"
```
"""


def read_MBPP_test_examples(data_path: str):

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    for i in range(10, 510):
    # for i in range(10, 30):
        ex = examples[i]
        prompt_shots = format_MBPP_prompt_zero_shot(ex)

        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_shots
        }
