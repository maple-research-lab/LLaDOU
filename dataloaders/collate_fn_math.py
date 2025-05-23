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


def extract_answer_gsm8k(answer: str):
    # find the last part starting with '#### xxx'
    return answer.split('####')[-1].strip()