def make_default_prompt(sample):
    prompt = f"### Input:\n{json.dumps(sample)}\n\n### Response:\n"

    return prompt

