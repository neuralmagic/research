def make_alpaca_prompt(sample):
    instruction = sample["instruction"].strip()
    input_text = sample.get("input", "").strip()
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text if input_text else 'N/A'}\n\n"
        f"### Response:\n"
    )

    return prompt
