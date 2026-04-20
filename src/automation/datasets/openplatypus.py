def make_openplatypus_prompt(sample):
    instruction = sample["instruction"].strip()
    input_text = sample.get("input", "").strip()

    if input_text == "":
        messages = [
            {
                "role": "user",
                "content": f"{instruction}",
            }
        ]

    else:
        messages = [
            {
                "role": "user",
                "content": f"{instruction}\n{input_text}",
            }
        ]

    return messages
